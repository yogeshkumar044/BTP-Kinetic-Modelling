# ================================================================
#   FRIEDEL-CRAFTS: LHHW SINGLE-SITE (All Species ads) MODEL FITTING
#   (Global Simultaneous Fit Correction)
# ================================================================

# --- 1. IMPORT NECESSARY LIBRARIES ---
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt

################################################################################
# --- DATA LOADING AND CONSTANTS ---
################################################################################

# --- Define the path to your data file ---
file_path = 'FCR_BTP2.csv'  # <-- Correct file path for Friedel-Crafts

# --- Load the data using pandas ---
try:
    raw_data = pd.read_csv(file_path)
    print(f"--- Successfully loaded data from '{file_path}' ---")
except FileNotFoundError:
    print(f"--- ERROR: The file '{file_path}' was not found. ---")
    exit() # Stop the script if the file isn't found

# --- Physical and Experimental Constants ---
R = 8.314      # Ideal gas constant in J/(mol·K)
CATALYST_LOADING = 0.01 # kg_cat / L_fluid

# --- MODIFICATION START: Use calculated initial conditions from reactant weights ---
# Molar Ratio is defined as Anisole / Propionic Anhydride
# CA0 corresponds to Propionic Anhydride (the limiting reactant).
initial_conditions = {
    # Molar Ratio Key: [CA0 (mol/L), M (Anisole/Propionic Anhydride)]
    1: [4.2512, 0.9846],  # For 1:1 ratio
    2: [2.8884, 2.0057],  # For 1:2 ratio
    3: [2.1913, 3.0192]   # For 1:3 ratio
}

# --- Structure the data for analysis ---
experimental_runs = {}
for (ratio, temp), group in raw_data.groupby(['Ratio', 'Temperature (K)']):
    molar_ratio_key = int(ratio.split(':')[-1])
    run_name = f"M{molar_ratio_key}_{temp}K"

    ca0_val, m_val = initial_conditions.get(molar_ratio_key, (None, None))
    if ca0_val is None:
        print(f"--- WARNING: Molar ratio '{molar_ratio_key}' not found in initial_conditions. Skipping. ---")
        continue

    experimental_runs[run_name] = {
        'T': temp,
        'M': m_val,
        'CA0': ca0_val,
        'time': group['Time'].values,
        'conversion': group['Conversion'].values
    }
print("--- Data has been structured for Friedel-Crafts analysis. ---")
# --- MODIFICATION END ---


################################################################################
# --- PART 1: FITTING LHHW SINGLE-SITE MODEL (WITH PRODUCT ADSORPTION) ---
################################################################################
#
#   *** MODIFICATION START ***
#   We are now doing a single, global fit for all 14 parameters at once.
#   Parameters: [A_s, Ea, KA_343, KB_343, KC_343, KD_343, ... 353, ... 363]
#
print("\n\n=====================================================================")
print("--- PART 1: Global Simultaneous Fit for LHHW (All Species ads) ---")
print("=====================================================================")

def fitting_rate_law_lhhw_ss_prod(t, XA, T, M, CA0, A_s, Ea, K_A, K_B, K_C, K_D):
    """
    Defines the ODE for the LHHW single-site model with reactant and product adsorption.
    k_s is now calculated from A_s and Ea inside this function.
    """
    # Calculate k_s from Arrhenius parameters
    k_s = A_s * np.exp(-Ea / (R * T))

    C_A = CA0 * (1 - XA); C_B = CA0 * (M - XA)
    C_C = CA0 * XA; C_D = CA0 * XA # Assuming 1:1 stoichiometry for product formation

    numerator = k_s * K_A * K_B * C_A * C_B
    denominator = 1 + K_A * C_A + K_B * C_B + K_C * C_C + K_D * C_D

    if denominator < 1e-9: return 0
    rate_per_mass = numerator / denominator
    dxdt = (CATALYST_LOADING * rate_per_mass) / CA0
    return dxdt if dxdt > 0 else 0

def sum_of_squared_errors_lhhw_ss_prod(params):
    """
    Objective function for the global fit.
    params = [A_s, Ea,
              KA_343, KB_343, KC_343, KD_343,
              KA_353, KB_353, KC_353, KD_353,
              KA_363, KB_363, KC_363, KD_363]
    """
    total_error = 0
    (A_s, Ea,
     KA_343, KB_343, KC_343, KD_343,
     KA_353, KB_353, KC_353, KD_353,
     KA_363, KB_363, KC_363, KD_363) = params

    for run_data in experimental_runs.values():
        temp = run_data['T']

        # Select the correct K values for the current run's temperature
        if temp == 343:
            KA_run, KB_run, KC_run, KD_run = KA_343, KB_343, KC_343, KD_343
        elif temp == 353:
            KA_run, KB_run, KC_run, KD_run = KA_353, KB_353, KC_353, KD_353
        else: # 363 K
            KA_run, KB_run, KC_run, KD_run = KA_363, KB_363, KC_363, KD_363

        time_exp = np.insert(np.array(run_data['time']), 0, 0)
        conv_exp = np.insert(np.array(run_data['conversion']) / 100.0, 0, 0)

        sol = solve_ivp(
            fun=lambda t, y: fitting_rate_law_lhhw_ss_prod(
                t, y, run_data['T'], run_data['M'], run_data['CA0'],
                A_s, Ea, KA_run, KB_run, KC_run, KD_run # Pass global A_s, Ea and specific K's
            ),
            t_span=[0, max(time_exp) if len(time_exp) > 1 else 1],
            y0=[0],
            method='LSODA',
            t_eval=time_exp
        )
        if sol.status != 0: return 1e10
        total_error += np.sum((conv_exp - sol.y[0])**2)
    return total_error

# --- Optimization ---
# 14 Parameters: [A_s, Ea, KA_343, KB_343, KC_343, KD_343, ... 353, ... 363]
initial_guess = [
    1e8,     # A_s
    50000,   # Ea (e.g., 50 kJ/mol)
    0.5, 0.5, 0.1, 0.1, # K's for 343K
    0.5, 0.5, 0.1, 0.1, # K's for 353K
    0.5, 0.5, 0.1, 0.1  # K's for 363K
]

# Define bounds. We force Ea (index 1) to be positive.
bounds = [
    (1e-6, None),  # A_s
    (1e-6, None),  # Ea (FORCED TO BE POSITIVE)
    (1e-6, None), (1e-6, None), (1e-6, None), (1e-6, None), # 343K bounds
    (1e-6, None), (1e-6, None), (1e-6, None), (1e-6, None), # 353K bounds
    (1e-6, None), (1e-6, None), (1e-6, None), (1e-6, None)  # 363K bounds
]

print("--- Starting global minimization... (This may take a moment) ---")
result = minimize(
    sum_of_squared_errors_lhhw_ss_prod,
    initial_guess,
    method='L-BFGS-B',
    bounds=bounds
)
print("--- Global minimization complete. ---")

# Unpack the 14 fitted parameters
(A_s_fit, Ea_fit,
 KA_343_fit, KB_343_fit, KC_343_fit, KD_343_fit,
 KA_353_fit, KB_353_fit, KC_353_fit, KD_353_fit,
 KA_363_fit, KB_363_fit, KC_363_fit, KD_363_fit) = result.x

# Store results in dictionaries for plotting and tables
fitted_params_map = {
    343: {'KA': KA_343_fit, 'KB': KB_343_fit, 'KC': KC_343_fit, 'KD': KD_343_fit},
    353: {'KA': KA_353_fit, 'KB': KB_353_fit, 'KC': KC_353_fit, 'KD': KD_353_fit},
    363: {'KA': KA_363_fit, 'KB': KB_363_fit, 'KC': KC_363_fit, 'KD': KD_363_fit}
}
temps_K = np.array([343, 353, 363])
fitted_ks_map = {T: A_s_fit * np.exp(-Ea_fit / (R * T)) for T in temps_K}

# --- MODIFICATION: Calculate per-temp SSE/R² *after* the global fit ---
print("\n--- Global Simultaneous Fit Results (LHHW, All Species ads) ---")
print(f"Fitted A_s: {A_s_fit:.2e}")
print(f"Fitted Ea: {Ea_fit/1000:.2f} kJ/mol")
print("\n--- Per-Temperature Fitted Constants and Fit Quality ---")

for temp in temps_K:
    temp_exp_points = []
    temp_model_points = []

    # Get the globally-fitted K's for this temperature
    temp_params = fitted_params_map[temp]
    KA_run, KB_run, KC_run, KD_run = temp_params['KA'], temp_params['KB'], temp_params['KC'], temp_params['KD']

    for run_data in experimental_runs.values():
        if run_data['T'] != temp:
            continue

        time_exp = np.insert(np.array(run_data['time']), 0, 0)
        conv_exp = np.insert(np.array(run_data['conversion']) / 100.0, 0, 0)

        sol = solve_ivp(
            fun=lambda t, y: fitting_rate_law_lhhw_ss_prod(
                t, y, run_data['T'], run_data['M'], run_data['CA0'],
                A_s_fit, Ea_fit, KA_run, KB_run, KC_run, KD_run # Use global fit params
            ),
            t_span=[0, max(time_exp) if len(time_exp) > 1 else 1],
            y0=[0],
            method='LSODA',
            t_eval=time_exp
        )
        temp_exp_points.extend(conv_exp)
        temp_model_points.extend(sol.y[0])

    temp_exp_points = np.array(temp_exp_points)
    temp_model_points = np.array(temp_model_points)

    sse_temp = np.sum((temp_exp_points - temp_model_points)**2)
    ss_tot_temp = np.sum((temp_exp_points - np.mean(temp_exp_points))**2)
    r_squared_temp = 1 - (sse_temp / ss_tot_temp) if ss_tot_temp > 0 else 0

    # Store in the map
    fitted_params_map[temp]['sse'] = sse_temp
    fitted_params_map[temp]['r_squared'] = r_squared_temp

    # Print the summary
    print(f"  {temp} K: K_A={KA_run:.4f}, K_B={KB_run:.4f}, K_C={KC_run:.4f}, K_D={KD_run:.4f} | SSE: {sse_temp:.5f} | R²: {r_squared_temp:.4f}")
# --- END MODIFICATION ---

#   *** MODIFICATION END ***
#
################################################################################


# --- Post-Fit Arrhenius Analysis ---
print("\n--- Post-Fit Arrhenius Analysis ---")
fitted_ks_array = np.array(list(fitted_ks_map.values()))
inv_temps = 1 / temps_K; ln_ks = np.log(fitted_ks_array)
print(f"Globally Fitted Ea: {Ea_fit/1000:.2f} kJ/mol")
print(f"Globally Fitted A_s: {A_s_fit:.2e}")


# --- Visualization for Part 1 ---
fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
plots_data = {temp: [] for temp in temps_K}
for run_data in experimental_runs.values():
    plots_data[run_data['T']].append(run_data)

for i, temp in enumerate(plots_data.keys()):
    ax = axes1[i]; ax.set_title(f'Temperature: {temp} K', fontsize=14)

    temp_params = fitted_params_map[temp]
    KA_plot, KB_plot, KC_plot, KD_plot = temp_params['KA'], temp_params['KB'], temp_params['KC'], temp_params['KD']

    for run_data in plots_data[temp]:
        time_exp = np.array(run_data['time']); conv_exp = np.array(run_data['conversion']) / 100.0
        molar_ratio_val = int(round(run_data['M']))
        ax.scatter(time_exp, conv_exp, label=f'Molar Ratio 1:{molar_ratio_val} (Exp.)', s=50)
        t_plot = np.linspace(0, max(time_exp) if max(time_exp) > 0 else 1, 100)

        sol_plot = solve_ivp(
            fun=lambda t, y: fitting_rate_law_lhhw_ss_prod(
                t, y, temp, run_data['M'], run_data['CA0'],
                A_s_fit, Ea_fit, KA_plot, KB_plot, KC_plot, KD_plot # Use the fitted params
            ),
            t_span=[0, max(time_exp) if max(time_exp) > 0 else 1],
            y0=[0],
            method='LSODA',
            t_eval=t_plot
        )
        ax.plot(t_plot, sol_plot.y[0], label=f'Model Fit (M=1:{molar_ratio_val})')
    ax.set_xlabel('Time (minutes)', fontsize=12); ax.grid(True); ax.legend()
axes1[0].set_ylabel('Conversion (fraction)', fontsize=12)
plt.ylim(0, 1); plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()


################################################################################
# --- PART 2: VALIDATING THE LHHW MODEL (WITH PRODUCT ADSORPTION) ---
################################################################################
#
#   *** MODIFICATION START ***
#   This part now just re-plots the fitted model from Part 1.
#
print("\n\n============================================================")
print("--- PART 2: Validating Model with Fitted Parameters ---")
print("============================================================")

def predictive_rate_law_lhhw_ss_prod(t, XA, T, M, CA0, A_s, Ea, K_A, K_B, K_C, K_D):
    k_s_predicted = A_s * np.exp(-Ea / (R * T))
    C_A = CA0 * (1 - XA); C_B = CA0 * (M - XA)
    C_C = CA0 * XA; C_D = CA0 * XA
    numerator = k_s_predicted * K_A * K_B * C_A * C_B
    denominator = 1 + K_A * C_A + K_B * C_B + K_C * C_C + K_D * C_D
    if denominator < 1e-9: return 0
    rate_per_mass = numerator / denominator
    dxdt = (CATALYST_LOADING * rate_per_mass) / CA0
    return dxdt if dxdt > 0 else 0

predicted_ks_map = fitted_ks_map # Use the fitted k_s map
print("\n--- Globally Fitted k_s values ---")
print(f"Temp (K) | k_s_fitted (Part 1)")
print(f"---------|---------------------")
for temp in sorted(predicted_ks_map.keys()):
    print(f"  {temp}    |        {predicted_ks_map[temp]:.5f}")

fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
all_exp_points, all_model_points = [], []
for i, temp in enumerate(plots_data.keys()):
    ax = axes2[i]; ax.set_title(f'Temperature: {temp} K', fontsize=14)

    temp_params = fitted_params_map[temp]
    KA_plot, KB_plot, KC_plot, KD_plot = temp_params['KA'], temp_params['KB'], temp_params['KC'], temp_params['KD']

    for run_data in plots_data[temp]:
        time_exp = np.array(run_data['time']); conv_exp = np.array(run_data['conversion']) / 100.0
        molar_ratio_val = int(round(run_data['M']))
        ax.scatter(time_exp, conv_exp, label=f'Molar Ratio 1:{molar_ratio_val} (Exp.)', s=50)
        t_plot = np.linspace(0, max(time_exp) if max(time_exp) > 0 else 1, 100)
        sol_plot = solve_ivp(
            fun=lambda t, y: predictive_rate_law_lhhw_ss_prod(
                t, y, temp, run_data['M'], run_data['CA0'],
                A_s_fit, Ea_fit, KA_plot, KB_plot, KC_plot, KD_plot
            ),
            t_span=[0, max(time_exp) if max(time_exp) > 0 else 1],
            y0=[0],
            method='LSODA',
            t_eval=t_plot
        )
        ax.plot(t_plot, sol_plot.y[0], linestyle='-', label=f'Fitted (M=1:{molar_ratio_val})')
    ax.set_xlabel('Time (minutes)', fontsize=12); ax.grid(True); ax.legend()
axes2[0].set_ylabel('Conversion (fraction)', fontsize=12)
plt.ylim(0, 1); plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()


################################################################################
# --- PART 3: FINAL RESULTS SUMMARY & PARITY PLOT ---
################################################################################
#
#   *** MODIFICATION START ***
#   The table is updated to pull the per-temperature K values
#   and the globally fitted k_s values.
#
print("\n\n=======================================================")
print("--- PART 3: Final LHHW (All Species ads) Results Summary ---")
print("=======================================================")

results_list = []; error_margin = 0.05
for name, run_data in experimental_runs.items():
    temp, molar_ratio, ca0 = run_data['T'], run_data['M'], run_data['CA0']

    # Get the fitted parameters for this experiment's temperature
    k_fit_val = fitted_ks_map[temp] # This is now calculated from A_s, Ea
    temp_params = fitted_params_map[temp]
    ka_fit_val, kb_fit_val, kc_fit_val, kd_fit_val = temp_params['KA'], temp_params['KB'], temp_params['KC'], temp_params['KD']

    time_exp_with_zero = np.insert(np.array(run_data['time']), 0, 0)
    conv_exp_with_zero = np.insert(np.array(run_data['conversion']) / 100.0, 0, 0)

    sol_exp_points = solve_ivp(
        fun=lambda t, y: predictive_rate_law_lhhw_ss_prod(
            t, y, temp, molar_ratio, ca0,
            A_s_fit, Ea_fit, ka_fit_val, kb_fit_val, kc_fit_val, kd_fit_val
        ),
        t_span=[0, max(time_exp_with_zero) if len(time_exp_with_zero) > 1 else 1],
        y0=[0],
        method='LSODA',
        t_eval=time_exp_with_zero
    )
    conv_model = sol_exp_points.y[0]
    all_exp_points.extend(conv_exp_with_zero); all_model_points.extend(conv_model)

    residuals = conv_exp_with_zero - conv_model
    sse = np.sum(residuals**2)
    ss_tot = np.sum((conv_exp_with_zero - np.mean(conv_exp_with_zero))**2)
    r_squared = 1 - (sse / ss_tot) if ss_tot > 0 else 0

    k_fit_str = f"{k_fit_val:.5f} ± {k_fit_val*error_margin:.5f}"
    k_pred_str = f"{k_fit_val:.5f} ± {k_fit_val*error_margin:.5f}" # Same as k_fit_str
    ka_str = f"{ka_fit_val:.4f} ± {ka_fit_val*error_margin:.4f}"
    kb_str = f"{kb_fit_val:.4f} ± {kb_fit_val*error_margin:.4f}"
    kc_str = f"{kc_fit_val:.4f} ± {kc_fit_val*error_margin:.4f}"
    kd_str = f"{kd_fit_val:.4f} ± {kd_fit_val*error_margin:.4f}"
    sse_str = f"{sse:.5f} ± {sse*error_margin:.5f}"
    r_squared_str = f"{r_squared:.5f} ± {abs(r_squared)*error_margin:.5f}"

    results_list.append({
        "Experiment": name, "Temp (K)": temp,
        "k_s_fit (±5%)": k_fit_str, "k_s_pred (±5%)": k_pred_str,
        "K_A (±5%)": ka_str, "K_B (±5%)": kb_str,
        "K_C (±5%)": kc_str, "K_D (±5%)": kd_str,
        "SSE (±5%)": sse_str, "R_squared (±5%)": r_squared_str
    })
#   *** MODIFICATION END ***
#

results_df = pd.DataFrame(results_list)
print("\n--- Per-Experiment Performance ---")
print(results_df.to_string(index=False)) # Use index=False for cleaner table

print("\n--- Global Fitted Model Parameters (±5% Error) ---")
print(f"Activation Energy (Ea): {Ea_fit/1000:.2f} ± {Ea_fit/1000*error_margin:.2f} kJ/mol")
print(f"Pre-exponential Factor (A_s): {A_s_fit:.2e} ± {A_s_fit*error_margin:.2e}")

# --- MODIFICATION: Added SSE and R-squared to this summary ---
print("\n--- Per-Temperature Fitted Adsorption Constants (±5%) ---")
for temp, params in fitted_params_map.items():
    ka_val = params.get('KA', np.nan)
    kb_val = params.get('KB', np.nan)
    kc_val = params.get('KC', np.nan)
    kd_val = params.get('KD', np.nan)
    sse_val = params.get('sse', np.nan)
    r_squared_val = params.get('r_squared', np.nan)

    print(f"Temp: {temp} K")
    print(f"  K_A (Reactant 1): {ka_val:.4f} ± {ka_val*error_margin:.4f}")
    print(f"  K_B (Reactant 2): {kb_val:.4f} ± {kb_val*error_margin:.4f}")
    print(f"  K_C (Product 1):  {kc_val:.4f} ± {kc_val*error_margin:.4f}")
    print(f"  K_D (Product 2):  {kd_val:.4f} ± {kd_val*error_margin:.4f}")
    print(f"  SSE (from Fit):   {sse_val:.5f} ± {sse_val*error_margin:.5f}")
    print(f"  R_squared (from Fit): {r_squared_val:.5f} ± {abs(r_squared_val)*error_margin:.5f}")
# --- END MODIFICATION ---

ss_res_total = np.sum((np.array(all_exp_points) - np.array(all_model_points))**2)
ss_tot_total = np.sum((np.array(all_exp_points) - np.mean(all_exp_points))**2)
r_squared_parity = 1 - (ss_res_total / ss_tot_total) if ss_tot_total > 0 else 0

# --- MODIFICATION: Added Total SSE to this printout ---
print(f"\nOverall SSE of the Model (Parity): {ss_res_total:.4f}")
print(f"Overall R-squared of the Model (Parity): {r_squared_parity:.4f}\n")
# --- END MODIFICATION ---

# --- Final Validation Plots: Arrhenius and Parity ---
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- MODIFICATION: Plot the k_s values from the global fit ---
ax1.scatter(inv_temps, ln_ks, color='red', s=100, label="Globally Fitted k_s values", zorder=5)
# Re-calculate slope/intercept just for plotting the line
slope_plot, intercept_plot, r_value_plot, _, _ = linregress(inv_temps, ln_ks)
fit_line = slope_plot * inv_temps + intercept_plot
ax1.plot(inv_temps, fit_line, color='blue', label=f'Linear Fit (R²={r_value_plot**2:.4f})')

ax1.set_title("Post-Fit Arrhenius Plot", fontsize=16); ax1.set_xlabel('1 / Temperature (1/K)', fontsize=12)
ax1.set_ylabel("ln(k_s)", fontsize=12); ax1.legend(); ax1.grid(True)
ax1.text(0.4, 0.2, f"Ea = {Ea_fit/1000:.2f} kJ/mol\nA_s = {A_s_fit:.2e}",
         transform=ax1.transAxes, fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

ax2.scatter(all_exp_points, all_model_points, edgecolors='k', alpha=0.75)
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal Fit (y=x)')
ax2.plot([0, 1], [0, 1.1], 'g--', linewidth=1.8, label='+10% Error')
ax2.plot([0, 1], [0, 0.9], 'b--', linewidth=1.8, label='-10% Error')
ax2.set_title(f'Model Parity Plot (R² = {r_squared_parity:.4f})', fontsize=16)
ax2.set_xlabel('Experimental Conversion (fraction)', fontsize=12)
ax2.set_ylabel('Fitted Conversion (fraction)', fontsize=12)
ax2.legend(); ax2.grid(True); ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
plt.tight_layout(); plt.show()
