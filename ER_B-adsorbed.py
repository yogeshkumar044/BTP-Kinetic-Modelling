# ================================================================
#   FRIEDEL-CRAFTS: ELEY-RIDEAL (B adsorbed) MODEL FITTING
#   (Per-Temperature Correction)
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
file_path = 'FCR_BTP2.csv'  # <-- Make sure this file path is correct

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

# --- Initial Conditions ---
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
        print(f"--- WARNING: Molar ratio '{molar_ratio_key}' not found. Skipping. ---")
        continue

    experimental_runs[run_name] = {
        'T': temp,
        'M': m_val,
        'CA0': ca0_val,
        'time': group['Time'].values,
        'conversion': group['Conversion'].values
    }
print("--- Data has been structured for Friedel-Crafts analysis. ---")


################################################################################
# --- PART 1: FITTING ELEY-RIDEAL MODEL (B adsorbed) ---
################################################################################
#
#   *** MODIFICATION START ***
#   We are changing this part to fit parameters for each temperature separately.
#
print("\n\n=====================================================================")
print("--- PART 1: Fitting Eley-Rideal (B ads) Parameters Per Temperature ---")
print("=====================================================================")

def fitting_rate_law_er_B(t, XA, T, M, CA0, k_s, K_A, K_B):
    """
    Defines the ODE for the Eley-Rideal model with B adsorbed.
    Rate = (ks * KA * KB * CA * CB) / (1 + KB * CB)
    Note: This function now takes *single* k_s, K_A, K_B values.
    """
    C_A = CA0 * (1 - XA); C_B = CA0 * (M - XA)
    numerator = k_s * K_A * K_B * C_A * C_B
    denominator = 1 + K_B * C_B

    if denominator < 1e-9: return 0

    rate_per_mass = numerator / denominator
    dxdt = (CATALYST_LOADING * rate_per_mass) / CA0
    return dxdt if dxdt > 0 else 0

def sum_of_squared_errors_er_B(params, target_temp):
    """
    Objective function to minimize FOR A SINGLE TEMPERATURE.
    """
    total_error = 0
    k_s, K_A, K_B = params # 3 parameters to fit

    # Loop only through experiments matching the target_temp
    for run_data in experimental_runs.values():
        if run_data['T'] != target_temp:
            continue

        time_exp = np.insert(np.array(run_data['time']), 0, 0)
        conv_exp = np.insert(np.array(run_data['conversion']) / 100.0, 0, 0)

        sol = solve_ivp(
            fun=lambda t, y: fitting_rate_law_er_B(
                t, y, run_data['T'], run_data['M'], run_data['CA0'],
                k_s, K_A, K_B # Pass the 3 params directly
            ),
            t_span=[0, max(time_exp) if len(time_exp) > 1 else 1],
            y0=[0],
            method='LSODA',
            t_eval=time_exp
        )
        if sol.status != 0: return 1e10
        total_error += np.sum((conv_exp - sol.y[0])**2)
    return total_error

# --- Optimization Loop ---
initial_guess = [0.1, 0.5, 0.5] # Initial guess for [ks, KA, KB]
bounds = [(1e-6, None)] * 3      # Bounds for [ks, KA, KB]

# Dictionaries to store the per-temperature results
fitted_params_map = {} # Will store {433: {'ks':..., 'KA':..., 'KB':...}, ...}
fitted_ks_map = {}     # Will store {433: ks_fit, 443: ks_fit, ...}

print("\n--- Per-Temperature Eley-Rideal (B ads) Fit Results ---")
unique_temps = sorted(list(set(raw_data['Temperature (K)'])))
for temp in unique_temps:
    print(f"Fitting for {temp} K...")
    result = minimize(
        fun=lambda p: sum_of_squared_errors_er_B(p, target_temp=temp),
        x0=initial_guess,
        method='L-BFGS-B',
        bounds=bounds
    )

    if result.success:
        ks_fit, KA_fit, KB_fit = result.x

        # --- Calculate SSE and R² for this temp ---
        temp_exp_points = []
        temp_model_points = []
        for run_data in experimental_runs.values():
            if run_data['T'] != temp:
                continue

            time_exp = np.insert(np.array(run_data['time']), 0, 0)
            conv_exp = np.insert(np.array(run_data['conversion']) / 100.0, 0, 0)

            sol = solve_ivp(
                fun=lambda t, y: fitting_rate_law_er_B(
                    t, y, run_data['T'], run_data['M'], run_data['CA0'],
                    ks_fit, KA_fit, KB_fit # Use the just-fitted params
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

        # --- MODIFICATION: Store SSE and R² in the map ---
        fitted_params_map[temp] = {
            'ks': ks_fit,
            'KA': KA_fit,
            'KB': KB_fit,
            'sse': sse_temp,
            'r_squared': r_squared_temp
        }
        fitted_ks_map[temp] = ks_fit # For Arrhenius plot

        print(f"  k_s: {ks_fit:.5f} | K_A: {KA_fit:.4f} | K_B: {KB_fit:.4f} | SSE: {sse_temp:.5f} | R²: {r_squared_temp:.4f}")

    else:
        print(f"  ERROR: Fit failed for {temp} K.")
        # --- MODIFICATION: Add nan keys for sse/r_squared on failure ---
        fitted_params_map[temp] = {
            'ks': np.nan,
            'KA': np.nan,
            'KB': np.nan,
            'sse': np.nan,
            'r_squared': np.nan
        }
        fitted_ks_map[temp] = np.nan

#   *** MODIFICATION END ***
#
################################################################################

print("\n--- Post-Fit Arrhenius Analysis ---")
temps_K = np.array(list(fitted_ks_map.keys()))
fitted_ks_array = np.array(list(fitted_ks_map.values()))
inv_temps = 1 / temps_K; ln_ks = np.log(fitted_ks_array)
slope, intercept, r_value, _, _ = linregress(inv_temps, ln_ks)
Ea_fit = -slope * R; As_fit = np.exp(intercept)
print(f"Calculated Ea: {Ea_fit/1000:.2f} kJ/mol | Calculated A_s: {As_fit:.2e} | R-squared: {r_value**2:.4f}")

# --- Visualization for Part 1 ---
fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
plots_data = {temp: [] for temp in unique_temps}
for run_data in experimental_runs.values():
    plots_data[run_data['T']].append(run_data)

for i, temp in enumerate(plots_data.keys()):
    ax = axes1[i]; ax.set_title(f'Temperature: {temp} K', fontsize=14)

    # Get the fitted parameters for this specific temperature
    temp_params = fitted_params_map[temp]
    ks_plot, KA_plot, KB_plot = temp_params['ks'], temp_params['KA'], temp_params['KB']

    for run_data in plots_data[temp]:
        time_exp = np.array(run_data['time']); conv_exp = np.array(run_data['conversion']) / 100.0
        molar_ratio_val = int(round(run_data['M']))
        ax.scatter(time_exp, conv_exp, label=f'Molar Ratio 1:{molar_ratio_val} (Exp.)', s=50)
        t_plot = np.linspace(0, max(time_exp), 100) if max(time_exp) > 0 else [0]

        sol_plot = solve_ivp(
            fun=lambda t, y: fitting_rate_law_er_B(
                t, y, temp, run_data['M'], run_data['CA0'],
                ks_plot, KA_plot, KB_plot # Use the per-temp params
            ),
            t_span=[0, max(time_exp)], y0=[0], method='LSODA', t_eval=t_plot
        )
        ax.plot(t_plot, sol_plot.y[0], label=f'Model Fit (M=1:{molar_ratio_val})')
    ax.set_xlabel('Time (minutes)', fontsize=12); ax.grid(True); ax.legend()
axes1[0].set_ylabel('Conversion (fraction)', fontsize=12)
plt.ylim(0, 1); plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()


################################################################################
# --- PART 2: VALIDATING THE ELEY-RIDEAL (B ads) MODEL ---
################################################################################
#
#   *** MODIFICATION START ***
#   The predictive model uses k_s from Arrhenius, but the
#   K_A and K_B fitted for that specific temperature.
#
print("\n\n============================================================")
print("--- PART 2: Validating Model with A_s and Ea ---")
print("============================================================")

def predictive_rate_law_er_B(t, XA, T, M, CA0, A_s, Ea, K_A, K_B):
    """
    Predictive rate law using Arrhenius for k_s, and per-temp K_A, K_B
    """
    k_s_predicted = A_s * np.exp(-Ea / (R * T)) # k_s from Arrhenius
    C_A = CA0 * (1 - XA); C_B = CA0 * (M - XA)

    # K_A and K_B are the fitted values for this temperature
    numerator = k_s_predicted * K_A * K_B * C_A * C_B
    denominator = 1 + K_B * C_B
    if denominator < 1e-9: return 0
    rate_per_mass = numerator / denominator
    dxdt = (CATALYST_LOADING * rate_per_mass) / CA0
    return dxdt if dxdt > 0 else 0
#   *** MODIFICATION END ***
#

predicted_ks_map = {T: As_fit * np.exp(-Ea_fit / (R * T)) for T in temps_K}
print("\n--- Comparison of Fitted vs. Predicted k_s ---")
print(f"Temp (K) | k_s_fitted (Part 1) | k_s_predicted (Part 2)")
print(f"---------|---------------------|-----------------------")
for temp in sorted(predicted_ks_map.keys()):
    print(f"  {temp}    |        {fitted_ks_map[temp]:.5f}      |         {predicted_ks_map[temp]:.5f}")

fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
all_exp_points, all_model_points = [], []
for i, temp in enumerate(plots_data.keys()):
    ax = axes2[i]; ax.set_title(f'Temperature: {temp} K', fontsize=14)

    # Get the fitted K_A and K_B for this temp
    temp_params = fitted_params_map[temp]
    KA_plot, KB_plot = temp_params['KA'], temp_params['KB']

    for run_data in plots_data[temp]:
        time_exp = np.array(run_data['time']); conv_exp = np.array(run_data['conversion']) / 100.0
        molar_ratio_val = int(round(run_data['M']))
        ax.scatter(time_exp, conv_exp, label=f'Molar Ratio 1:{molar_ratio_val} (Exp.)', s=50)
        t_plot = np.linspace(0, max(time_exp), 100) if max(time_exp) > 0 else [0]

        sol_plot = solve_ivp(
            fun=lambda t, y: predictive_rate_law_er_B(
                t, y, temp, run_data['M'], run_data['CA0'],
                As_fit, Ea_fit, KA_plot, KB_plot # Use per-temp KA, KB
            ),
            t_span=[0, max(time_exp)], y0=[0], method='LSODA', t_eval=t_plot
        )
        ax.plot(t_plot, sol_plot.y[0], linestyle='--', label=f'Predicted (M=1:{molar_ratio_val})')
    ax.set_xlabel('Time (minutes)', fontsize=12); ax.grid(True); ax.legend()
axes2[0].set_ylabel('Conversion (fraction)', fontsize=12)
plt.ylim(0, 1); plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()


################################################################################
# --- PART 3: FINAL RESULTS SUMMARY & PARITY PLOT ---
################################################################################
#
#   *** MODIFICATION START ***
#   The results table is updated to pull the per-temperature
#   K_A and K_B values from our `fitted_params_map`.
#
print("\n\n=======================================================")
print("--- PART 3: Final Eley-Rideal (B ads) Results Summary ---")
print("=======================================================")

results_list = []; error_margin = 0.05
for name, run_data in experimental_runs.items():
    temp, molar_ratio, ca0 = run_data['T'], run_data['M'], run_data['CA0']

    # Get the fitted parameters for this experiment's temperature
    k_fit_val = fitted_params_map[temp]['ks']
    ka_fit_val = fitted_params_map[temp]['KA']
    kb_fit_val = fitted_params_map[temp]['KB']

    # Get the predicted k_s value from the Arrhenius fit
    k_pred_val = predicted_ks_map[temp]

    time_exp_with_zero = np.insert(np.array(run_data['time']), 0, 0)
    conv_exp_with_zero = np.insert(np.array(run_data['conversion']) / 100.0, 0, 0)

    sol_exp_points = solve_ivp(
        fun=lambda t, y: predictive_rate_law_er_B(
            t, y, temp, molar_ratio, ca0,
            As_fit, Ea_fit, ka_fit_val, kb_fit_val # Use per-temp K_A, K_B
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

    # Format all results with ±5% error scope
    k_fit_str = f"{k_fit_val:.5f} ± {k_fit_val*error_margin:.5f}"
    k_pred_str = f"{k_pred_val:.5f} ± {k_pred_val*error_margin:.5f}"
    ka_str = f"{ka_fit_val:.4f} ± {ka_fit_val*error_margin:.4f}"
    kb_str = f"{kb_fit_val:.4f} ± {kb_fit_val*error_margin:.4f}"
    sse_str = f"{sse:.5f} ± {sse*error_margin:.5f}"
    r_squared_str = f"{r_squared:.5f} ± {abs(r_squared)*error_margin:.5f}"

    results_list.append({
        "Experiment": name, "Temp (K)": temp,
        "k_s_fit (±5%)": k_fit_str, "k_s_pred (±5%)": k_pred_str,
        "K_A (±5%)": ka_str, "K_B (±5%)": kb_str,
        "SSE (±5%)": sse_str, "R_squared (±5%)": r_squared_str
    })
#   *** MODIFICATION END ***
#

results_df = pd.DataFrame(results_list)
print("\n--- Per-Experiment Performance ---")
print(results_df.to_string(index=False)) # Use index=False for cleaner table

# --- Global Summary ---
print("\n--- Global Model Parameters (from Arrhenius Fit) ---")
print(f"Activation Energy (Ea): {Ea_fit/1000:.2f} ± {Ea_fit/1000*error_margin:.2f} kJ/mol")
print(f"Pre-exponential Factor (A_s): {As_fit:.2e} ± {As_fit*error_margin:.2e}")

# --- MODIFICATION: Added SSE and R-squared to this summary ---
print("\n--- Per-Temperature Fitted Adsorption Constants (±5%) ---")
for temp, params in fitted_params_map.items():
    ka_val = params.get('KA', np.nan)
    kb_val = params.get('KB', np.nan)
    sse_val = params.get('sse', np.nan)
    r_squared_val = params.get('r_squared', np.nan)

    print(f"Temp: {temp} K")
    print(f"  K_A (Constant):         {ka_val:.4f} ± {ka_val*error_margin:.4f}")
    print(f"  K_B (Adsorption Const): {kb_val:.4f} ± {kb_val*error_margin:.4f}")
    print(f"  SSE (Part 1 Fit):       {sse_val:.5f} ± {sse_val*error_margin:.5f}")
    print(f"  R_squared (Part 1 Fit): {r_squared_val:.5f} ± {abs(r_squared_val)*error_margin:.5f}")
# --- END MODIFICATION ---

ss_res_total = np.sum((np.array(all_exp_points) - np.array(all_model_points))**2)
ss_tot_total = np.sum((np.array(all_exp_points) - np.mean(all_exp_points))**2)
r_squared_parity = 1 - (ss_res_total / ss_tot_total) if ss_tot_total > 0 else 0

print(f"\nOverall SSE of the Predictive Model (Parity): {ss_res_total:.4f}")
print(f"Overall R-squared of the Predictive Model (Parity): {r_squared_parity:.4f}\n")

# --- Final Validation Plots: Arrhenius and Parity ---
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

ax1.scatter(inv_temps, ln_ks, color='red', s=100, label="Fitted k_s values", zorder=5)
fit_line = slope * inv_temps + intercept
ax1.plot(inv_temps, fit_line, color='blue', label=f'Linear Fit (R²={r_value**2:.4f})')
ax1.set_title("Post-Fit Arrhenius Plot", fontsize=16); ax1.set_xlabel('1 / Temperature (1/K)', fontsize=12)
ax1.set_ylabel("ln(k_s)", fontsize=12); ax1.legend(); ax1.grid(True)
ax1.text(0.4, 0.2, f"Ea = {Ea_fit/1000:.2f} kJ/mol\nA_s = {As_fit:.2e}",
         transform=ax1.transAxes, fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

ax2.scatter(all_exp_points, all_model_points, edgecolors='k', alpha=0.75)
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal Fit (y=x)')
ax2.plot([0, 1], [0, 1.1], 'g--', linewidth=1.8, label='+10% Error')
ax2.plot([0, 1], [0, 0.9], 'b--', linewidth=1.8, label='-10% Error')
ax2.set_title(f'Model Parity Plot (R² = {r_squared_parity:.4f})', fontsize=16)
ax2.set_xlabel('Experimental Conversion (fraction)', fontsize=12)
ax2.set_ylabel('Predicted Conversion (fraction)', fontsize=12)
ax2.legend(); ax2.grid(True); ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
plt.tight_layout(); plt.show()
