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
file_path = 'FCR_BTP2.csv'

# --- Load the data using pandas ---
try:
    raw_data = pd.read_csv(file_path)
    print(f"--- Successfully loaded data from '{file_path}' ---")
except FileNotFoundError:
    print(f"--- ERROR: The file '{file_path}' was not found. Please upload it to your session. ---")
    exit()

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

    # Look up CA0 and M from the initial_conditions dictionary
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
print("--- Data has been structured for analysis using updated initial conditions. ---")
# --- MODIFICATION END ---

# --- Physical and Experimental Constants ---
R = 8.314  # Ideal gas constant in J/(mol·K)
CATALYST_LOADING = 0.01 # Defined as kg_cat / L_fluid

################################################################################
# --- PART 1: FITTING PSEUDO-HOMOGENEOUS MODEL ---
################################################################################

print("\n\n=====================================================================")
print("--- PART 1: Fitting PH Model for k_343, k_353, and k_363 ---")
print("=====================================================================")

# --- MODIFICATION: Added CA0 as an argument to the rate law function ---
def fitting_rate_law_ph(t, XA, T, M, CA0, k_343, k_353, k_363):
    """
    PH Rate Law: dXA/dt = k * W * C_A0 * (1-XA) * (M-XA)
    """
    if T == 343:
        k_temp = k_343
    elif T == 353:
        k_temp = k_353
    else: # T == 363
        k_temp = k_363

    dxdt = k_temp * CATALYST_LOADING * CA0 * (1 - XA) * (M - XA)
    return dxdt if dxdt > 0 else 0

def sum_of_squared_errors_ph(params):
    k_343, k_353, k_363 = params
    total_error = 0
    for run_data in experimental_runs.values():
        time_exp = np.array(run_data['time'])
        conv_exp = np.array(run_data['conversion']) / 100.0
        time_exp_with_zero = np.insert(time_exp, 0, 0)
        conv_exp_with_zero = np.insert(conv_exp, 0, 0)

        # --- MODIFICATION: Pass CA0 from run_data into the ODE solver ---
        sol = solve_ivp(
            fun=lambda t, y: fitting_rate_law_ph(t, y, run_data['T'], run_data['M'], run_data['CA0'], k_343, k_353, k_363),
            t_span=[0, max(time_exp_with_zero) if len(time_exp_with_zero) > 0 else 1],
            y0=[0],
            method='LSODA',
            t_eval=time_exp_with_zero
        )

        if sol.status != 0:
            return 1e10

        total_error += np.sum((conv_exp_with_zero - sol.y[0])**2)
    return total_error

# Optimization to find the best-fit k values
initial_guess_ph = [0.1, 0.1, 0.1]
bounds_ph = [(1e-6, None), (1e-6, None), (1e-6, None)]
result_ph = minimize(sum_of_squared_errors_ph, initial_guess_ph, method='L-BFGS-B', bounds=bounds_ph)
k343_fit_ph, k353_fit_ph, k363_fit_ph = result_ph.x
fitted_ks_map_ph = {343: k343_fit_ph, 353: k353_fit_ph, 363: k363_fit_ph}

# --- MODIFICATION: Calculate per-temp SSE/R² *after* the global fit ---
print("\n--- PH Model Global Fit Results ---")
ph_temp_stats = {} # Dictionary to store sse/r2
temps_K = np.array([343, 353, 363])

for temp in temps_K:
    temp_exp_points = []
    temp_model_points = []
    k_fit_val = fitted_ks_map_ph[temp]

    for run_data in experimental_runs.values():
        if run_data['T'] != temp:
            continue

        time_exp = np.insert(np.array(run_data['time']), 0, 0)
        conv_exp = np.insert(np.array(run_data['conversion']) / 100.0, 0, 0)

        sol = solve_ivp(
            fun=lambda t, y: fitting_rate_law_ph(
                t, y, run_data['T'], run_data['M'], run_data['CA0'],
                k343_fit_ph, k353_fit_ph, k363_fit_ph
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
    ph_temp_stats[temp] = {'sse': sse_temp, 'r_squared': r_squared_temp}

    # Print the summary
    print(f"  {temp} K: k_fitted = {k_fit_val:.5f} | SSE: {sse_temp:.5f} | R²: {r_squared_temp:.4f}")
# --- END MODIFICATION ---


# Arrhenius analysis for the PH model
print("\n--- PH Model Post-Fit Arrhenius Analysis ---")
fitted_ks_array_ph = np.array(list(fitted_ks_map_ph.values()))
inv_temps = 1 / temps_K
ln_k_ph = np.log(fitted_ks_array_ph)
slope_ph, intercept_ph, r_value_ph, _, _ = linregress(inv_temps, ln_k_ph)
Ea_fit_ph = -slope_ph * R
A_fit_ph = np.exp(intercept_ph)
print(f"Calculated Ea: {Ea_fit_ph/1000:.2f} kJ/mol | Calculated A: {A_fit_ph:.2f} | R-squared: {r_value_ph**2:.4f}")

# Visualization of Part 1
fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
plots_data = {343: [], 353: [], 363: []}
for run_data in experimental_runs.values():
    plots_data[run_data['T']].append(run_data)

for i, temp in enumerate(plots_data.keys()):
    ax = axes1[i]
    ax.set_title(f'Temperature: {temp} K', fontsize=14)
    for run_data in plots_data[temp]:
        time_exp = np.array(run_data['time'])
        conv_exp = np.array(run_data['conversion']) / 100.0
        molar_ratio_val = int(round(run_data['M']))
        ax.scatter(time_exp, conv_exp, label=f'Molar Ratio 1:{molar_ratio_val} (Exp.)', s=50)
        t_plot = np.linspace(0, max(time_exp) if len(time_exp) > 0 else 1, 100)

        # --- MODIFICATION: Pass CA0 into the plotting ODE solver ---
        sol_plot = solve_ivp(fun=lambda t, y: fitting_rate_law_ph(t, y, temp, run_data['M'], run_data['CA0'], *fitted_ks_map_ph.values()),
                             t_span=[0, max(time_exp) if len(time_exp) > 0 else 1], y0=[0], method='LSODA', t_eval=t_plot)
        ax.plot(t_plot, sol_plot.y[0], label=f'PH Model Fit (M=1:{molar_ratio_val})')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.grid(True)
    ax.legend()
axes1[0].set_ylabel('Conversion (fraction)', fontsize=12)
plt.ylim(0, 1)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

################################################################################
# --- PART 2: VALIDATING THE PH MODEL ---
################################################################################

print("\n\n===========================================================")
print("--- PART 2: Validating PH Model with A and Ea ---")
print("===========================================================")

# --- MODIFICATION: Added CA0 as an argument to the predictive rate law function ---
def predictive_rate_law_ph(t, XA, T, M, CA0, A, Ea):
    k_predicted = A * np.exp(-Ea / (R * T))
    dxdt = k_predicted * CATALYST_LOADING * CA0 * (1 - XA) * (M - XA)
    return dxdt if dxdt > 0 else 0

predicted_ks_map_ph = {T: A_fit_ph * np.exp(-Ea_fit_ph / (R * T)) for T in temps_K}

print("\n--- Comparison of Fitted vs. Predicted Rate Constants (PH Model) ---")
print(f"Temp (K) | k_fitted (Part 1) | k_predicted (Part 2)")
print(f"---------|-------------------|----------------------")
for temp in sorted(predicted_ks_map_ph.keys()):
    print(f"  {temp}    |      {fitted_ks_map_ph[temp]:.5f}      |        {predicted_ks_map_ph[temp]:.5f}")

# Visualization of Part 2
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
all_exp_points, all_model_points = [], []

for i, temp in enumerate(plots_data.keys()):
    ax = axes2[i]
    ax.set_title(f'Temperature: {temp} K', fontsize=14)
    for run_data in plots_data[temp]:
        time_exp = np.array(run_data['time'])
        conv_exp = np.array(run_data['conversion']) / 100.0
        molar_ratio_val = int(round(run_data['M']))
        ax.scatter(time_exp, conv_exp, label=f'Molar Ratio 1:{molar_ratio_val} (Exp.)', s=50)
        t_plot = np.linspace(0, max(time_exp) if len(time_exp) > 0 else 1, 100)

        # --- MODIFICATION: Pass CA0 into the predictive ODE solver ---
        sol_plot = solve_ivp(fun=lambda t, y: predictive_rate_law_ph(t, y, temp, run_data['M'], run_data['CA0'], A_fit_ph, Ea_fit_ph),
                             t_span=[0, max(time_exp) if len(time_exp) > 0 else 1], y0=[0], method='LSODA', t_eval=t_plot)
        ax.plot(t_plot, sol_plot.y[0], linestyle='--', label=f'Predicted (M=1:{molar_ratio_val})')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.grid(True)
    ax.legend()
axes2[0].set_ylabel('Conversion (fraction)', fontsize=12)
plt.ylim(0, 1)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

################################################################################
# --- PART 3: FINAL RESULTS SUMMARY ---
################################################################################

print("\n\n=======================================================")
print("--- PART 3: Final PH Model Results Summary with Error Scope ---")
print("=======================================================")

results_list_ph = []
error_margin = 0.05

for name, run_data in experimental_runs.items():
    temp = run_data['T']
    molar_ratio = run_data['M']
    ca0 = run_data['CA0']
    time_exp = np.array(run_data['time'])
    conv_exp = np.array(run_data['conversion']) / 100.0

    time_exp_with_zero = np.insert(time_exp, 0, 0)
    conv_exp_with_zero = np.insert(conv_exp, 0, 0)

    # --- MODIFICATION: Pass CA0 for calculating model points for parity plot ---
    sol_exp_points = solve_ivp(fun=lambda t, y: predictive_rate_law_ph(t, y, temp, molar_ratio, ca0, A_fit_ph, Ea_fit_ph),
                               t_span=[0, max(time_exp_with_zero) if len(time_exp_with_zero) > 0 else 1], y0=[0], method='LSODA', t_eval=time_exp_with_zero)
    conv_model = sol_exp_points.y[0]

    all_exp_points.extend(conv_exp_with_zero)
    all_model_points.extend(conv_model)

    residuals = conv_exp_with_zero - conv_model
    sse = np.sum(residuals**2)
    ss_tot = np.sum((conv_exp_with_zero - np.mean(conv_exp_with_zero))**2)
    r_squared = 1 - (sse / ss_tot) if ss_tot > 0 else 0

    k_fit_str = f"{fitted_ks_map_ph[temp]:.5f} ± {fitted_ks_map_ph[temp]*error_margin:.5f}"
    k_pred_str = f"{predicted_ks_map_ph[temp]:.5f} ± {predicted_ks_map_ph[temp]*error_margin:.5f}"
    sse_str = f"{sse:.5f} ± {sse*error_margin:.5f}"
    r_squared_str = f"{r_squared:.5f} ± {abs(r_squared)*error_margin:.5f}"

    results_list_ph.append({
        "Experiment": name, "Temp (K)": temp, "Molar Ratio": f"1:{int(round(molar_ratio))}",
        "k_fitted (±5%)": k_fit_str,
        "k_predicted (±5%)": k_pred_str,
        "SSE (±5%)": sse_str,
        "R_squared (±5%)": r_squared_str
    })

results_df_ph = pd.DataFrame(results_list_ph)
print("\n--- Per-Experiment Performance ---")
print(results_df_ph.to_string(index=False))

print("\n--- Global Model Parameters (from Arrhenius Fit) ---")
print(f"Activation Energy (Ea): {Ea_fit_ph/1000:.2f} ± {Ea_fit_ph/1000*error_margin:.2f} kJ/mol")
print(f"Pre-exponential Factor (A): {A_fit_ph:.2e} ± {A_fit_ph*error_margin:.2e}")

# --- MODIFICATION: Added Per-Temperature Summary from Part 1 ---
print("\n--- Per-Temperature Fit Quality (from Part 1 Global Fit, ±5%) ---")
for temp, stats in ph_temp_stats.items():
    k_fit_val = fitted_ks_map_ph[temp]
    sse_val = stats.get('sse', np.nan)
    r_squared_val = stats.get('r_squared', np.nan)

    print(f"Temp: {temp} K")
    print(f"  k_fitted:        {k_fit_val:.5f} ± {k_fit_val*error_margin:.5f}")
    print(f"  SSE (Part 1 Fit):       {sse_val:.5f} ± {sse_val*error_margin:.5f}")
    print(f"  R_squared (Part 1 Fit): {r_squared_val:.5f} ± {abs(r_squared_val)*error_margin:.5f}")
# --- END MODIFICATION ---

ss_res_total = np.sum((np.array(all_exp_points) - np.array(all_model_points))**2)
ss_tot_total = np.sum((np.array(all_exp_points) - np.mean(all_exp_points))**2)
r_squared_parity = 1 - (ss_res_total / ss_tot_total) if ss_tot_total > 0 else 0

print(f"\nOverall SSE of the Predictive PH Model (Parity): {ss_res_total:.4f}")
print(f"Overall R-squared of the Predictive PH Model (Parity): {r_squared_parity:.4f}\n")

# Final Validation Plots
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Arrhenius Plot
ax1.scatter(inv_temps, ln_k_ph, color='red', s=100, label='Fitted k values', zorder=5)
fit_line_ph = slope_ph * inv_temps + intercept_ph
ax1.plot(inv_temps, fit_line_ph, color='blue', label=f'Linear Fit (R²={r_value_ph**2:.3f})')
ax1.set_title('PH Model Arrhenius Plot', fontsize=16)
ax1.set_xlabel('1 / Temperature (1/K)', fontsize=12)
ax1.set_ylabel('ln(k)', fontsize=12)
ax1.legend()
ax1.grid(True)
ax1.text(0.4, 0.2, f'Ea = {Ea_fit_ph/1000:.2f} kJ/mol\nA = {A_fit_ph:.2e}',
         transform=ax1.transAxes, fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# Parity Plot
ax2.scatter(all_exp_points, all_model_points, edgecolors='k', alpha=0.75)
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal Fit (y=x)')
# --- MODIFICATION: Changed error lines for clarity ---
ax2.plot([0, 1], [0, 1.1], 'g--', linewidth=1.8, label='+10% Error')
ax2.plot([0, 1], [0, 0.9], 'b--', linewidth=1.8, label='-10% Error')
# --- END MODIFICATION ---
ax2.set_title(f'PH Model Parity Plot (R² = {r_squared_parity:.4f})', fontsize=16)
ax2.set_xlabel('Experimental Conversion (fraction)', fontsize=12)
ax2.set_ylabel('Predicted Conversion (fraction)', fontsize=12)
ax2.legend()
ax2.grid(True)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.show()
