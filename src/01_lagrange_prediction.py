"""
01_lagrange_prediction.py  –  CS334 Data Mining Project
========================================================
Technique 1 : Lagrange Interpolation → Prediction Model

What this script does
---------------------
1. Loads the NIR calibration and validation data.
2. Finds the single wavelength whose absorbance correlates most
   with glucose concentration.
3. Builds a Lagrange interpolating polynomial through a set of
   representative (absorbance, glucose) calibration points.
4. Uses that polynomial to predict glucose on the validation set.
5. Reports Root Mean Squared Error (RMSE) and R² score.
6. Plots the fitted curve and the predictions.

Why Lagrange for NIR?
---------------------
Each sample has an absorbance spectrum (absorption vs wavelength).
At the right wavelength, absorbance tracks glucose linearly—but noise,
temperature shifts, and interferents make the relationship slightly
curved. Lagrange interpolation lets us fit a smooth polynomial through
representative calibration points without assuming the curve is
perfectly linear.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ── import our shared data loader ────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_dataset, top_correlated_wavelengths


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 – Lagrange interpolation: the core math
# ═══════════════════════════════════════════════════════════════════════════════

def lagrange_basis(x, x_nodes, i):
    """
    Compute the i-th Lagrange basis polynomial  L_i(x).

    Formula
    -------
                    n
        L_i(x)  =  ∏    (x − x_j) / (x_i − x_j)
                  j=0
                  j≠i

    In words:
      • For every OTHER node x_j (not x_i), compute (x − x_j) / (x_i − x_j).
      • Multiply all those fractions together.
      • The result is a polynomial that equals 1 at x_i and 0 at every other node.

    Parameters
    ----------
    x       : float or array – the point(s) where we evaluate the basis
    x_nodes : 1-D array      – the interpolation nodes (calibration x-values)
    i       : int            – which basis polynomial to compute

    Returns
    -------
    float or array – value(s) of L_i(x)
    """
    result = np.ones_like(np.asarray(x, dtype=float))
    xi = x_nodes[i]

    for j, xj in enumerate(x_nodes):
        if j != i:
            result *= (x - xj) / (xi - xj)   # one factor at a time

    return result


def lagrange_predict(x, x_nodes, y_nodes):
    """
    Evaluate the Lagrange interpolating polynomial at point(s) x.

    Formula
    -------
                  n
        P(x)  =  Σ  y_i * L_i(x)
                 i=0

    In words:
      • Each known y_i value is weighted by L_i(x).
      • When x equals node x_i, L_i(x)=1 and all other L_j(x)=0,
        so P(x_i) = y_i exactly (the curve passes through every node).

    Parameters
    ----------
    x       : float or array  – query point(s) to predict
    x_nodes : 1-D array       – node x-values (absorbance values)
    y_nodes : 1-D array       – node y-values (glucose concentrations)

    Returns
    -------
    float or array – predicted y value(s) at x
    """
    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x)

    for i in range(len(x_nodes)):
        total += y_nodes[i] * lagrange_basis(x, x_nodes, i)

    return total


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Load data
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TECHNIQUE 1 : LAGRANGE INTERPOLATION PREDICTION")
print("=" * 60)

data = load_dataset()
X_cal      = data['X_cal']       # (273, 4200) absorbance matrix
y_cal      = data['y_cal']       # (273,)  glucose in mM
X_val      = data['X_val']       # (120, 4200)
y_val      = data['y_val']       # (120,)
wavelengths = data['wavelengths']


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Pick the best wavelength (highest |correlation| with glucose)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Step 2] Finding most glucose-correlated wavelength …")

top_idx, top_nm, top_r = top_correlated_wavelengths(
    X_cal, y_cal, wavelengths, n=1
)
best_col = top_idx[0]               # column index in the 4200-column matrix
best_nm  = top_nm[0]                # wavelength in nm
best_r   = top_r[0]                 # Pearson correlation coefficient

print(f"  Best wavelength : {best_nm:.1f} nm")
print(f"  Pearson r       : {best_r:.4f}")

# Extract absorbance at the chosen wavelength for every sample
abs_cal = X_cal[:, best_col]        # calibration absorbance values  (273,)
abs_val = X_val[:, best_col]        # validation absorbance values   (120,)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Select representative interpolation nodes from calibration set
# ═══════════════════════════════════════════════════════════════════════════════
#
# We cannot use all 273 calibration points as nodes:
#   • A degree-272 polynomial would be wildly unstable (Runge's phenomenon).
#   • Lagrange interpolation at equally-spaced nodes on the absorbance axis
#     avoids edge oscillations.
#
# Strategy: pick N_NODES = 15 points evenly spaced across the absorbance range.

print("\n[Step 3] Selecting interpolation nodes …")

N_NODES = 15        # 15 nodes  →  degree-14 polynomial (a safe, smooth choice)

# Sort calibration samples by absorbance at the chosen wavelength
sort_order   = np.argsort(abs_cal)      # indices that would sort abs_cal
abs_sorted   = abs_cal[sort_order]
gluc_sorted  = y_cal[sort_order]

# Pick N_NODES evenly spaced indices along the sorted array
node_indices = np.linspace(0, len(abs_sorted) - 1, N_NODES, dtype=int)

x_nodes = abs_sorted[node_indices]      # node absorbance values
y_nodes = gluc_sorted[node_indices]     # corresponding glucose values

print(f"  Using {N_NODES} nodes (degree-{N_NODES - 1} polynomial)")
print("  Node absorbances :", np.round(x_nodes, 4))
print("  Node glucose mM  :", np.round(y_nodes, 2))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – Evaluate on calibration set (sanity check)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Step 4] Evaluating on calibration data …")

y_pred_cal = lagrange_predict(abs_cal, x_nodes, y_nodes)

rmse_cal = np.sqrt(mean_squared_error(y_cal, y_pred_cal))
r2_cal   = r2_score(y_cal, y_pred_cal)

print(f"  Calibration RMSE : {rmse_cal:.3f} mM")
print(f"  Calibration R²   : {r2_cal:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 – Predict on validation set (the real test)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Step 5] Predicting on validation data …")

y_pred_val = lagrange_predict(abs_val, x_nodes, y_nodes)

# Clip predictions to the physical range [0, 50] mM
y_pred_val = np.clip(y_pred_val, 0, 50)

rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val   = r2_score(y_val, y_pred_val)

print(f"  Validation RMSE  : {rmse_val:.3f} mM")
print(f"  Validation R²    : {r2_val:.4f}")

# Show a few individual predictions vs actual
print("\n  Sample predictions (first 10 validation samples):")
print(f"  {'Actual':>10}  {'Predicted':>10}  {'Error':>8}")
print("  " + "-" * 34)
for actual, pred in zip(y_val[:10], y_pred_val[:10]):
    print(f"  {actual:10.2f}  {pred:10.2f}  {abs(actual - pred):8.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 – Plot results
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[Step 6] Saving plots …")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Technique 1 – Lagrange Interpolation Prediction\n"
    f"Wavelength: {best_nm:.1f} nm  |  Nodes: {N_NODES}",
    fontsize=13, fontweight='bold'
)

# ── Left: fitted curve over calibration data ─────────────────────────────────
ax = axes[0]
ax.scatter(abs_cal, y_cal, color='steelblue', alpha=0.5, s=30,
           label='Calibration samples')
ax.scatter(x_nodes, y_nodes, color='red', zorder=5, s=80,
           marker='D', label='Interpolation nodes')

# Draw the Lagrange polynomial as a smooth line
x_line  = np.linspace(abs_cal.min(), abs_cal.max(), 400)
y_line  = lagrange_predict(x_line, x_nodes, y_nodes)
y_line  = np.clip(y_line, 0, 55)   # clip for clean plot
ax.plot(x_line, y_line, color='crimson', linewidth=2,
        label=f'Lagrange poly (deg {N_NODES - 1})')

ax.set_xlabel(f'Absorbance at {best_nm:.1f} nm', fontsize=11)
ax.set_ylabel('Glucose concentration (mM)', fontsize=11)
ax.set_title('Fitted Lagrange Polynomial', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Right: predicted vs actual on validation set ─────────────────────────────
ax = axes[1]
ax.scatter(y_val, y_pred_val, color='darkorange', alpha=0.7, s=40,
           label=f'Validation (n={len(y_val)})')

# Perfect-prediction reference line
lo = min(y_val.min(), y_pred_val.min()) - 2
hi = max(y_val.max(), y_pred_val.max()) + 2
ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='Perfect prediction')

ax.set_xlabel('Actual Glucose (mM)', fontsize=11)
ax.set_ylabel('Predicted Glucose (mM)', fontsize=11)
ax.set_title(f'Validation: Actual vs Predicted\nRMSE={rmse_val:.2f} mM  R²={r2_val:.3f}',
             fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/01_lagrange_results.png', dpi=150)
plt.close()
print("  Saved → 01_lagrange_results.png")

print("\n✓ Technique 1 complete.")