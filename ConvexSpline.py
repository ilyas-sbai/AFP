from pathlib import Path

from cvxopt import matrix, solvers
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline


def ConvexSpline(x, y, spot=None, sOrder=3, numPieces=None, MinConstraint=0, dK=0.19):
    # 1. Clean and Sort Data
    x = np.array(x, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # 2. Setup Knots
    deg = int(sOrder - 1)
    
    # MATLAB logic: numPieces defaults to 2 + fix(length(x)/2)
    # This was treated as the *Dimension* of the spline in spap2
    if numPieces is None:
        numPieces = int(2 + len(x) / 2)
    
    # MATLAB constraint: numPieces <= length(x) - sOrder + 1
    # This ensures we don't have more params than data points
    max_pieces = len(x) - deg 
    if numPieces > max_pieces:
        numPieces = max_pieces

    # --- KNOT PLACEMENT FIX ---
    # To match MATLAB's 'spap2(numPieces)' dimension count:
    # Target Dimension = numPieces
    # Dimension = (num_inner_knots) + deg + 1
    # Therefore: num_inner_knots = numPieces - deg - 1
    
    n_inner = numPieces - deg - 1
    
    if n_inner > 0:
        # Use percentiles to approximate 'optimal' placement by data density
        qs = np.linspace(0, 100, n_inner + 2)[1:-1]
        inner_knots = np.unique(np.percentile(x, qs))
    else:
        inner_knots = np.array([])
        
    # Pad knots: [min, min, min, ..., inner, ..., max, max, max]
    knots = np.concatenate(([x.min()] * (deg + 1), inner_knots, [x.max()] * (deg + 1)))

    # 3. Build Basis Matrix
    B = BSpline.design_matrix(x, knots, deg).toarray()
    n_coeffs = B.shape[1]

    # 4. Build Constraint Grid
    xMin, xMax = x.min(), x.max()
    grid = np.arange(xMin + dK/2, xMax - dK/2, dK)
    
    # Derivatives
    # Note: It's faster to create one BSpline and reuse it than loop n_coeffs
    # but your loop method is safe and correct for clarity.
    A_convex_list = []
    Aeq1_list = []
    
    for i in range(n_coeffs):
        c = np.zeros(n_coeffs); c[i] = 1.0
        sp_temp = BSpline(knots, c, deg)
        A_convex_list.append(sp_temp.derivative(2)(grid))
        Aeq1_list.append(sp_temp.derivative(1)(grid))

    D2B_vals = np.array(A_convex_list).T  # Shape: (grid_len, n_coeffs)
    D1B_vals = np.array(Aeq1_list).T      # Shape: (grid_len, n_coeffs)
    
    # Inequality: -f''(x) <= 0  (Convexity)
    A_ineq = -D2B_vals 
    
    # 5. Equality Constraints
    # Constraint 1: PDF integrates to 1 => CDF(max) = 1
    row_cdf = D1B_vals[-1, :]
    
    if spot is None:
        A_eq = row_cdf.reshape(1, -1)
        b_eq = np.array([1.0])
    else:
        # Constraint 2: Martingale (mean = spot)
        # Integral x * f''(x) dx ~ sum( grid * f''(x) * dK )
        # f''(x) is D2B_vals @ c
        # We need: dK * grid @ (D2B_vals @ c) = spot
        # Vector: dK * (grid @ D2B_vals)
        row_spot = dK * (grid @ D2B_vals)
        
        A_eq = np.vstack([row_cdf, row_spot])
        b_eq = np.array([1.0, spot])

    # 6. QP Formulation
    # Minimize ||Bc - y||^2
    # Objective: (1/2) c' (2 B'B) c + (-2 y'B) c
    # CVXOPT minimizes (1/2) x' P x + q' x
    # So P = 2 * B'B and q = -2 * B'y
    # OR (simpler): P = B'B, q = -B'y (minimizes 1/2 ||...||^2) - equivalent solution
    
    H = B.T @ B
    f = -B.T @ y
    
    # Regularization
    H = H + np.eye(n_coeffs) * 1e-9

    P_mat = matrix(H)
    q_mat = matrix(f)
    
    G_mat = matrix(A_ineq)
    h_mat = matrix(np.zeros(A_ineq.shape[0]) - MinConstraint)
    
    A_mat = matrix(A_eq)
    b_mat = matrix(b_eq)

    # 7. Solve
    solvers.options['show_progress'] = False
    try:
        sol = solvers.qp(P_mat, q_mat, G_mat, h_mat, A_mat, b_mat)
        coeffs = np.array(sol['x']).flatten()
    except ValueError as e:
        print("Optimization failed.")
        raise e

    return BSpline(knots, coeffs, deg)


### Parameters
ROOT = Path(__file__).resolve().parents[1]  # project root
INPUT_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "data"

### Load the data
bs_prices_path = INPUT_DIR / 'blended_volatility_bs_prices.parquet'
data = pd.read_parquet(bs_prices_path)

### Convexify the put prices
import numpy as np
import pandas as pd

def fit_group_convex_spline(g: pd.DataFrame) -> pd.Series:
    # Keep only rows usable for the fit
    gg = g[['strike_price', 'bs_put_price', 'S0']].dropna()

    # If too few points, return NaNs for this group
    if len(gg) < 6:  # adjust threshold as you like
        return pd.Series(np.nan, index=g.index, name='bs_put_price_fit')

    x = gg['strike_price'].to_numpy(float)
    y = gg['bs_put_price'].to_numpy(float)

    # Spot: if it's constant per group, take the first; otherwise choose a rule
    spot = float(gg['S0'].iloc[0])

    # Fit spline (returns a scipy BSpline callable)
    sp = ConvexSpline(x, y, spot=spot)

    # Evaluate back on ALL rows of the original group (including those with NaNs)
    fitted = sp(g['strike_price'].to_numpy(float))

    return pd.Series(fitted, index=g.index, name='bs_put_price_fit')


data['bs_put_price_fit'] = (
    data.groupby(['date', 'exdate'], group_keys=False)
        .apply(fit_group_convex_spline)
)
