from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
from scipy.interpolate import BSpline


### Parameters
K_h_factor = 1.1
K_l_factor = 0.9

ROOT = Path(__file__).resolve().parents[1]  # project root
INPUT_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "data"

# Files to load
filtered_option_path = INPUT_DIR / "filtered_options.parquet"
s_p_500_path = INPUT_DIR / "SP500.csv"
risk_free_path = INPUT_DIR / "rf.csv"

### Load the data
option_data = pd.read_parquet(filtered_option_path)
sp500 = pd.read_csv(s_p_500_path, parse_dates=['caldt'])
rf = pd.read_csv(risk_free_path, parse_dates=['date'])

### Blend implied volatilities
option_merged = pd.merge(option_data, sp500, left_on='Date', right_on='caldt', how='left')
option_merged = option_merged.rename(columns={'spindx':'S0'})
option_merged['moneyness_plus'] = option_merged['S0'] * 1.4
option_merged['moneyness_minus'] = option_merged['S0'] * 0.6
option_merged['strike_price'] = option_merged['strike_price'] / 1000

# Keep options within the range of [0.6 * S_0, 1.4 * S_0]
option_merged = option_merged[(option_merged['strike_price']>=option_merged['moneyness_minus']) & 
                              (option_merged['strike_price']<=option_merged['moneyness_plus'])]

option_merged['K_h'] = option_merged['S0'] * K_h_factor
option_merged['K_l'] = option_merged['S0'] * K_l_factor

option_merged['w_K'] = np.select(
    [option_merged['strike_price'].ge(option_merged['K_h']), 
     option_merged['strike_price'].le(option_merged['K_l'])],
    [0, 1],  # if K>=K_h, w=0; if K<= K_l, w=1
    default=(option_merged['K_h'] - option_merged['strike_price']) / (option_merged['K_h'] - option_merged['K_l'])
)

# Default volatilities for missing call/put
grouped_by_strike = option_merged.groupby(['exdate', 'strike_price'])
call_vols = option_merged[option_merged['cp_flag'] == 'C'].groupby(['exdate', 'strike_price'])['impl_volatility'].first()
put_vols = option_merged[option_merged['cp_flag'] == 'P'].groupby(['exdate', 'strike_price'])['impl_volatility'].first()
default_call = call_vols.groupby('exdate').median()
default_put = put_vols.groupby('exdate').median()

option_merged = pd.merge(option_merged, default_call.rename('default_call_vol'), left_on='exdate', right_index=True, how='left')
option_merged = pd.merge(option_merged, default_put.rename('default_put_vol'), left_on='exdate', right_index=True, how='left')

def blend_sigma(group):
    first_row = group.iloc[0]
    call_rows = group[group['cp_flag'] == 'C']
    put_rows = group[group['cp_flag'] == 'P']

    # Get call volatility (actual or default)
    if len(call_rows) > 0:
        call_vol = call_rows.iloc[0]['impl_volatility']
    else:
        call_vol = first_row['default_call_vol']

    # Get put volatility and weight (actual or default)
    if len(put_rows) > 0:
        put_vol = put_rows.iloc[0]['impl_volatility']
    else:
        put_vol = first_row['default_put_vol']

    # w_K is the same for all rows in this group (same strike_price)
    w = first_row['w_K']
    blended = w * put_vol + (1 - w) * call_vol
    group['sigma_blended'] = blended
    return group

option_merged = option_merged.groupby(['exdate', 'strike_price'], group_keys=False).apply(blend_sigma)

## Compute Black-Scholes put prices
put_options = option_merged[option_merged['cp_flag'] == 'P']  # keep put options only
put_options = pd.merge(put_options, rf, left_on='Date', right_on='date', how='left')  # add risk free data

def black_scholes_put_vectorized(S, K, T, r, sigma):
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate Black-Scholes put option price
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

put_options['T'] = (put_options['exdate'] - put_options['Date']).dt.days / 365.25
put_options['bs_put_price'] = black_scholes_put_vectorized(
    S=put_options['S0'],  # Current stock price
    K=put_options['strike_price'],  # Strike price
    T=put_options['T'],  # Time to expiration in years
    r=put_options['dtb4wk'] / 100,  # Risk-free rate
    sigma=put_options['sigma_blended']  # Implied volatility
)


### Convexify the put prices
# Prepare the data
df = put_options[['Date', 'exdate', 'strike_price', 'bs_put_price', 'T', 'S0', 'dtb4wk']].copy()
df = df.dropna()


def convex_spline(
    x: np.ndarray,
    y: np.ndarray,
    spot: float | None = None,
    spline_order: int = 3,
    num_pieces: int | None = None,
    min_constraint: float = 0.0,
    dk: float = 0.1
) -> BSpline:
    """
    Fit a convex quadratic spline approximation to data (x, y).
    
    Uses quadratic programming to enforce convexity constraints while
    minimizing the least squares fit to the data. Quadratic splines allow
    for discontinuous densities.
    
    Parameters
    ----------
    x : np.ndarray
        Vector of strike prices (not necessarily evenly spaced)
    y : np.ndarray
        Vector of put prices
    spot : float, optional
        Current spot price for additional constraint
    spline_order : int, default=3
        Order of spline (3 for quadratic, 4 for cubic, etc.)
    num_pieces : int, optional
        Number of spline pieces (controls knot placement)
        Default is 2 + len(x) // 2
    min_constraint : float, default=0.0
        Constraint that D+D- P is greater than (0 or -1/len(x))
    dk : float, default=0.1
        Spacing for convexity check grid (NOT spacing between strikes)
        
    Returns
    -------
    BSpline
        Fitted convex spline object
    """
    # Ensure column vectors and proper types
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Set defaults
    if num_pieces is None:
        num_pieces = 2 + len(x) // 2
    
    # Ensure not too many spline knots for fitting
    num_pieces = min(num_pieces, len(x) - spline_order + 1)
    
    # Create initial knot sequence
    # Note: Python's BSpline uses a different convention than MATLAB
    # We need to create knots with proper multiplicity at boundaries
    interior_knots = np.linspace(x.min(), x.max(), num_pieces + 1)
    
    # Create augmented knot sequence (with multiplicity k at boundaries)
    knots = np.concatenate([
        np.repeat(interior_knots[0], spline_order),
        interior_knots[1:-1],
        np.repeat(interior_knots[-1], spline_order)
    ])
    
    # Build B-spline basis matrix
    # This is equivalent to MATLAB's spcol
    n_basis = len(knots) - spline_order
    B = np.zeros((len(x), n_basis))
    
    for i in range(n_basis):
        # Create basis function with coefficient = 1 at position i
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        basis_spline = BSpline(knots, coeffs, spline_order - 1)
        B[:, i] = basis_spline(x)
    
    # Create grid for convexity constraints
    x_min, x_max = x.min(), x.max()
    grid = np.arange(x_min + dk/2, x_max, dk)
    
    # Evaluate second derivatives on grid (for convexity: f'' >= 0)
    D2B = np.zeros((len(grid), n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        basis_spline = BSpline(knots, coeffs, spline_order - 1)
        # Second derivative
        d2_spline = basis_spline.derivative(2)
        D2B[:, i] = d2_spline(grid)
    
    A = -D2B  # Negative for convexity constraint: -f'' <= 0 => f'' >= 0
    
    # First derivative for equality constraints
    D1B = np.zeros((len(grid), n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        basis_spline = BSpline(knots, coeffs, spline_order - 1)
        d1_spline = basis_spline.derivative(1)
        D1B[:, i] = d1_spline(grid)
    
    # Constraint that last CDF value is 1
    Aeq1 = D1B[-1:, :]
    
    # Setup equality constraints
    if spot is not None:
        # Constraint that sum(rn_probs * x) = spot
        Aeq2 = dk * grid @ (-A)
        Aeq = np.vstack([Aeq1, Aeq2.reshape(1, -1)])
        beq_1 = 1.0 if min_constraint == 0 else 0.0
        beq = np.array([beq_1, spot])
    else:
        Aeq = Aeq1
        beq_1 = 1.0 if min_constraint == 0 else 0.0
        beq = np.array([beq_1])
    
    # Setup quadratic programming problem
    # min 0.5 * c^T * H * c + f^T * c
    # subject to: A * c <= b, Aeq * c = beq
    H = B.T @ B
    f = -B.T @ y
    b = -min_constraint * np.ones(A.shape[0])
    
    # Solve using scipy's quadratic programming
    result = optimize.minimize(
        fun=lambda c: 0.5 * c @ H @ c + f @ c,
        x0=np.zeros(n_basis),
        method='SLSQP',
        constraints=[
            {'type': 'ineq', 'fun': lambda c: -(A @ c - b)},  # A*c <= b
            {'type': 'eq', 'fun': lambda c: Aeq @ c - beq}
        ],
        options={'maxiter': 500, 'disp': False}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")
    
    coeffs = result.x
    
    # Return BSpline object
    return BSpline(knots, coeffs, spline_order - 1)

# Group by Date and exdate to convexify each expiration separately
def convexify_group(group):
    """Apply convexification to a single Date-exdate group."""
    # Average any duplicate strikes within this group
    group_agg = group.groupby('strike_price', as_index=False).agg({
        'bs_put_price': 'first',
        'Date': 'first',
        'exdate': 'first'
    })
    group_agg = group_agg.sort_values('strike_price').reset_index(drop=True)

    # Need at least 3 points for spline fitting
    if len(group_agg) < 3:
        # If too few points, return original prices
        print('3 Strikes needed')
        group_agg['fitted_price'] = group_agg['bs_put_price']
        group_agg['residuals'] = 0.0
        return group_agg

    # Extract arrays
    K = group_agg['strike_price'].values
    P_obs = group_agg['bs_put_price'].values

    try:
        # Apply the new convexification function
        bspline= convex_spline(
            x=K,
            y=P_obs,
        )
        fitted_prices = bspline(K)
        group_agg['fitted_price'] = fitted_prices
        group_agg['residuals'] = P_obs - fitted_prices

    except Exception as e:
        # If optimization fails, return original prices
        print(f"Warning: Convexification failed for Date={group_agg['Date'].iloc[0]}, exdate={group_agg['exdate'].iloc[0]}: {e}")
        group_agg['fitted_price'] = group_agg['bs_put_price']
        group_agg['residuals'] = 0.0

    return group_agg

# Apply convexification to each Date-exdate group
results = df.groupby(['Date', 'exdate'], group_keys=False).apply(convexify_group)
results = results.reset_index(drop=True)

# Save results
results.to_parquet(OUTPUT_DIR / "convexified_puts.parquet", index=False)

print(f"Processed {len(results)} strike-expiration combinations across {results.groupby(['Date', 'exdate']).ngroups} date-expiration groups")
print(f"Mean absolute residual: {results['residuals'].abs().mean():.6f}")
