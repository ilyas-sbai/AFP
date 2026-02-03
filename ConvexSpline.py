from pathlib import Path

import pandas as pd
import numpy as np
import matlab.engine
import os

# Engine setup (at global level)
if 'eng' not in globals():
    print("Starting MATLAB Engine...")
    eng = matlab.engine.start_matlab()
    eng.addpath(os.getcwd(), nargout=0)


def fit_group_convex_spline(g: pd.DataFrame) -> pd.Series:
    """
    Fits spline in MATLAB and evaluates it immediately on the group's data.
    """
    # Filter for valid fitting data (removes NaNs)
    valid_data = g[['strike_price', 'bs_put_price', 'S0']].dropna()
    
    # Safety check: need enough points to fit
    if len(valid_data) < 4:
        return pd.Series(np.nan, index=g.index, name='bs_put_price_fit')

    # Prepare Inputs for MATLAB
    # MATLAB Engine expects Python lists, not Numpy arrays
    x_fit = matlab.double(valid_data['strike_price'].tolist())
    y_fit = matlab.double(valid_data['bs_put_price'].tolist())
    spot  = float(valid_data['S0'].iloc[0])
    
    # Define an empty placeholder for optional arguments we want to skip
    empty = matlab.double([])

    try:
        # Call ConvexSpline(x, y, spot, sOrder, numPieces, MinConstraint, dK) in ConvexSpline.m to fit the spline
        sp_struct = eng.ConvexSpline(x_fit, y_fit, spot, empty, empty, empty, 0.1, nargout=1) # empty for default values

        # Evaluate the spline for all strikes in the group
        x_eval = matlab.double(g['strike_price'].tolist())
        
        # Calculate fitted values using MATLAB's fnval
        fitted_vals = eng.fnval(sp_struct, x_eval)
        
        # Convert list-of-lists result back to 1D numpy array
        result = np.array(fitted_vals).flatten()
        
    except Exception as e:
        print(f"Fit failed for group: {e}")
        result = np.full(len(g), np.nan)

    return pd.Series(result, index=g.index, name='bs_put_price_fit')


### Parameters
ROOT = Path(__file__).resolve().parents[1]  # project root
INPUT_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "data"

### Load the data
bs_prices_path = INPUT_DIR / 'blended_volatility_bs_prices.parquet'
data = pd.read_parquet(bs_prices_path)

### Convexify the put prices
data['bs_put_price_fit'] = (
    data.groupby(['Date', 'exdate'], group_keys=False)
        .apply(fit_group_convex_spline)
)

# Save results
data.to_parquet(OUTPUT_DIR / "convexified_prices.parquet", index=False)
