from pathlib import Path

import pandas as pd
import numpy as np
import matlab.engine
import os
from tqdm import tqdm

# Engine Setup
if 'eng' not in globals():
    print("Starting MATLAB Engine...")
    eng = matlab.engine.start_matlab()
    eng.addpath(os.getcwd(), nargout=0)

def fit_and_grid_group(g: pd.DataFrame, 
                       grid_spacing: float = 5.0, 
                       grid_range_pct: float = 0.4,
                       cache_dir: Path = None) -> tuple:
    """
    Fits spline in MATLAB.
    Returns:
        1. pd.Series: Fitted values on original strikes.
        2. pd.DataFrame: Fitted values on new even grid (for FFT Algorithm).
    """
    # Filter for valid fitting data
    valid_data = g[['strike_price', 'bs_put_price', 'S0']].dropna()
    
    # Initialize empty returns
    empty_series = pd.Series(np.nan, index=g.index, dtype=float)
    empty_grid_df = pd.DataFrame()

    # Check if we have enough data points (need at least 4 for cubic spline)
    if len(valid_data) < 4:
        return empty_series, empty_grid_df
    
    # Metadata for the grid row
    current_date = g['Date'].iloc[0]
    exdate = g['exdate'].iloc[0]
    spot  = float(valid_data['S0'].iloc[0])

    # Try loading the spline object if it exists
    spline_loaded = False
    if cache_dir:
        # Create a safe filename: spline_YYYYMMDD_EXPYYYYMMDD.mat
        d_str = pd.to_datetime(current_date).strftime('%Y%m%d')
        e_str = pd.to_datetime(exdate).strftime('%Y%m%d')
        mat_filename = f"spline_{d_str}_{e_str}.mat"
        mat_path = cache_dir / mat_filename

        if mat_path.exists():
            try:
                eng.load(str(mat_path), 'sp', nargout=0)
                spline_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load cache {mat_path}: {e}")

    # Fit or use loaded spline
    if not spline_loaded:
        # Get raw lists
        x_vals = valid_data['strike_price'].tolist()
        y_vals = valid_data['bs_put_price'].tolist()

        # Add an anchor point deep in the money to force the spline to extrapolate with a linear slope (and not constant values)
        anchor_k = spot * 1.5
        anchor_p = anchor_k - spot
        x_vals.append(anchor_k)
        y_vals.append(anchor_p)

        # Prepare inputs for fitting
        x_fit = matlab.double(x_vals)
        y_fit = matlab.double(y_vals)

        # Fit the Spline
        # ConvexSpline(x, y, spot, sOrder=3, numPieces=[], MinConstraint=[], dK=1)
        empty = matlab.double([])
        sp_struct = eng.ConvexSpline(x_fit, y_fit, spot, empty, empty, empty, 1.0, nargout=1)
        eng.workspace['sp'] = sp_struct # save in the Matlab workspace to avoid passing the object back and forth to Python

        # Saving the spline object if caching is enable
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            eng.save(str(mat_path), 'sp', nargout=0)
    
    # Generate coarser grid (to avoid numerical instability when we recover the distribution downstream)
    # Construct the grid based on the provided spacing
    grid_min = spot * (1 - grid_range_pct)
    grid_max = spot * (1 + grid_range_pct)

    # Align min/max to grid_spacing to ensure clean integer numbers
    start = np.floor(grid_min / grid_spacing) * grid_spacing
    end = np.ceil(grid_max / grid_spacing) * grid_spacing

    # Create the grid points
    grid_points = np.arange(start, end + grid_spacing, grid_spacing)

    # Enforce odd length
    if len(grid_points) % 2 == 0:
        grid_points = np.append(grid_points, grid_points[-1] + grid_spacing)

    # Convert to matlab double for evaluation
    mat_grid = matlab.double(grid_points.tolist())
    eng.workspace['mat_grid_val'] = mat_grid

    # Evaluate spline on grid        
    fitted_prices_mat = eng.eval("fnval(sp, mat_grid_val)", nargout=1)
    fitted_prices = np.array(fitted_prices_mat).flatten()
    fitted_prices = np.maximum(fitted_prices, 0.0) # Ensure non-negative prices
    fitted_prices = np.maximum.accumulate(fitted_prices) # Fix numerical noise by enforcing monotonicity

    # Evaluate on original strikes for quality control
    mat_orig_strikes = matlab.double(valid_data['strike_price'].tolist())
    eng.workspace['x_val'] = mat_orig_strikes

    qc_prices_mat = eng.eval("fnval(sp, x_val)", nargout=1)
    qc_prices = np.array(qc_prices_mat).flatten()
    qc_prices = np.maximum(qc_prices, 0.0) # Ensure non-negative prices
    qc_prices = np.maximum.accumulate(qc_prices) # Fix numerical noise by enforcing monotonicity

    # Format Outputs
    # Original Strikes
    res_series = pd.Series(qc_prices, index=valid_data.index)
    final_series = empty_series.copy()
    final_series.update(res_series)

    # Grid DataFrame (new coarse grid)
    res_grid_df = pd.DataFrame({
        'Date': current_date,
        'exdate': exdate,
        'S0': spot,
        'grid_strikes': grid_points,
        'grid_prices': fitted_prices
    })

    return final_series, res_grid_df


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    INPUT_DIR = ROOT / "data"
    OUTPUT_DIR = ROOT / "data"
    
    # Temp dirs for both outputs
    QC_CHUNKS = OUTPUT_DIR / "temp_qc_chunks"
    GRID_CHUNKS = OUTPUT_DIR / "temp_grid_chunks"
    QC_CHUNKS.mkdir(exist_ok=True, parents=True)
    GRID_CHUNKS.mkdir(exist_ok=True, parents=True)

    CACHE_PATH = ROOT / "data" / "spline_cache" 
    
    print("Loading input data...")
    bs_prices_path = INPUT_DIR / 'blended_volatility_bs_prices.parquet'
    data = pd.read_parquet(bs_prices_path)

    unique_dates = np.sort(data['Date'].unique())
    print(f"Found {len(unique_dates)} unique dates to process.")

    # Iterate through dates
    for current_date in tqdm(unique_dates, desc="Processing Dates"):
        
        date_str = pd.to_datetime(current_date).strftime('%Y-%m-%d')
        
        qc_chunk_path = QC_CHUNKS / f"qc_{date_str}.parquet"
        grid_chunk_path = GRID_CHUNKS / f"grid_{date_str}.parquet"

        if qc_chunk_path.exists() and grid_chunk_path.exists():
            continue

        daily_data = data[data['Date'] == current_date].copy()

        # Storage for this day
        daily_series_list = []
        daily_grid_dfs = []
        
        # Group by expiration
        for _, group in daily_data.groupby('exdate'):
            # Call the updated function
            res_series, res_grid_df = fit_and_grid_group(
                group, 
                grid_spacing=5.0, 
                grid_range_pct=0.4, 
                cache_dir=CACHE_PATH
                )
            
            daily_series_list.append(res_series)
            if not res_grid_df.empty:
                daily_grid_dfs.append(res_grid_df)
            
        # Handle Market Data (Original Strikes)
        if daily_series_list:
            full_day_fit = pd.concat(daily_series_list)
            daily_data.loc[full_day_fit.index, 'bs_put_price_fit'] = full_day_fit
        else:
            daily_data['bs_put_price_fit'] = np.nan
        
        daily_data.to_parquet(qc_chunk_path, index=False)

        # Handle Grid Data (New Strikes)
        if daily_grid_dfs:
            full_grid_day = pd.concat(daily_grid_dfs, ignore_index=True)
            full_grid_day.to_parquet(grid_chunk_path, index=False)
        else:
            # Create empty placeholder if day failed
            pd.DataFrame(columns=['Date', 'exdate', 'S0', 'grid_strikes', 'grid_prices']).to_parquet(grid_chunk_path)

    print("Processing complete. Merging files...")
    
    # Merge QC Files
    qc_files = sorted(list(QC_CHUNKS.glob("qc_*.parquet")))
    if qc_files:
        pd.read_parquet(qc_files).to_parquet(OUTPUT_DIR / "convexified_market_check.parquet", index=False)
        print("Saved QC file.")

    # Merge Grid Files
    grid_files = sorted(list(GRID_CHUNKS.glob("grid_*.parquet")))
    if grid_files:
        pd.read_parquet(grid_files).to_parquet(OUTPUT_DIR / "convexified_grid_prices.parquet", index=False)
        print("Saved Grid file.")