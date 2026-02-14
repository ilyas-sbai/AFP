from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


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
# Add S&P level and risk free rate data to the option data
data = pd.merge(option_data, sp500, left_on='t0', right_on='caldt', how='left') # add the index level
data = data.rename(columns={'spindx':'S0'})
data = pd.merge(data, rf, left_on='t0', right_on='date', how='left')  # add risk free data

data['T'] = (data['exdate'] - data['t0']).dt.days / 365.25 # time to maturity

# Pivot the data to have the implied volatility data for calls and puts for a given date/exdate/strike on the same row
data_pivot = data.pivot_table(
    index=['FOMC_Date', 't0', 'exdate', 'S0', 'strike_price', 'T', 'dtb4wk'], # one row per option date/expiration date/strike price combination
    columns='cp_flag', # one column for calls, one for put
    values='impl_volatility' # implied volatility for the calls and the put
).reset_index() # add back the index columns as regulat columns
data_pivot = data_pivot.rename(columns={'C': 'sigma_C', 'P': 'sigma_P'})

# Filter for moneyness range between 0.6*S_0 and 1.4*S_0
data_pivot['strike_price'] = data_pivot['strike_price'] / 1000 # because strike is quoted in 1000s on option metrics
data_pivot = data_pivot[
    (data_pivot['strike_price'] >= data_pivot['S0'] * 0.6) &
    (data_pivot['strike_price'] <= data_pivot['S0'] * 1.4)
].copy()

# Compute the weight
K = data_pivot['strike_price']
Kl = data_pivot['S0'] * K_l_factor
Kh = data_pivot['S0'] * K_h_factor

conditions = [K <= Kl, K >= Kh] # list of series of Boolean evaluating the conditions
choices = [1.0, 0.0] #  weight given to the put, weight given to the call
middle_case = (Kh - K) / (Kh - Kl) # middle case: (Kh - K) / (Kh - Kl)
data_pivot['w'] = np.select(conditions, choices, default=middle_case) # outside the Kl bound, weight to put, outside the Kh bound, weight to call, else, blend

band = data_pivot["strike_price"].between(Kl, Kh, inclusive="both") # [Kl, Kh] band
missing_iv = data_pivot[["sigma_P", "sigma_C"]].isna().any(axis=1) # strikes that are missing either the put or call implied volatility
data_pivot = data_pivot.loc[~(band & missing_iv)].copy() # remove strikes that are within the band and that do not have both put and call implied volatility

# Impute missing implied volatilities
data_pivot = data_pivot.sort_values(['exdate', 'strike_price'])
grouped = data_pivot.groupby('exdate')
lowest_strike_vol = grouped['sigma_P'].transform('first') # put implied volatility of the lowest strike (per expiration date)
highest_strike_vol = grouped['sigma_C'].transform('last') # call implied volatility of the highest strike (per expiration date)

mask_left_tail_missing = data_pivot['sigma_P'].isna() # missing put implied volatility when it is required
data_pivot.loc[mask_left_tail_missing, 'sigma_P'] = lowest_strike_vol[mask_left_tail_missing]

mask_right_tail_missing = data_pivot['sigma_C'].isna() # missing call implied volatility when it is required
data_pivot.loc[mask_right_tail_missing, 'sigma_C'] = lowest_strike_vol[mask_right_tail_missing]

# Blend volatilities
data_pivot['sigma_blended'] = (
    data_pivot['w'] * data_pivot['sigma_P'] +
    (1 - data_pivot['w']) * data_pivot['sigma_C']
)

# Compute Black-Scholes Put Prices
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

data_pivot['bs_put_price'] = black_scholes_put(
    S=data_pivot['S0'],
    K=data_pivot['strike_price'],
    T=data_pivot['T'],
    r=data_pivot['dtb4wk'] / 100,
    sigma=data_pivot['sigma_blended']
)

# Save results
data_pivot.to_parquet(OUTPUT_DIR / "blended_volatility_bs_prices.parquet", index=False)
