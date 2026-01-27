from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import norm


### Parameters
K_h_factor = 1.1
K_l_factor = 0.9

ROOT = Path(__file__).resolve().parents[1] # project root

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
option_merged = option_merged[(option_merged['strike_price']>=option_merged['moneyness_minus']) & (option_merged['strike_price']<=option_merged['moneyness_plus'])]
option_merged['K_h'] = option_merged['S0'] * K_h_factor
option_merged['K_l'] = option_merged['S0'] * K_l_factor
option_merged['w_K'] = np.select(
    [option_merged['strike_price'].ge(option_merged['K_h']), option_merged['strike_price'].le(option_merged['K_l'])],
    [0, 1], # if K>=K_h, w=1; if K<= K_l, w=0
    default=(option_merged['K_h'] - option_merged['strike_price']) / (option_merged['K_h'] - option_merged['K_l']) # if K_l<K<K_h, w=(K_h-K)/(K_h-K_l)
)

## Get default implied volatility
# Lowest strike put per expiration date
lowest_strikes = (option_merged[option_merged['cp_flag'] == 'P']
                  .groupby('exdate')['strike_price']
                  .min()
                  .rename('lowest_strike_put'))

# Highest strike call per expiration date
highest_strikes = (option_merged[option_merged['cp_flag'] == 'C']
                   .groupby('exdate')['strike_price']
                   .max()
                   .rename('highest_strike_call'))

# Get default put volatility (from lowest strike puts)
default_put_vol = (option_merged[(option_merged['cp_flag'] == 'P')]
                   .merge(lowest_strikes, on='exdate')
                   .query('strike_price == lowest_strike_put') # [lambda x: x['strike_price'] == x['lowest_strike_put']]
                   .groupby('exdate')['impl_volatility']
                   .mean()
                   .rename('default_put_vol'))

#  Get default call volatility (from highest strike calls)
default_call_vol = (option_merged[(option_merged['cp_flag'] == 'C')]
                    .merge(highest_strikes, on='exdate')
                    .query('strike_price == highest_strike_call')
                    .groupby('exdate')['impl_volatility']
                    .mean()
                    .rename('default_call_vol'))

# Merge everything back to broadcast
option_merged = (option_merged
                 .merge(lowest_strikes, on='exdate', how='left')
                 .merge(highest_strikes, on='exdate', how='left')
                 .merge(default_put_vol, on='exdate', how='left')
                 .merge(default_call_vol, on='exdate', how='left'))

## Compute blended implied volatility
def blend_sigma(group):
    call_rows = group[group['cp_flag'] == 'C']
    put_rows = group[group['cp_flag'] == 'P']
    
    # Get first row to access default values and w_K
    first_row = group.iloc[0]
    
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
put_options = option_merged[option_merged['cp_flag'] == 'P'] # keep put options only
put_options = pd.merge(put_options, rf, left_on='Date', right_on='date', how='left') # add risk free data

def black_scholes_put_vectorized(S, K, T, r, sigma):
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate Black-Scholes put option price
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return put_price

put_options['T'] = (put_options['exdate'] - put_options['Date']).dt.days / 365.25
put_options['bs_put_price'] = black_scholes_put_vectorized(
    S=put_options['S0'],                    # Current stock price
    K=put_options['strike_price'],          # Strike price
    T=put_options['T'],                     # Time to expiration in years
    r=put_options['dtb4wk'] / 100,         # Risk-free rate (assuming dtb4wk is in percentage)
    sigma=put_options['sigma_blended']    # Implied volatility
)


### Convexify the put prices
# Prepare the data
df = put_options[['strike_price', 'bs_put_price', 'T']].copy()
df = df.dropna()

# Average any duplicate strikes
df = df.groupby('strike_price', as_index=False).agg({
    'bs_put_price': 'mean',
    'T': 'first'
})
df = df.sort_values('strike_price').reset_index(drop=True)

# Extract arrays
K = df['strike_price'].values
P_obs = df['bs_put_price'].values
n = len(K)

# Set parameters
T = df['T'].iloc[0]  # time to maturity in years
r = 0.05  # risk-free rate (adjust as needed)
disc = np.exp(-r * T)  # discount factor

# Set up the optimization
P = cp.Variable(n)  # fitted prices at each strike

# Calculate slopes between strikes
dK = np.diff(K)
slopes = cp.multiply(1.0 / dK, P[1:] - P[:-1])

# Define constraints
constraints = [
    slopes[1:] >= slopes[:-1],  # convexity: increasing slopes
    slopes >= 0,                 # probability constraint
    slopes <= disc,              # probability constraint
    slopes[-1] == disc,          # terminal CDF = 1
    P >= 0,                      # price bounds
    P <= disc * K                # price bounds
]

# Solve the optimization problem
objective = cp.Minimize(cp.sum_squares(P - P_obs))
problem = cp.Problem(objective, constraints)
problem.solve(solver='OSQP', verbose=False)

# Extract results
P_fit = P.value

# Create results dataframe
results = pd.DataFrame({
    'strike_price': K,
    'observed_price': P_obs,
    'fitted_price': P_fit,
    'T': T,
    'r': r
})
results.to_parquet(OUTPUT_DIR / "convexified_puts.parquet", index=False)
