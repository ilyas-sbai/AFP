import pandas as pd
from pathlib import Path

### Parameters
UNIQUE_STRIKES = 8 # number of required unique strike per maturity

ROOT = Path(__file__).resolve().parents[1] # project root

INPUT_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "data"

# Files to load
raw_option_path = INPUT_DIR / "rawoptiondata.csv"
fomc_path = INPUT_DIR / "FOMC.csv"


### Load the data
raw_option_data = pd.read_csv(raw_option_path, parse_dates=["date", "exdate", "last_date"])
fomc_data = pd.read_csv(
    fomc_path,
    usecols=["Date", "Release Date", "Type"],
    parse_dates=["Date", "Release Date"]
)


### Filter the data to keep valid options
filtered_option_data = raw_option_data[
    (raw_option_data['best_bid'] > 0) &             # positive bid price
    (raw_option_data['impl_volatility'] > 0) &      # positive implied volatility
    (
        (raw_option_data['open_interest'] > 0) |    # positive open interest
        (raw_option_data['volume'] > 0)             # or positive volume
    )
].copy()

strike_counts = (
    filtered_option_data
    .groupby(['exdate', 'cp_flag'])['strike_price'] # group options by expiration date and type (put/call) and look at strikes
    .transform('nunique')                           # count number of unique strikes
)

filtered_option_data = (
    filtered_option_data[strike_counts >= UNIQUE_STRIKES] # filter options to keep those having more than UNIQUE_STRIKES number of strikes
    .reset_index(drop=True)
    .copy()
)


### Keep options expiring the next two Fridays following an announcement
fomc_data = fomc_data[fomc_data['Type']=='Statement']               # keep statements only (not minutes...)
fomc_data[(fomc_data['Date'] >= pd.to_datetime('2012-01-01')) & (fomc_data['Date'] < pd.to_datetime('2025-01-01'))] # keep data between 2012 and 2024
fomc_data['t0'] = fomc_data['Date'] - pd.offsets.Week(weekday=4)    # get T_0, the Friday on or before the announcement

w = pd.offsets.Week(1)
fomc_data['t1'] = fomc_data['t0'] + w       # first Friday following the announcement
fomc_data['t2'] = fomc_data['t0'] + 2*w     # second Friday following the announcement

merged_data = pd.merge(fomc_data, raw_option_data, left_on='t0', right_on='date', how='left') # align t0 with the trading days of options and drop options trading on other days
merged_data = merged_data[(merged_data['exdate']==merged_data['t1']) | (merged_data['exdate']==merged_data['t2'])] # keep only data for options expiring at T_1 or T_2

merged_data.to_parquet(OUTPUT_DIR / "filtered_options.parquet", index=False)