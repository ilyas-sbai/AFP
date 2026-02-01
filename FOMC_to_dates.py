from pathlib import Path

import pandas as pd


### Parameters
ROOT = Path(__file__).resolve().parents[1] # project root

INPUT_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "data"
# File to load
fomc_path = INPUT_DIR / "monetary-policy-surprises-data.xlsx"

fomc_data = pd.read_excel(fomc_path, sheet_name=1, parse_dates=['Date'])
fomc_data = fomc_data.loc[fomc_data['Unscheduled'] == 0]

fomc_data = fomc_data.loc[fomc_data['Date'] >= pd.to_datetime('2012-01-01')]
fomc_data[['Date']].to_pickle(OUTPUT_DIR/'fomc_date.pickle')
fomc_data[['SP500']].to_pickle(OUTPUT_DIR/'sp500_ret.pickle')