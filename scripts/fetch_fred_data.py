# scripts/fetch_fred_data.py

import os
from fredapi import Fred
import pandas as pd

def get_fred_api_key():
    """Load FRED API key from hidden file."""
    key_path = os.path.expanduser("~/.fred_api_key")
    if not os.path.exists(key_path):
        raise FileNotFoundError("FRED API key not found at ~/.fred_api_key")
    with open(key_path, "r") as f:
        return f.read().strip()

def fetch_series(series_ids, start="1990-01-01"):
    fred = Fred(api_key=get_fred_api_key())
    data = {}
    for code, label in series_ids.items():
        print(f"Fetching {label}...")
        series = fred.get_series(code, observation_start=start)
        data[label] = series
    df = pd.DataFrame(data)
    df.index.name = "date"
    return df

def save_to_csv(df, filename="../data/raw_macro.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename)
    print(f"Saved FRED data to {filename}")

if __name__ == "__main__":
    series_ids = {
        "GDP": "gdp",
        "CPIAUCSL": "cpi",
        "UNRATE": "unemployment",
        "FEDFUNDS": "fed_rate",
        "PAYEMS": "employment",
        "M2SL": "money_supply"
    }
    df = fetch_series(series_ids)
    save_to_csv(df)
