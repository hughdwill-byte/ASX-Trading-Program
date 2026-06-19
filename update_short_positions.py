import pandas as pd
import io
import requests
from datetime import date, timedelta

def fetch_latest_asic_short_positions(max_days_back: int = 5):
    base_url = "https://webservices.weblink.com.au/api/shortposition/"

    for offset in range(max_days_back):
        check_date = date.today() - timedelta(days=offset)
        yyyymmdd = check_date.strftime("%Y%m%d")
        url = base_url.format(day=yyyymmdd)

        print(f"Trying {url} ...")
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            continue

        # ASIC CSVs always start with a header line containing "ReportDate"
        if "ReportDate" not in r.text:
            continue  # skip HTML pages or bad responses

        df = pd.read_csv(io.StringIO(r.text))
        print(f"✅ Loaded {len(df)} rows from {yyyymmdd}")
        return df

    raise RuntimeError("❌ No valid ASIC short-position file found for the past few days.")

if __name__ == "__main__":
    try:
        df = fetch_latest_asic_short_positions()
        print(df.head())
    except Exception as e:
        print("Failed to fetch short-position data:", e)
