"""
Download recent futures bars from Yahoo Finance and append them to CSV files
in data/, so history keeps growing every day.

Yahoo limits: 1m bars go back ~7 days, 5m/15m bars ~60 days. Running this
daily keeps all of it.

Output: data/<SYMBOL>_<interval>.csv with columns
        ts (UTC epoch seconds, bar open), open, high, low, close, volume
"""
import os
import sys
import pandas as pd
import yfinance as yf

SYMBOLS = {"NQ": "NQ=F", "GC": "GC=F", "ES": "ES=F"}
INTERVALS = {"1m": "7d", "5m": "60d", "15m": "60d"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def fetch(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period, progress=False,
                     auto_adjust=False, prepost=True, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    idx = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    df.insert(0, "ts", ((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).astype("int64"))
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df.reset_index(drop=True)


def merge_save(name, interval, new):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}_{interval}.csv")
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old, new]).drop_duplicates("ts", keep="last")
    new = new.sort_values("ts")
    new.to_csv(path, index=False, float_format="%.2f")
    first = pd.to_datetime(new.ts.iloc[0], unit="s")
    last = pd.to_datetime(new.ts.iloc[-1], unit="s")
    print(f"{name} {interval}: {len(new)} bars, {first} -> {last} UTC")


def main():
    failed = 0
    for name, ticker in SYMBOLS.items():
        for interval, period in INTERVALS.items():
            try:
                df = fetch(ticker, interval, period)
                if df.empty:
                    print(f"{name} {interval}: no data returned")
                    failed += 1
                    continue
                merge_save(name, interval, df)
            except Exception as e:      # keep going if one download fails
                print(f"{name} {interval}: failed ({e})")
                failed += 1
    if failed == len(SYMBOLS) * len(INTERVALS):
        sys.exit("Every download failed")


if __name__ == "__main__":
    main()
