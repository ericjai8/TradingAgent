"""
MNQ paper-trading bot. Runs inside GitHub Actions on weekdays.

Every minute it downloads NQ 5-minute bars from Yahoo (delayed data),
re-runs today's strategies on the bars that have closed, and posts any new
entries and exits to Discord. Trades are logged to paper/trades.csv.
Nothing here places real orders.

Strategies (same rules as the backtests):
  KZ   Killzone first candle: the 9:30 and 13:30 ET 5m candles. If the body is
       >= 80% of the range and the range >= 0.5 ATR(14), go with it at the next
       bar's open. Stop 1 tick beyond the candle's other end. Target 1.5R.
  ORB  15-min opening range breakout with daily bias: range = 9:30-9:45. First
       5m close outside it before 11:30 ET, only if it agrees with the daily bias
       (prior day close vs 21-day EMA). Stop 1 tick beyond the opposite side.
       Target 2R.
  Fills: entry at next bar open + 1 tick, stop checked from the entry bar on,
  stop wins ties, target must trade 1 tick through, flat at 15:55 ET.

Usage: python paper/bot.py [--once] [--until HHMM]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ET = ZoneInfo("America/New_York")
TICK = 0.25
PTV = 2.0          # MNQ $ per point per micro
MICROS = 3
FEES = 1.0         # $ per micro round trip
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
LOG = os.path.join(HERE, "trades.csv")
WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")


# ------------------------------------------------------------------ data
def get_5m():
    df = yf.download("NQ=F", interval="5m", period="5d", progress=False,
                     auto_adjust=False, prepost=True, threads=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    idx = df.index.tz_convert(ET) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(ET)
    df.index = idx
    return df


def daily_bias():
    d = yf.download("NQ=F", interval="1d", period="6mo", progress=False, auto_adjust=False, threads=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    c = d["Close"].dropna()
    today = datetime.now(ET).date()
    c = c[[ts.date() < today for ts in c.index]]       # completed sessions only
    ema = c.ewm(span=21, adjust=False).mean()
    diff = c.iloc[-1] - ema.iloc[-1]
    return 1 if diff > 0 else -1 if diff < 0 else 0


def atr14(df):
    h, l, c = df.high.values, df.low.values, df.close.values
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr, index=df.index).ewm(alpha=1 / 14, adjust=False).mean()


# ------------------------------------------------------------ strategies
def hm(ts):
    return ts.hour * 100 + ts.minute


def signals(df, bias, today):
    """Signals from CLOSED bars of today's session. Returns list of dicts."""
    A = atr14(df)
    day = df[[ts.date() == today and 930 <= hm(ts) < 1600 for ts in df.index]]
    out = []
    # Killzone first candle
    for t in (930, 1330):
        rows = day[[hm(ts) == t for ts in day.index]]
        if len(rows) == 0:
            continue
        ts = rows.index[0]
        b = rows.iloc[0]
        rng = b.high - b.low
        if rng > 0 and rng >= 0.5 * A.loc[ts] and abs(b.close - b.open) >= 0.8 * rng:
            d = 1 if b.close > b.open else -1
            stop = (b.low if d == 1 else b.high) - d * TICK
            out.append(dict(strat=f"KZ {t // 100}:{t % 100:02d}", sig_ts=ts, d=d, stop=stop, r=1.5))
    # 15-min ORB with daily bias, stop opposite side
    rng_bars = day[[930 <= hm(ts) < 945 for ts in day.index]]
    if len(rng_bars) == 3 and bias != 0:
        hi, lo = rng_bars.high.max(), rng_bars.low.min()
        for ts, b in day[[945 <= hm(ts) < 1130 for ts in day.index]].iterrows():
            d = 1 if b.close > hi else -1 if b.close < lo else 0
            if d == 0:
                continue
            if d == bias:
                stop = (lo if d == 1 else hi) - d * TICK
                out.append(dict(strat="ORB 15m", sig_ts=ts, d=d, stop=stop, r=2.0))
            break
    return out


def play(df, s):
    """Fill and manage one signal on the closed bars available."""
    after = df[df.index > s["sig_ts"]]
    after = after[[ts.date() == s["sig_ts"].date() for ts in after.index]]
    if len(after) == 0:
        return dict(s, status="waiting")
    e_ts = after.index[0]
    d = s["d"]
    entry = after.iloc[0].open + d * TICK
    risk = (entry - s["stop"]) * d
    if risk <= 0:
        return dict(s, status="skipped", note="opened past the stop")
    tgt = entry + d * s["r"] * risk
    res = dict(s, status="open", entry_ts=e_ts, entry=entry, risk=risk, target=tgt)
    for k, (ts, b) in enumerate(after.iterrows()):
        if hm(ts) >= 1555:
            x = b.open - d * TICK
            return dict(res, status="closed", exit_ts=ts, exit=x, how="time exit 15:55")
        if (d == 1 and b.low <= s["stop"]) or (d == -1 and b.high >= s["stop"]):
            f = s["stop"] if k == 0 else (min(b.open, s["stop"]) if d == 1 else max(b.open, s["stop"]))
            return dict(res, status="closed", exit_ts=ts, exit=f - d * TICK, how="stopped out")
        if (d == 1 and b.high >= tgt + TICK) or (d == -1 and b.low <= tgt - TICK):
            return dict(res, status="closed", exit_ts=ts, exit=tgt, how="target hit")
    return res


# ------------------------------------------------------------- reporting
def post(msg):
    print(msg, flush=True)
    if not WEBHOOK:
        return
    for _ in range(3):
        try:
            r = requests.post(WEBHOOK, json={"content": msg[:1900]}, timeout=15)
            if r.status_code in (200, 204):
                return
            time.sleep(2)
        except Exception as e:
            print("discord error", e)
            time.sleep(2)


def money(pts):
    return pts * PTV * MICROS - FEES * MICROS


def fmt(x):
    return f"{x:,.2f}"


def load_state(today):
    try:
        st = json.load(open(STATE))
        if st.get("date") == str(today):
            return st
    except Exception:
        pass
    return {"date": str(today), "posted": [], "greeted": False, "summary": False}


def save_and_commit(st, trades_rows):
    json.dump(st, open(STATE, "w"), indent=1, default=str)
    if trades_rows:
        new = pd.DataFrame(trades_rows)
        if os.path.exists(LOG):
            old = pd.read_csv(LOG)
            new = pd.concat([old, new]).drop_duplicates(["date", "strategy", "entry_time"], keep="last")
        new.to_csv(LOG, index=False)
    if os.environ.get("GITHUB_ACTIONS"):
        subprocess.run("git add paper && (git diff --cached --quiet || "
                       "(git commit -q -m 'Paper trades update' && git pull -q --rebase && git push -q))",
                       shell=True, cwd=os.path.dirname(HERE))


def step(today, bias, st):
    df = get_5m()
    if df is None:
        print("no data this minute")
        return st, []
    now_et = datetime.now(ET)
    closed = df[[ts + timedelta(minutes=5) <= now_et for ts in df.index]]
    last_bar = closed.index[-1] if len(closed) else None
    rows = []
    for s in signals(closed, bias, today):
        t = play(closed, s)
        key_in = f"{t['strat']}|{s['sig_ts']}|in"
        key_out = f"{t['strat']}|{s['sig_ts']}|out"
        side = "LONG" if t["d"] == 1 else "SHORT"
        if t["status"] in ("open", "closed") and key_in not in st["posted"]:
            stop_pts = t["risk"]
            post(f"PAPER ENTRY | {t['strat']} | {side} {MICROS} MNQ @ {fmt(t['entry'])} "
                 f"({t['entry_ts']:%H:%M} ET bar) | stop {fmt(t['stop'])} ({stop_pts:.1f} pts, ${stop_pts * PTV * MICROS:,.0f}) "
                 f"| target {fmt(t['target'])} ({t['r']:g}R) | data delayed")
            st["posted"].append(key_in)
        if t["status"] == "closed" and key_out not in st["posted"]:
            pts = (t["exit"] - t["entry"]) * t["d"]
            R = pts / t["risk"]
            post(f"PAPER EXIT | {t['strat']} | {side} | {t['how']} @ {fmt(t['exit'])} ({t['exit_ts']:%H:%M} ET bar) "
                 f"| {R:+.2f}R | ${money(pts):+,.0f} at {MICROS} micros")
            st["posted"].append(key_out)
        if t["status"] in ("open", "closed"):
            rows.append(dict(date=str(today), strategy=t["strat"], side=side, entry_time=str(t["entry_ts"]),
                             entry=t["entry"], stop=t["stop"], target=t["target"], stop_pts=round(t["risk"], 2),
                             exit_time=str(t.get("exit_ts", "")), exit=t.get("exit", ""),
                             result=t.get("how", "open"),
                             R=round((t["exit"] - t["entry"]) * t["d"] / t["risk"], 3) if t["status"] == "closed" else "",
                             usd_3_micros=round(money((t["exit"] - t["entry"]) * t["d"]), 2) if t["status"] == "closed" else ""))
    st["last_bar"] = str(last_bar)
    return st, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--until", type=int, default=1610, help="stop at this ET time (HHMM)")
    ap.add_argument("--from", dest="start", type=int, default=0, help="exit if started before this ET time")
    a = ap.parse_args()
    now = datetime.now(ET)
    today = now.date()
    if now.weekday() >= 5:
        print("weekend"); return
    if hm(now) >= a.until:
        print("past this run's end time"); return
    try:
        bias = daily_bias()
    except Exception as e:
        print("bias error", e); bias = 0
    st = load_state(today)
    if not st["greeted"] and not a.once:
        post(f"Paper bot online for {today:%a %b %d}. Daily bias: {'long only' if bias == 1 else 'short only' if bias == -1 else 'none'} "
             f"(ORB 15m only trades with it). Watching NQ 5m bars, {MICROS} MNQ micros. Data is delayed about 10 minutes.")
        st["greeted"] = True
    while True:
        try:
            st, rows = step(today, bias, st)
            save_and_commit(st, rows)
        except Exception as e:
            print("step error", repr(e), flush=True)
        now = datetime.now(ET)
        if a.once or hm(now) >= a.until:
            break
        time.sleep(60)
    if not a.once and hm(datetime.now(ET)) >= 1600 and not st.get("summary"):
        try:
            log = pd.read_csv(LOG)
            t = log[(log.date == str(today)) & (log.R.astype(str) != "")]
            if len(t):
                total = pd.to_numeric(t.usd_3_micros).sum()
                post(f"Paper day summary {today:%a %b %d}: {len(t)} trade(s), "
                     f"{pd.to_numeric(t.R).sum():+.2f}R total, ${total:+,.0f} at {MICROS} micros. "
                     f"All-time paper log: {len(log)} trades, {pd.to_numeric(log.R, errors='coerce').sum():+.2f}R.")
            else:
                post(f"Paper day summary {today:%a %b %d}: no trades today.")
        except Exception as e:
            post(f"Paper day summary {today:%a %b %d}: no trades logged.")
        st["summary"] = True
        save_and_commit(st, [])


if __name__ == "__main__":
    main()
