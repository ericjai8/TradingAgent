"""
9 EMA x VWAP paper bot — separate from bot.py. Runs in GitHub Actions around
the clock (CME hours: Sun 6pm ET to Fri 5pm ET) and posts to Discord.

Every minute it pulls NQ 5m bars from Yahoo (delayed ~10 min), replays the
strategy in strategy/emavwap.py on the closed bars, and posts any new events:
entry, TP1 (+ stop to breakeven), TP2, R milestones, trend warning, exit.
Closed trades go to paper/emavwap_trades.csv. Nothing places real orders.

Discord: uses the DISCORD_WEBHOOK_EMAVWAP secret (its own channel). If that is
not set it falls back to DISCORD_WEBHOOK, with every message tagged [EMA×VWAP].

Usage: python paper/emavwap_bot.py [--once] [--minutes N] [--dry]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "strategy"))
import emavwap as E  # noqa: E402

ET = E.ET
STATE = os.path.join(HERE, "emavwap_state.json")
LOG = os.path.join(HERE, "emavwap_trades.csv")
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_EMAVWAP") or os.environ.get("DISCORD_WEBHOOK", "")
TAG = "" if os.environ.get("DISCORD_WEBHOOK_EMAVWAP") else "[EMA×VWAP] "
PLAN = E.DEFAULTS
DRY = False


# ------------------------------------------------------------------ data
def _yf(**kw):
    x = yf.download("NQ=F", interval="5m", progress=False, auto_adjust=False, prepost=True, threads=False, **kw)
    if x is None or x.empty:
        return pd.DataFrame()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    x.index = x.index.tz_convert("UTC") if x.index.tz is not None else x.index.tz_localize("UTC")
    return x


HIST = None


def bars():
    """60 days of 5m bars (fetched once per run) + the latest 2 days (every call).
    Only CLOSED bars are returned."""
    global HIST
    if HIST is None:
        HIST = _yf(period="60d")
        print(f"history: {len(HIST)} bars", flush=True)
    cur = _yf(period="2d")
    df = pd.concat([HIST, cur])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(minutes=5) <= now]


# ------------------------------------------------------------- discord
def post(msg):
    msg = TAG + msg
    print(msg, flush=True)
    if DRY or not WEBHOOK:
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


def f(x):
    return f"{x:,.2f}"


def t_et(ts):
    return ts.tz_convert(ET).strftime("%a %H:%M ET")


def side(t):
    return "LONG" if t["d"] == 1 else "SHORT"


def risk_usd(t):
    return t["stop_pts"] * E.PTV * t["qty"] + 2 * E.FEE_SIDE * t["qty"]


def leg_exit(t, leg):
    for x in t["exits"]:
        if x["leg"] == leg:
            return x
    return None


def entry_msg(t, joining=False):
    head = "JOINING OPEN TRADE" if joining else ("🟢 BUY" if t["d"] == 1 else "🔴 SELL")
    tp = []
    if t["tp1"] is not None:
        tp.append(f"TP1 {f(t['tp1'])} ({t['r1']:g}R)")
    if t["tp2"] is not None:
        tp.append(f"TP2 {f(t['tp2'])} ({t['r2']:g}R or ⚠)")
    return (f"{head} | {side(t)} MNQ @ {f(t['entry'])} ({t_et(t['entry_ts'])} 5m bar) | "
            f"stop {f(t['stop'])} ({t['stop_pts']:.1f} pts) | {' | '.join(tp)} | runner to trend EXIT | "
            f"1R = {t['risk']:.1f} pts | {t['qty']} micros, ${risk_usd(t):,.0f} at risk | data delayed ~10 min")


def legs_text(t):
    g = {}
    for x in t["exits"]:                       # group identical fills: "Stop 3@29,633.72 $-428"
        key = (x["why"], round(x["px"], 2))
        q, u = g.get(key, (0, 0.0))
        g[key] = (q + x["qty"], u + x["usd"])
    return " | ".join(f"{w} {q}@{f(px)} ${u:+,.0f}" for (w, px), (q, u) in g.items())


def exit_msg(t):
    legs = legs_text(t)
    return (f"⬛ EXIT {side(t)} — {t['exit_why']} | trade {t['R']:+.2f}R, ${t['usd']:+,.0f} ({t['qty']} micros) | "
            f"max reached {t['maxR']}R | {legs}")


def event_msg(e, t):
    k = e["kind"]
    if k == "entry":
        return entry_msg(t)
    if k in ("tp1", "tp2"):
        x = leg_exit(t, k)
        if x is None:
            return None
        extra = " | stop on the rest → breakeven" if k == "tp1" and PLAN["be_after_tp1"] else ""
        return f"✅ {x['why']} hit | {side(t)} 1 micro @ {f(x['px'])} | ${x['usd']:+,.0f}{extra}"
    if k == "r":
        filled = {t["r1"] if x["leg"] == "tp1" else t["r2"]
                  for x in t["exits"] if x["leg"] in ("tp1", "tp2") and x["ts"] == e["ts"] and x["why"].startswith("TP")}
        rs = [r for r in range(e["r_from"], e["r_to"] + 1) if r not in filled]   # TP message already says it
        if not rs:
            return None
        return f"📈 {side(t)} reached {' · '.join(f'{r}R' for r in rs)}"
    if k == "warn":
        s = f"⚠ Trend warning ({e['score']}/4) on the {side(t)}"
        if e.get("closed_tp2"):
            x = leg_exit(t, "tp2")
            if x:
                s += f" | closed TP2 micro @ {f(x['px'])} ${x['usd']:+,.0f}"
        return s
    if k == "exit":
        return exit_msg(t)
    return None   # 'be' is folded into the TP1 message


# ------------------------------------------------------------- state
def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {}


def save(st, closed):
    st["posted"] = st.get("posted", [])[-3000:]
    json.dump(st, open(STATE, "w"), indent=1, default=str)
    if closed:
        rows = [dict(id=t["id"], side=side(t), entry_time=str(t["entry_ts"].tz_convert(ET)), entry=t["entry"],
                     stop=t["stop0"], stop_pts=round(t["stop_pts"], 2), risk_1R_pts=round(t["risk"], 2),
                     exit_time=str(t["exit_ts"].tz_convert(ET)), exit_reason=t["exit_why"], R=round(t["R"], 3),
                     usd=round(t["usd"], 2), micros=t["qty"], max_R=t["maxR"],
                     legs=legs_text(t))
                for t in closed]
        new = pd.DataFrame(rows)
        if os.path.exists(LOG):
            new = pd.concat([pd.read_csv(LOG), new]).drop_duplicates("id", keep="last")
        new.to_csv(LOG, index=False)
    if os.environ.get("GITHUB_ACTIONS") and not DRY:
        subprocess.run("git add paper/emavwap_state.json paper/emavwap_trades.csv 2>/dev/null; "
                       "git diff --cached --quiet || (git commit -q -m 'EMA x VWAP paper update' && "
                       "git pull -q --rebase && git push -q)", shell=True, cwd=os.path.join(HERE, ".."))


# ------------------------------------------------------------- main step
def step(st, df=None):
    df = bars() if df is None else df
    if len(df) < 500:
        print("not enough bars", len(df)); return st, []
    trades, events, open_t = E.run(df, **PLAN)
    by_id = {t["id"]: t for t in trades}
    if open_t:
        by_id[open_t["id"]] = open_t
    since = pd.Timestamp(st["since"])
    posted = set(st.get("posted", []))
    last_bar = df.index[-1]

    if not st.get("joined") and open_t is not None and open_t["entry_ts"] < since:
        post(entry_msg(open_t, joining=True))
        posted.add(f"{open_t['id']}|entry")
    st["joined"] = True

    for e in events:
        if e["ts"] < since or e["trade_id"] not in by_id:
            continue
        t = by_id[e["trade_id"]]
        if t["entry_ts"] < since and f"{t['id']}|entry" not in posted:
            continue                      # trade started before the bot; we never announced it
        key = f"{e['trade_id']}|{e['kind']}|{e['ts'].isoformat()}"
        if e["kind"] == "entry":
            key = f"{t['id']}|entry"
        if key in posted:
            continue
        if e["kind"] == "warn" and f"{t['id']}|warn1" in posted and not e.get("closed_tp2"):
            posted.add(key)                   # one warning post per trade
            continue
        if e["kind"] == "warn":
            posted.add(f"{t['id']}|warn1")
        msg = event_msg(e, t)
        if msg:
            post(msg)
        posted.add(key)

    closed = [t for t in trades if t["exit_ts"] >= since]
    # daily summary after the 5pm ET close
    now_et = datetime.now(ET)
    day = now_et.date() if now_et.hour >= 17 else None
    if day and str(day) not in st.get("summaries", []) and now_et.weekday() < 5:
        # CME trading day: 6pm ET the evening before -> 5pm ET
        today = [t for t in closed if (t["exit_ts"].tz_convert(ET) + timedelta(hours=6)).date() == day]
        ids, usd_all, n_all = set(), 0.0, 0
        if os.path.exists(LOG):
            lg = pd.read_csv(LOG)
            ids, usd_all, n_all = set(lg["id"]), float(pd.to_numeric(lg["usd"], errors="coerce").sum()), len(lg)
        extra = [t for t in closed if t["id"] not in ids]
        usd_all += sum(t["usd"] for t in extra); n_all += len(extra)
        if today:
            post(f"📊 Day summary {day:%a %b %d} (CME day, 6pm–5pm ET): {len(today)} trade(s), "
                 f"${sum(t['usd'] for t in today):+,.0f}, {sum(t['R'] for t in today):+.2f}R | "
                 f"since the bot started: {n_all} trades, ${usd_all:+,.0f}")
        else:
            post(f"📊 Day summary {day:%a %b %d}: no completed trades.")
        st.setdefault("summaries", []).append(str(day))
        st["summaries"] = st["summaries"][-30:]

    st["posted"] = list(posted)
    st["last_bar"] = str(last_bar)
    return st, closed


def market_open(now_et):
    wd, hm = now_et.weekday(), now_et.hour * 100 + now_et.minute
    if wd == 5:
        return False                        # Saturday
    if wd == 6 and hm < 1800:
        return False                        # Sunday before the open
    if wd == 4 and hm >= 1715:
        return False                        # Friday after the close
    return True


def main():
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--minutes", type=int, default=340, help="run this long, then exit")
    ap.add_argument("--dry", action="store_true", help="print instead of posting / committing")
    a = ap.parse_args()
    DRY = a.dry
    st = load_state()
    if "since" not in st:
        st["since"] = pd.Timestamp.now(tz="UTC").isoformat()
        post("EMA×VWAP paper bot online. Plan: 3 MNQ micros per signal — TP1 1 micro at 2R (then stop → breakeven), "
             "TP2 1 micro at 4R or at the ⚠, runner to the trend-change EXIT. Hard stop 2.5×ATR. All sessions, 5m bars, "
             "data delayed ~10 min. Paper only — no real orders.")
    end = datetime.now(ET) + timedelta(minutes=a.minutes)
    while True:
        now = datetime.now(ET)
        if not market_open(now) and not a.once:
            print("market closed", now); break
        try:
            st, closed = step(st)
            save(st, closed)
        except Exception as e:
            print("step error", repr(e), flush=True)
        if a.once or datetime.now(ET) >= end:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
