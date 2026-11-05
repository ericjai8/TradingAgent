"""
9 EMA x VWAP (v3.3.1) — Python engine.

Same rules as the TradingView indicator / auto-sim strategy:
  Signal  9 EMA crosses session VWAP on a closed bar, close beyond both 9 EMA and
          VWAP, 9 EMA beyond 21 EMA, VWAP slope over 5 bars beyond 0.05 pts.
          Session VWAP resets at 18:00 ET (CME trading day). Only taken when flat.
  Entry   bar close (+1 tick slippage), 3 MNQ micros.
  R       1.2 x ATR(14) at entry (same as the indicator's R labels).
  Stop    hard stop 2.5 x ATR(14) on every contract.
  TP1     1 micro at 2R. After it fills, the stop on the rest moves to breakeven.
  TP2     1 micro at 4R, or closed at the bar close when the warning (score 2/4)
          first prints while the trade is in profit.
  Runner  held until the trend-change exit: score >= 3/4 for 2 closes.
          Score (long; short mirrored): close < VWAP, 9 EMA < 21 EMA,
          VWAP slope down, swing structure broken against the trade (pivot 5).
Fills follow TradingView's broker emulator: orders placed at a close work from
the next bar; inside a bar price goes open -> nearer extreme -> other extreme ->
close; limits fill at their price, stops at the stop (or the open on a gap) plus
1 tick; commissions $0.62 per side per micro.

run(bars) -> (trades, events). bars: DataFrame indexed by tz-aware time (bar
open) with open/high/low/close/volume. Deterministic: same bars, same output.
"""
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
TICK = 0.25
PTV = 2.0            # MNQ $ per point per micro
FEE_SIDE = 0.62      # $ per micro per side

DEFAULTS = dict(ema_fast=9, ema_slow=21, use21=True, use_slope=True, slope_lb=5, min_slope=0.05,
                atr_len=14, atr_mult=1.2, hard_stop=2.5,
                exit_score=3, confirm=2, warn_score=2, pivot=5,
                contracts=3, tp1_r=2.0, tp1_q=1, tp2_r=4.0, tp2_q=1,
                tp2_at_warn=True, be_after_tp1=True, slip_ticks=1,
                exit_mode="plan")   # "plan" | "single" (1 micro, stop + trend exit) | "classic" (v2.1 exit)


def indicators(df, p):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = df["volume"].to_numpy(float) if "volume" in df else np.ones(len(df))
    ema9 = pd.Series(c).ewm(span=p["ema_fast"], adjust=False).mean().to_numpy()
    ema21 = pd.Series(c).ewm(span=p["ema_slow"], adjust=False).mean().to_numpy()
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / p["atr_len"], adjust=False).mean().to_numpy()
    et = df.index.tz_convert(ET)
    tday = (et + pd.Timedelta(hours=6)).date            # 18:00 ET starts the next trading day
    tp = (h + l + c) / 3
    g = pd.Series(tday)
    cv = pd.Series(v).groupby(g).cumsum().to_numpy()
    cpv = pd.Series(tp * v).groupby(g).cumsum().to_numpy()
    vwap = np.where(cv > 0, cpv / np.where(cv > 0, cv, 1), tp)
    lb = p["slope_lb"]
    slope = np.full(len(c), np.nan)
    slope[lb:] = vwap[lb:] - vwap[:-lb]
    return dict(o=o, h=h, l=l, c=c, ema9=ema9, ema21=ema21, atr=atr, vwap=vwap, slope=slope, et=et)


def pivots(x, n, high=True):
    """Pivot confirmed at bar k (center k-n): value or nan."""
    out = np.full(len(x), np.nan)
    for k in range(2 * n, len(x)):
        cnt = k - n
        left = x[cnt - n:cnt]
        right = x[cnt + 1:k + 1]
        if high and x[cnt] > left.max() and x[cnt] >= right.max():
            out[k] = x[cnt]
        if not high and x[cnt] < left.min() and x[cnt] <= right.min():
            out[k] = x[cnt]
    return out


def run(bars, **kw):
    p = dict(DEFAULTS, **kw)
    I = indicators(bars, p)
    o, h, l, c = I["o"], I["h"], I["l"], I["c"]
    ema9, ema21, atr, vwap, slope = I["ema9"], I["ema21"], I["atr"], I["vwap"], I["slope"]
    idx = bars.index
    n = len(c)
    slip = p["slip_ticks"] * TICK
    ph = pivots(h, p["pivot"], True)
    pl = pivots(l, p["pivot"], False)

    rising = np.nan_to_num(slope, nan=0) > p["min_slope"]
    falling = np.nan_to_num(slope, nan=0) < -p["min_slope"]
    xup = np.r_[False, (ema9[1:] > vwap[1:]) & (ema9[:-1] <= vwap[:-1])]
    xdn = np.r_[False, (ema9[1:] < vwap[1:]) & (ema9[:-1] >= vwap[:-1])]
    ok = np.ones(n, bool)
    long_sig = xup & (c > ema9) & (c > vwap) & ((ema9 > ema21) if p["use21"] else ok) & (rising if p["use_slope"] else ok)
    short_sig = xdn & (c < ema9) & (c < vwap) & ((ema9 < ema21) if p["use21"] else ok) & (falling if p["use_slope"] else ok)

    mode = p["exit_mode"]
    if mode == "plan":
        q1 = min(p["tp1_q"], p["contracts"]); q2 = min(p["tp2_q"], p["contracts"] - q1)
        qr = p["contracts"] - q1 - q2
    else:
        q1, q2, qr = 0, 0, 1

    trades, events = [], []
    swing_h = swing_l = np.nan
    sh_taken = sl_taken = True
    T = None          # open trade dict

    def ev(k, kind, text, **extra):
        events.append(dict(ts=idx[k], kind=kind, text=text, trade_id=T["id"] if T else None, **extra))

    def close_leg(k, leg, px, why):
        q = T["legs"][leg]
        if q <= 0:
            return
        pts = (px - T["fill"]) * T["d"]
        usd = pts * PTV * q - 2 * FEE_SIDE * q
        T["exits"].append(dict(ts=idx[k], leg=leg, qty=q, px=px, pts=pts, usd=usd, why=why))
        T["legs"][leg] = 0

    def open_qty():
        return sum(T["legs"].values()) if T else 0

    def finish(k):
        T["exit_ts"] = idx[k]
        T["usd"] = sum(x["usd"] for x in T["exits"])
        T["pts_avg"] = sum(x["pts"] * x["qty"] for x in T["exits"]) / sum(x["qty"] for x in T["exits"])
        T["R"] = T["pts_avg"] / T["risk"]
        T["exit_why"] = T["exits"][-1]["why"]
        trades.append(T)

    for k in range(n):
        # swing structure (global, every bar)
        bull_break = bear_break = False
        if not np.isnan(ph[k]):
            swing_h, sh_taken = ph[k], False
        if not np.isnan(pl[k]):
            swing_l, sl_taken = pl[k], False
        if not sh_taken and c[k] > swing_h:
            bull_break, sh_taken = True, True
        if not sl_taken and c[k] < swing_l:
            bear_break, sl_taken = True, True

        exit_now = False
        if T is not None and k > T["k"]:
            d = T["d"]
            # ---- intrabar: working stop / limits from the previous close
            up_first = (h[k] - o[k]) < (o[k] - l[k])     # TradingView: nearer extreme first
            path = [h[k], l[k]] if up_first else [l[k], h[k]]
            prev = o[k]
            stopped = False
            for ext in path + [c[k]]:
                lo_, hi_ = min(prev, ext), max(prev, ext)
                fav = (ext > prev) if d == 1 else (ext < prev)
                if not fav:
                    if (d == 1 and lo_ <= T["stop"]) or (d == -1 and hi_ >= T["stop"]):
                        px = T["stop"]
                        if prev == o[k] and ((d == 1 and o[k] < T["stop"]) or (d == -1 and o[k] > T["stop"])):
                            px = o[k]                          # gapped through
                        px -= d * slip
                        why = "BE" if T["be"] else "Stop"
                        for leg in ("tp1", "tp2", "run"):
                            close_leg(k, leg, px, why)
                        stopped = True
                        break
                else:
                    for leg, tgt in (("tp1", T["tp1"]), ("tp2", T["tp2"])):
                        if T["legs"][leg] > 0 and tgt is not None and lo_ <= tgt <= hi_:
                            close_leg(k, leg, tgt, f"{'TP1' if leg == 'tp1' else 'TP2'} {T['r1' if leg == 'tp1' else 'r2']:g}R")
                            ev(k, leg, "")
                prev = ext
            if stopped or open_qty() == 0:
                finish(k)
                ev(k, "exit", "")
                T = None
            else:
                # ---- at the close
                if T["legs"]["tp1"] == 0 and q1 > 0 and not T["be"] and p["be_after_tp1"]:
                    T["stop"] = T["entry"]
                    T["be"] = True
                    ev(k, "be", "")
                if mode == "classic":
                    cls = xdn[k] or (c[k] < ema9[k] and c[k - 1] >= ema9[k - 1]) if d == 1 else \
                          xup[k] or (c[k] > ema9[k] and c[k - 1] <= ema9[k - 1])
                    if cls:
                        close_leg(k, "run", c[k] - d * slip, "9 EMA")
                        finish(k); ev(k, "exit", ""); T = None; exit_now = True
                else:
                    if d == 1:
                        if bear_break: T["struct"] = True
                        if bull_break: T["struct"] = False
                        score = int(c[k] < vwap[k]) + int(ema9[k] < ema21[k]) + int(falling[k]) + int(T["struct"])
                    else:
                        if bull_break: T["struct"] = True
                        if bear_break: T["struct"] = False
                        score = int(c[k] > vwap[k]) + int(ema9[k] > ema21[k]) + int(rising[k]) + int(T["struct"])
                    T["hit"] = T["hit"] + 1 if score >= p["exit_score"] else 0
                    warn_now = score >= p["warn_score"] and not T["warned"]
                    T["warned"] = score >= p["warn_score"]
                    # R milestones (bar high/low, while still in the trade)
                    curR = int(np.floor(((h[k] - T["entry"]) if d == 1 else (T["entry"] - l[k])) / T["risk"]))
                    T["maxR"] = max(T["maxR"], curR)
                    if curR >= 2 and curR > T["lastR"]:
                        ev(k, "r", "", r_from=max(T["lastR"] + 1, 2), r_to=curR)
                        T["lastR"] = curR
                    if T["hit"] >= p["confirm"]:
                        for leg in ("tp1", "tp2", "run"):
                            close_leg(k, leg, c[k] - d * slip, f"Trend {score}/4")
                        finish(k); ev(k, "exit", ""); T = None; exit_now = True
                    elif warn_now:
                        in_profit = (c[k] > T["entry"]) if d == 1 else (c[k] < T["entry"])
                        closed_tp2 = False
                        if p["tp2_at_warn"] and T["legs"]["tp2"] > 0 and in_profit:
                            close_leg(k, "tp2", c[k] - d * slip, "Warning")
                            closed_tp2 = True
                        ev(k, "warn", "", score=score, closed_tp2=closed_tp2)

        # ---- entries (flat only)
        if T is None and not exit_now and not np.isnan(atr[k]) and k > 30:
            d = 1 if long_sig[k] else -1 if short_sig[k] else 0
            if d:
                risk = atr[k] * p["atr_mult"]
                stop = c[k] - d * atr[k] * p["hard_stop"] if mode != "classic" else -d * 1e9   # v2.1 had no stop
                T = dict(id=f"{idx[k].isoformat()}|{'L' if d == 1 else 'S'}", k=k, entry_ts=idx[k], d=d,
                         entry=c[k], fill=c[k] + d * slip, risk=risk, stop=stop, stop0=stop, stop_pts=abs(c[k] - stop),
                         r1=p["tp1_r"], r2=p["tp2_r"],
                         tp1=c[k] + d * risk * p["tp1_r"] if q1 else None,
                         tp2=c[k] + d * risk * p["tp2_r"] if q2 else None,
                         legs=dict(tp1=q1, tp2=q2, run=qr), qty=q1 + q2 + qr,
                         be=False, struct=False, hit=0, warned=False, lastR=0, maxR=0, exits=[])
                ev(k, "entry", "")
    return trades, events, T
