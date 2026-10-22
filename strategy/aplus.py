"""
Python port of "A+ Top-Down Setup v1.20.1" (Pine v5), signal logic only.

Faithful to the script, with two honest-timing choices:
  * 15m trend = last COMPLETED 15m bar's close vs its 21 EMA.
  * Daily bias (for an optional filter; the script never scores it) =
    previous completed day's close vs 21-day EMA of daily closes.
Entry is the NEXT bar's open after the signal bar closes.
"""
import numpy as np
import pandas as pd
from lab import atr as atr_fn, TICK

P = dict(min_body_pct=0.55, max_opp_wick=0.80, atr_mult=0.85, swing_lookback=5,
         min_score=9.0, reversion_bonus=2.0, structure_bonus=2.3, mtf_bonus=1.6)


def trend_15m(df):
    """+1/-1/0 per 5m bar using only completed 15m bars."""
    t = df["time"]
    key = t.dt.floor("15min")
    g = df.groupby(key, sort=True)
    c15 = g["close"].last()
    last_ts = g["ts"].max()
    ema = c15.ewm(span=21, adjust=False).mean()
    tr = np.sign(c15 - ema)
    # value becomes known at the 5m bar that completes the 15m bar
    known = pd.Series(tr.values, index=last_ts.values)
    s = pd.Series(np.nan, index=df["ts"].values)
    s.loc[known.index] = known.values
    return s.ffill().fillna(0).values


def daily_bias(df):
    """Prior completed trading day's close vs 21 EMA of daily closes."""
    dc = df.groupby("tday")["close"].last()
    ema = dc.ewm(span=21, adjust=False).mean()
    b = np.sign(dc - ema).shift(1).fillna(0)
    return df["tday"].map(b).values


def aplus_signals(df, p=P, use_15m=True, bias_filter=False, sessions=None):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    hm = df["hm"].values
    n = len(df)
    A = atr_fn(df)
    t15 = trend_15m(df)
    bias = daily_bias(df)
    lb = p["swing_lookback"]

    last_ph = prev_ph = last_pl = prev_pl = np.nan
    structure = 0
    locked = 0
    bull_box = bear_box = None        # (top, bottom)
    sigs = []
    for i in range(2 * lb + 2, n):
        # ---- FVGs (formed on this bar)
        if l[i] > h[i - 2] and c[i - 1] > o[i - 1]:
            bull_box = (l[i], h[i - 2])
        if h[i] < l[i - 2] and c[i - 1] < o[i - 1]:
            bear_box = (l[i - 2], h[i])
        # ---- pivots confirmed now (pivot at i-lb)
        q = i - lb
        wh, wl = h[q - lb:q + lb + 1], l[q - lb:q + lb + 1]
        if h[q] == wh.max() and (wh == h[q]).sum() == 1:
            prev_ph, last_ph = last_ph, h[q]
        if l[q] == wl.min() and (wl == l[q]).sum() == 1:
            prev_pl, last_pl = last_pl, l[q]
        if not np.isnan([last_ph, prev_ph, last_pl, prev_pl]).any():
            if last_ph > prev_ph and last_pl > prev_pl:
                structure = 1
            elif last_ph < prev_ph and last_pl < prev_pl:
                structure = -1
        if structure == -1 and not np.isnan(last_ph) and c[i] > last_ph:
            structure = 1
        if structure == 1 and not np.isnan(last_pl) and c[i] < last_pl:
            structure = -1
        if structure == 1 and locked == -1:
            locked = 0
        if structure == -1 and locked == 1:
            locked = 0

        # ---- candle quality
        body = abs(c[i] - o[i]); rng = h[i] - l[i]
        uw = h[i] - max(o[i], c[i]); lw = min(o[i], c[i]) - l[i]
        body_ok = body > rng * p["min_body_pct"]
        atr_ok = rng > A[i] * p["atr_mult"]
        sbull = c[i] > o[i] and body_ok and atr_ok and lw <= body * p["max_opp_wick"] and c[i] > h[i - 1]
        sbear = c[i] < o[i] and body_ok and atr_ok and uw <= body * p["max_opp_wick"] and c[i] < l[i - 1]
        if not (sbull or sbear):
            continue

        near_bull = near_bear = rev_bull = rev_bear = False
        if bull_box:
            top, bot = bull_box
            if l[i] <= top and h[i] >= bot:
                near_bull = True
                rev_bull = c[i - 1] < bot
        if bear_box:
            top, bot = bear_box
            if h[i] >= bot and l[i] <= top:
                near_bear = True
                rev_bear = c[i - 1] > top

        s = 2.5
        if sbull and bull_box and c[i] > bull_box[0]: s += 1.5
        if sbear and bear_box and c[i] < bear_box[1]: s += 1.5
        if rev_bull and sbull: s += p["reversion_bonus"]
        if rev_bear and sbear: s += p["reversion_bonus"]
        d = 1 if sbull else -1
        if structure == d: s += p["structure_bonus"]
        if structure == -d: s -= 1.3
        if use_15m:
            if t15[i] == d: s += p["mtf_bonus"]
            if t15[i] == -d: s -= 1.4
        if near_bull or near_bear: s += 2.4
        s += 1.0                                   # delta always agrees with candle colour
        if s < p["min_score"] or locked == d:
            continue
        locked = d
        if bias_filter and bias[i] != d:
            continue
        if sessions and not any(a <= hm[i] < b for a, b in sessions):
            continue
        if d == 1:
            stop = (bull_box[1] - 3 * TICK) if near_bull and bull_box else min(l[i], l[i - 1]) - rng * 0.1
        else:
            stop = (bear_box[0] + 3 * TICK) if near_bear and bear_box else max(h[i], h[i - 1]) + rng * 0.1
        sigs.append((i, d, stop, s))
    return sigs
