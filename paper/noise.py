"""Noise-area momentum (prop-safe version) for the paper bot, on 1-minute bars.
Same rules as the backtest (strategy_lab/noise_safe.py, stop 1.0x, max 2/day,
stop after a loss):
  sigma(m)  = mean over the previous 14 sessions of |close(m)/open - 1|, where m is
              the minute of the session (bar starting at 9:30 + m)
  upper     = max(today's 9:30 open, prior session close) * (1 + sigma(m))
  lower     = min(...) * (1 - sigma(m))
  every 30 min from 10:00 to 15:30 (close of that minute's bar):
    flat and close > upper -> long at next bar open (mirror short)
    in a long and close < max(upper, VWAP) -> exit at next bar open
  hard stop = 1.0 x sigma(m) x open from entry, checked every minute
  flat at 15:55; max 2 trades a day; no new entry after a losing trade
Fills: 1 tick slippage on entries and exits.
"""
import numpy as np
import pandas as pd

TICK = 0.25
DEC = set(range(30, 390, 30))


def _prep(df, lookback=14):
    """df: 1m bars with a tz-aware America/New_York DatetimeIndex and columns
    open, high, low, close, volume."""
    d = df.copy()
    d["time"] = d.index
    hm = d.time.dt.hour * 100 + d.time.dt.minute
    d = d[(hm >= 930) & (hm < 1600)].copy()
    d["tday"] = d.time.dt.date
    d["mod"] = (d.time.dt.hour - 9) * 60 + d.time.dt.minute - 30
    opens = d.groupby("tday").open.first()
    pclose = d.groupby("tday").close.last().shift(1)
    piv = d.pivot_table(index="tday", columns="mod", values="close", aggfunc="last")
    sigma = (piv.div(opens, axis=0) - 1).abs().rolling(lookback).mean().shift(1)
    tp = (d.high + d.low + d.close) / 3
    vol = d.volume.clip(lower=1)
    d["vwap"] = (tp * vol).groupby(d.tday).cumsum() / vol.groupby(d.tday).cumsum()
    return d, opens, pclose, sigma


def day_trades(df, day, stop_mult=1.0, max_trades=2, stop_after_loss=True, tgt_r=1.5, range_exit=False):
    """Trades for one session `day` (datetime.date) using only bars in df.
    Returns a list of dicts in the bot's trade format; an unfinished trade has
    status 'open'."""
    d, opens, pclose, sigma = _prep(df)
    if day not in sigma.index or pd.isna(pclose.get(day)) or sigma.loc[day].isna().all():
        return []
    g = d[d.tday == day]
    sg_row = sigma.loc[day]
    o, h, l, c, vw, mod = (g[k].values for k in ("open", "high", "low", "close", "vwap", "mod"))
    t = list(g.time)
    op = opens[day]
    up_ref, dn_ref = max(op, pclose[day]), min(op, pclose[day])
    out = []
    pos = 0; n = 0; lost = False; i = 0; N = len(c)
    entry = stop = risk = None; ent_i = sig_i = None

    def rec(status, x=None, xi=None, how=None):
        return dict(strat="Noise area", sig_ts=t[sig_i], d=pos, stop=stop, r=tgt_r,
                    target=(entry + pos * tgt_r * risk) if tgt_r else None,
                    status=status, entry_ts=t[ent_i], entry=entry, risk=risk,
                    exit_ts=(t[xi] if xi is not None else None), exit=x, how=how)

    while i < N:
        m = mod[i]
        if pos:
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                fill = stop if i == ent_i else (min(o[i], stop) if pos == 1 else max(o[i], stop))
                out.append(rec("closed", fill - pos * TICK, i, "stopped out"))
                lost = True; pos = 0; i += 1
                continue
            if tgt_r:
                tg = entry + pos * tgt_r * risk
                if (pos == 1 and h[i] >= tg + TICK) or (pos == -1 and l[i] <= tg - TICK):
                    out.append(rec("closed", tg, i, "target hit"))
                    pos = 0; i += 1
                    continue
            if m >= 385:
                x = o[i] - pos * TICK
                out.append(rec("closed", x, i, "time exit 15:55"))
                lost = lost or (x - entry) * pos < 0
                pos = 0
                break
            if range_exit and m in DEC and i + 1 < N:
                sg = sg_row.get(m, np.nan)
                ub, lb = up_ref * (1 + sg), dn_ref * (1 - sg)
                if (pos == 1 and c[i] < max(ub, vw[i])) or (pos == -1 and c[i] > min(lb, vw[i])):
                    x = o[i + 1] - pos * TICK
                    out.append(rec("closed", x, i + 1, "back inside range/VWAP"))
                    lost = lost or (x - entry) * pos < 0
                    pos = 0; i += 1
                    continue
        elif m in DEC and m <= 360 and i + 1 < N and n < max_trades and not (stop_after_loss and lost):
            sg = sg_row.get(m, np.nan)
            if not np.isnan(sg):
                ub, lb = up_ref * (1 + sg), dn_ref * (1 - sg)
                sig = 1 if c[i] > ub else -1 if c[i] < lb else 0
                if sig:
                    sig_i, ent_i = i, i + 1
                    entry = o[ent_i] + sig * TICK
                    risk = stop_mult * sg * op
                    stop = entry - sig * risk
                    pos = sig; n += 1
        i += 1
    if pos:
        out.append(rec("open"))
    return out
