"""
Sweep & reclaim at liquidity levels — rule spec (shorts shown; longs mirror):

  Levels (known before the session): prior-day high/low, overnight high/low
  (18:00 -> 09:30 ET), prior-week high/low.
  Window: signals only 09:30-11:30 ET. Flat by 15:55.

  1. SWEEP     a bar trades above a level.
  2. RECLAIM   within `reclaim_bars`, a bar CLOSES back below the level.
               (if price closes above and stays, the sweep is accepted -> no trade)
  3. DISPLACE  within `disp_bars` after the reclaim, a strong bearish candle
               (body > 60% range, opp. wick <= 0.7 body, range > atr_mult*ATR)
               that leaves a bearish FVG (bar after it: high < low two bars back).
  4. ENTRY     sell limit at the FVG's near edge, valid `limit_bars`; fills only if
               price trades 1 tick through it.
  5. STOP      1 tick above the sweep's extreme.
  6. TARGET    'level': nearest level below entry that is >= min_rr * risk away
               (skip the trade if none); or a fixed R multiple.
  Max `max_per_day` trades, each level once per day, one position at a time.
  Daily bias (prior close vs 21-day EMA) optional.
"""
import numpy as np
FUNNEL = {}
def _f(k): FUNNEL[k] = FUNNEL.get(k, 0) + 1
ATTEMPTS = []
def _a(**kw): ATTEMPTS.append(kw)
import pandas as pd
from lab import TICK, SLIP_TICKS, atr
from aplus import daily_bias


def day_levels(df):
    g = df.groupby("tday")
    rth = df[(df.hm >= 930) & (df.hm < 1600)].groupby("tday")
    on = df[(df.hm >= 1800) | (df.hm < 930)].groupby("tday")
    lv = pd.DataFrame({
        "PDH": rth["high"].max().shift(1), "PDL": rth["low"].min().shift(1),
        "ONH": on["high"].max(), "ONL": on["low"].min(),
    })
    wk = df.assign(w=df["tday"].dt.to_period("W-FRI")).groupby("w")
    wh, wl = wk["high"].max().shift(1), wk["low"].min().shift(1)
    per = lv.index.to_period("W-FRI")
    lv["PWH"] = wh.reindex(per).values
    lv["PWL"] = wl.reindex(per).values
    return lv


def sweep_trades(df, target="level", min_rr=2.0, bias_filter=True,
                 reclaim_bars=3, disp_bars=6, limit_bars=12, atr_mult=0.8,
                 max_per_day=2, window=(930, 1130), flat_hm=1555, stop_mode="sweep", max_stop=None):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    hm, tday = df["hm"].values, df["tday"].values
    A = atr(df)
    bias = daily_bias(df)
    lv = day_levels(df)
    slip = SLIP_TICKS * TICK
    trades = []
    for day, idx in df.groupby("tday").indices.items():
        if day not in lv.index:
            continue
        L = lv.loc[day].dropna().to_dict()
        if not L:
            continue
        highs = {k: v for k, v in L.items() if k.endswith("H")}
        lows = {k: v for k, v in L.items() if k.endswith("L")}
        used, taken, i_pos = set(), 0, 0
        bars = idx[(hm[idx] >= window[0]) & (hm[idx] < 1555)]
        if len(bars) == 0:
            continue
        k = 0
        while k < len(bars) and taken < max_per_day:
            i = bars[k]
            if hm[i] >= window[1]:
                break
            setup = None
            for side, levels in ((-1, highs), (1, lows)):
                if setup:
                    break
                for name, lvl in levels.items():
                    if name in used:
                        continue
                    swept = h[i] > lvl if side == -1 else l[i] < lvl
                    if not swept:
                        continue
                    _f("1 level swept in window")
                    if bias_filter and bias[i] != side:
                        used.add(name)
                        _f("x against daily bias")
                        _a(day=day, side=side, level=name, lvl=lvl, i_sweep=i, stage="against bias")
                        continue
                    setup = (side, name, lvl)
                    break
            if not setup:
                k += 1
                continue
            d, name, lvl = setup
            used.add(name)
            # --- reclaim
            ext = h[i] if d == -1 else l[i]
            rec = None
            for j in range(i, min(i + reclaim_bars + 1, idx[-1] + 1)):
                ext = max(ext, h[j]) if d == -1 else min(ext, l[j])
                if (d == -1 and c[j] < lvl) or (d == 1 and c[j] > lvl):
                    rec = j
                    break
            _f("2 sweep with bias")
            if rec is None:
                _a(day=day, side=d, level=name, lvl=lvl, i_sweep=i, stage="no reclaim")
                k += 1
                continue
            _f("3 reclaimed")
            # --- displacement + FVG
            fvg = None
            for j in range(rec, min(rec + disp_bars + 1, idx[-1])):
                ext = max(ext, h[j]) if d == -1 else min(ext, l[j])
                body, rng = abs(c[j] - o[j]), h[j] - l[j]
                if rng <= 0:
                    continue
                uw, lw = h[j] - max(o[j], c[j]), min(o[j], c[j]) - l[j]
                strong = body > 0.6 * rng and rng > atr_mult * A[j] and (
                    (d == -1 and c[j] < o[j] and uw <= 0.7 * body) or
                    (d == 1 and c[j] > o[j] and lw <= 0.7 * body))
                t = j + 1   # FVG confirms when the next bar closes
                if strong and t <= idx[-1]:
                    if d == -1 and h[t] < l[j - 1]:
                        fvg, conf = h[t], t      # near edge for a short
                    elif d == 1 and l[t] > h[j - 1]:
                        fvg, conf = l[t], t
                    if fvg is not None:
                        break
            if fvg is None:
                _a(day=day, side=d, level=name, lvl=lvl, i_sweep=i, i_rec=rec, stage="no displacement")
                k += 1
                continue
            _f("4 displacement + FVG")
            jd = conf - 1
            if stop_mode == "sweep":
                stop = ext + TICK if d == -1 else ext - TICK
            elif stop_mode == "disp":      # beyond the displacement candle
                stop = h[jd] + TICK if d == -1 else l[jd] - TICK
            else:                          # "fvg": beyond the FVG's far edge
                stop = l[jd - 1] + TICK if d == -1 else h[jd - 1] - TICK
            entry = fvg
            risk = (entry - stop) * d
            if max_stop is not None and risk > max_stop:
                k += 1
                continue
            if risk <= 0:
                _a(day=day, side=d, level=name, lvl=lvl, i_sweep=i, i_rec=rec, i_disp=conf - 1, stage="no displacement")
                k += 1
                continue
            # --- target
            if target == "level":
                if d == -1:
                    cands = [v for v in lows.values() if entry - v >= min_rr * risk]
                    tgt = max(cands) if cands else None
                else:
                    cands = [v for v in highs.values() if v - entry >= min_rr * risk]
                    tgt = min(cands) if cands else None
                if tgt is None:
                    _a(day=day, side=d, level=name, lvl=lvl, i_sweep=i, i_rec=rec, i_disp=conf - 1,
                       entry=entry, stop=stop, stage="target under 2R")
                    k += 1
                    continue
                _f("5 next level >= 2R away")
            else:
                tgt = entry + d * float(target) * risk
            # --- limit fill
            fill = None
            for j in range(conf + 1, min(conf + 1 + limit_bars, idx[-1] + 1)):
                if hm[j] >= flat_hm and hm[j] < 1800:
                    break
                # Price must pass the entry before it can reach the stop, so a bar
                # that touches both is a fill (then stopped on the same bar).
                if (d == -1 and h[j] >= entry + TICK) or (d == 1 and l[j] <= entry - TICK):
                    fill = j
                    break
                if (d == -1 and h[j] >= stop) or (d == 1 and l[j] <= stop):
                    break                     # gapped to the stop without a fill
            if fill is None:
                _a(day=day, side=d, level=name, lvl=lvl, i_sweep=i, i_rec=rec, i_disp=conf - 1,
                   i_conf=conf, entry=entry, stop=stop, target=tgt, stage="limit not filled")
                k = np.searchsorted(bars, conf) + 1
                continue
            _f("6 limit filled")
            # --- manage
            exit_px, j = None, fill
            while j <= idx[-1]:
                if hm[j] >= flat_hm and hm[j] < 1800:
                    exit_px = o[j] - d * slip
                    break
                if (d == -1 and h[j] >= stop) or (d == 1 and l[j] <= stop):
                    px = max(o[j], stop) if (d == -1 and j > fill) else stop
                    if d == 1:
                        px = min(o[j], stop) if j > fill else stop
                    exit_px = px - d * slip
                    break
                if j > fill and ((d == -1 and l[j] <= tgt - TICK) or (d == 1 and h[j] >= tgt + TICK)):
                    exit_px = tgt
                    break
                j += 1
            if exit_px is None:
                exit_px, j = c[idx[-1]], idx[-1]
            pts = (exit_px - entry) * d
            trades.append(dict(tday=pd.Timestamp(day), dir=d, level=name, entry=entry,
                               stop=stop, target=tgt, risk_pts=risk, pts=pts,
                               R=pts / risk, plan_rr=abs(tgt - entry) / risk,
                               entry_hm=hm[fill], bars=j - fill + 1,
                               i_sweep=i, i_rec=rec, i_disp=conf - 1, i_conf=conf,
                               i_fill=fill, i_exit=j, exit_px=exit_px))
            _a(day=day, side=d, level=name, lvl=lvl, i_sweep=i, i_rec=rec, i_disp=conf - 1,
               i_conf=conf, i_fill=fill, i_exit=j, entry=entry, stop=stop, target=tgt,
               exit_px=exit_px, R=pts / risk, stage="traded")
            taken += 1
            k = np.searchsorted(bars, j) + 1
    return pd.DataFrame(trades)
