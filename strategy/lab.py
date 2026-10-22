"""
Strategy Lab — clean bar-based backtester for MNQ.

Rules that keep it honest:
  * Signals use only bars that have CLOSED. Entry is the NEXT bar's open.
  * The entry bar itself is checked for a stop-out.
  * If stop and target are both touched in one bar, the stop wins.
  * Targets must trade 1 tick THROUGH to count as filled.
  * Market exits (stops, trails, time exits) pay slippage; so do entries.
  * Trailing stops only move using bars that have already closed.
"""
import json
import numpy as np
import pandas as pd

TICK = 0.25
PT_VALUE = 2.0          # MNQ: $2 per point per micro
COMMISSION_RT = 1.00    # $ per micro, round trip (assumption)
SLIP_TICKS = 1          # per market fill (entry, stop, time exit)


# ------------------------------------------------------------------ data
def load_bars(*files):
    rows = []
    for f in files:
        rows += json.load(open(f))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
    # Futures trading day: 18:00 ET rolls to the next session date
    td = (df["time"] + pd.Timedelta(hours=6)).dt.date
    df["tday"] = pd.to_datetime(td)
    df["hm"] = df["time"].dt.hour * 100 + df["time"].dt.minute
    return df


def atr(df, n=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values


# ----------------------------------------------------------- signal logic
def tight_swing_signals(df, piv=3, max_risk_atr=0.4, rth_only=True):
    """
    Confirmed higher-low in an uptrend (HH + HL) -> long.
    Confirmed lower-high in a downtrend (LH + LL) -> short.
    A pivot at bar p is only known at bar p+piv (when it CONFIRMS).
    Trade only if the swing (= stop) is within max_risk_atr * ATR of the
    confirmation bar's close.
    Returns list of (signal_bar_index, direction, stop_price).
    """
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    a = atr(df)
    hm = df["hm"].values
    n = len(df)
    sigs = []
    last_hi = [np.nan, np.nan]   # previous, latest confirmed swing highs
    last_lo = [np.nan, np.nan]
    for i in range(2 * piv, n):
        p = i - piv                       # candidate pivot bar, confirmed now
        win_h = h[p - piv:p + piv + 1]
        win_l = l[p - piv:p + piv + 1]
        new_hi = h[p] == win_h.max() and (win_h == h[p]).sum() == 1
        new_lo = l[p] == win_l.min() and (win_l == l[p]).sum() == 1
        if new_hi:
            last_hi = [last_hi[1], h[p]]
        if new_lo:
            last_lo = [last_lo[1], l[p]]
            up = last_hi[1] > last_hi[0] and last_lo[1] > last_lo[0]
            risk = c[i] - l[p]
            if up and 0 < risk <= max_risk_atr * a[i]:
                sigs.append((i, 1, l[p]))
        if new_hi:
            dn = last_hi[1] < last_hi[0] and last_lo[1] < last_lo[0]
            risk = h[p] - c[i]
            if dn and 0 < risk <= max_risk_atr * a[i]:
                sigs.append((i, -1, h[p]))
    if rth_only:
        # signal must confirm 09:30-15:30 ET so the entry happens in RTH
        sigs = [s for s in sigs if 930 <= hm[s[0]] < 1530]
    return sigs


# ------------------------------------------------------------- execution
def simulate(df, sigs, exit_mode="trail", target_r=2.0, flat_hm=1555,
             one_at_a_time=True):
    """
    exit_mode: 'target' -> fixed target_r * R target
               'trail'  -> once +1R reached, stop trails 1R behind best price
    Returns a DataFrame of trades with R and points (after slippage; before
    commission, which is applied per contract later).
    """
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    hm, tday, t = df["hm"].values, df["tday"].values, df["time"].values
    n = len(df)
    slip = SLIP_TICKS * TICK
    out, busy_until = [], -1
    for (i, d, stop) in sigs:
        e = i + 1
        if e >= n or (one_at_a_time and e <= busy_until):
            continue
        if tday[e] != tday[i]:
            continue
        entry = o[e] + d * slip
        risk = (entry - stop) * d
        if risk <= 0:            # gapped through the stop: skip
            continue
        tgt = entry + d * target_r * risk
        best = entry
        cur_stop = stop
        mae = 0.0
        exit_px, j, why = None, e, None
        while j < n:
            # time exit: flat before close (or day boundary)
            if (hm[j] >= flat_hm and hm[j] < 1800) or tday[j] != tday[e]:
                exit_px, why = o[j] - d * slip, "time"
                break
            lo_adv = (l[j] - entry) if d == 1 else (entry - h[j])
            mae = min(mae, lo_adv)
            # stop first (conservative)
            if (d == 1 and l[j] <= cur_stop) or (d == -1 and h[j] >= cur_stop):
                fill = min(o[j], cur_stop) if d == 1 else max(o[j], cur_stop)
                exit_px, why = fill - d * slip, "stop" if cur_stop == stop else "trail"
                break
            if exit_mode == "target":
                if (d == 1 and h[j] >= tgt + TICK) or (d == -1 and l[j] <= tgt - TICK):
                    exit_px, why = tgt, "target"
                    break
            # update trail using this CLOSED bar (applies from next bar)
            best = max(best, h[j]) if d == 1 else min(best, l[j])
            if exit_mode == "trail" and (best - entry) * d >= risk:
                new_stop = best - d * risk
                cur_stop = max(cur_stop, new_stop) if d == 1 else min(cur_stop, new_stop)
            j += 1
        if exit_px is None:
            break
        pts = (exit_px - entry) * d
        out.append(dict(entry_time=t[e], tday=tday[e], dir=d, entry=entry,
                        stop=stop, risk_pts=risk, exit=exit_px, why=why,
                        pts=pts, R=pts / risk, mae_pts=mae, bars=j - e + 1))
        busy_until = j
    return pd.DataFrame(out)


def random_control(df, trades, seed=0, exit_mode="trail"):
    """Same days, same count, same stop distances, random time + direction."""
    rng = np.random.default_rng(seed)
    hm, tday = df["hm"].values, df["tday"].values
    rth = np.where((hm >= 930) & (hm < 1530))[0]
    sigs = []
    for _, tr in trades.iterrows():
        i = rng.choice(rth)
        d = rng.choice([1, -1])
        sigs.append((i, d, df["close"].values[i] - d * tr["risk_pts"]))
    sigs.sort()
    return simulate(df, sigs, exit_mode=exit_mode)


def summarize(tr, label=""):
    if len(tr) == 0:
        return dict(label=label, trades=0)
    R = tr["R"].values
    wins = R > 0
    gp, gl = R[wins].sum(), -R[~wins].sum()
    days = tr["tday"].nunique()
    return dict(label=label, trades=len(tr), win_rate=wins.mean(),
                avg_R=R.mean(), median_R=np.median(R),
                profit_factor=gp / gl if gl > 0 else np.inf,
                avg_risk_pts=tr["risk_pts"].mean(),
                trades_per_day=len(tr) / max(days, 1))
