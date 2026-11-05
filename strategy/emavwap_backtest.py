"""Backtest the 9 EMA x VWAP engine on archived NQ 5m bars (all sessions)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import emavwap as E

HERE = os.path.dirname(os.path.abspath(__file__))

def load_5m():
    a = json.load(open(os.path.join(HERE, "archive", "claude_ohlcv_5m_2024_2026.json")))
    df = pd.DataFrame(a, columns=["ts", "open", "high", "low", "close", "volume"])
    live = os.path.join(HERE, "..", "data", "NQ_5m.csv")
    if os.path.exists(live):
        b = pd.read_csv(live)[["ts", "open", "high", "low", "close", "volume"]]
        df = pd.concat([df, b])
    df = df.drop_duplicates("ts", keep="last").sort_values("ts")
    df.index = pd.to_datetime(df.ts, unit="s", utc=True)
    return df.drop(columns="ts")

def session(ts):
    t = ts.tz_convert(E.ET); m = t.hour * 100 + t.minute
    if m >= 1800 or m < 200: return "Asia (6pm-2am ET)"
    if m < 830: return "London (2-8:30am ET)"
    if m < 1200: return "NY AM (8:30-12 ET)"
    return "NY PM (12-5pm ET)"

def stats(tr, label):
    if not tr: return dict(label=label, trades=0)
    u = np.array([t["usd"] for t in tr]); eq = np.cumsum(u)
    dd = (eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    w = u[u > 0]; ls = u[u <= 0]
    streak = cur = 0
    for x in u:
        cur = cur + 1 if x <= 0 else 0; streak = max(streak, cur)
    top = np.sort(u)[::-1]
    return dict(label=label, trades=len(u), win=f"{(u>0).mean():.0%}", pf=round(w.sum()/-ls.sum(), 2) if ls.sum() else None,
                net=round(u.sum()), avg=round(u.mean(), 1), maxdd=round(dd), lose_streak=streak,
                best=round(top[0]), net_ex_top5=round(u.sum() - top[:5].sum()),
                avg_stop_pts=round(np.mean([t["stop_pts"] for t in tr]), 1))

if __name__ == "__main__":
    df = load_5m()
    print("bars", len(df), df.index[0], "->", df.index[-1])
    out = []
    for mode, lab in (("plan", "Plan: 3 micros, 2R / 4R-or-warning / runner"),
                      ("single", "1 micro, stop + trend exit (indicator as-is)"),
                      ("classic", "1 micro, classic v2.1 exit (9 EMA)")):
        tr, ev, _ = E.run(df, exit_mode=mode)
        out.append(stats(tr, lab))
        if mode == "plan":
            plan = tr
    print(pd.DataFrame(out).to_string(index=False))
    t = pd.DataFrame([dict(year=x["entry_ts"].year, sess=session(x["entry_ts"]), side="L" if x["d"]==1 else "S", **x) for x in plan])
    print("\nPLAN by year"); print(pd.DataFrame([stats([plan[i] for i in g.index], y) for y, g in t.groupby("year")]).to_string(index=False))
    print("\nPLAN by session"); print(pd.DataFrame([stats([plan[i] for i in g.index], s) for s, g in t.groupby("sess")]).to_string(index=False))
    print("\nPLAN by side"); print(pd.DataFrame([stats([plan[i] for i in g.index], s) for s, g in t.groupby("side")]).to_string(index=False))
    print("\nexit reasons (legs)"); print(pd.Series([x["why"].split(" ")[0] for tt in plan for x in tt["exits"]]).value_counts().to_string())
    print("\nmax R reached: median", np.median([x["maxR"] for x in plan]), " share >=4R", f"{np.mean([x['maxR']>=4 for x in plan]):.0%}", " share >=8R", f"{np.mean([x['maxR']>=8 for x in plan]):.0%}")
    t.drop(columns=["exits","legs"]).to_csv(os.path.join(HERE, "emavwap_plan_trades.csv"), index=False)
