"""
Daily hunt: merge archived MNQ bars with the repo's Yahoo NQ bars, run the
Sweep & Reclaim rules on every session, rebuild the replay page, and print a
summary of the most recent session.

Run from the strategy/ folder:  python daily_hunt.py
Writes: replay.html (the page), replay_data.json, summary.md
"""
import json, os
import pandas as pd
import sweep
from lab import load_bars, COMMISSION_RT, PT_VALUE
from sweep import day_levels
from aplus import daily_bias

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)


def merged_15m():
    old = pd.DataFrame(json.load(open(P("archive", "claude_ohlcv_15m_2020_2026.json"))),
                       columns=["ts", "open", "high", "low", "close", "volume"])
    y = pd.read_csv(P("..", "data", "NQ_15m.csv"))[["ts", "open", "high", "low", "close", "volume"]]
    m = pd.concat([old, y[y.ts > old.ts.max()]]).drop_duplicates("ts").sort_values("ts")
    path = P("merged_15m.json")
    json.dump(m.values.tolist(), open(path, "w"))
    return load_bars(path)


def build(df):
    sweep.ATTEMPTS.clear()
    sweep.sweep_trades(df, "level", 2.0, True)
    att = pd.DataFrame(sweep.ATTEMPTS)
    lv = day_levels(df)
    bias = pd.Series(daily_bias(df), index=df.index)
    hm = df.hm.values
    days = []
    for day, idx in df.groupby("tday").indices.items():
        rth = idx[(hm[idx] >= 930) & (hm[idx] < 1600)]
        if len(rth) < 10:
            continue
        show = idx[(hm[idx] >= 600) & (hm[idx] < 1615)]
        a0 = show[0]
        bars = [[round(float(df.open[i]), 2), round(float(df.high[i]), 2),
                 round(float(df.low[i]), 2), round(float(df.close[i]), 2)] for i in show]
        L = {k: (None if pd.isna(v) else round(float(v), 2)) for k, v in lv.loc[day].items()} if day in lv.index else {}
        ev = []
        sub = att[att.day == day] if len(att) else att
        for r in sub.to_dict("records"):
            e = {"side": int(r["side"]), "level": r["level"], "stage": r["stage"]}
            for k in ["i_sweep", "i_rec", "i_disp", "i_conf", "i_fill", "i_exit"]:
                if k in r and not pd.isna(r[k]):
                    e[k[2:]] = int(r[k]) - int(a0)
            for k in ["entry", "stop", "target", "exit_px"]:
                if k in r and not pd.isna(r[k]):
                    e[k] = round(float(r[k]), 2)
            if r["stage"] == "traded":
                risk = abs(r["entry"] - r["stop"])
                e["R"] = round(float(r["R"] - COMMISSION_RT / (PT_VALUE * risk)), 2)
            ev.append(e)
        days.append({"d": pd.Timestamp(day).strftime("%Y-%m-%d"), "t0": df.time[a0].strftime("%H:%M"),
                     "b": int(bias[rth[0]]), "L": L, "bars": bars, "ev": ev})
    return days


TEXT = {"against bias": "swept, but against the daily bias (ignored)",
        "no reclaim": "swept, but never closed back inside within 3 bars",
        "no displacement": "swept and reclaimed, but no strong candle with an FVG",
        "target under 2R": "full setup formed, but no level was 2R away (skipped)",
        "limit not filled": "setup formed, limit placed, never filled",
        "traded": "TRADED"}


def summary(days):
    d = days[-1]
    bias = {1: "long only", -1: "short only", 0: "none"}[d["b"]]
    lines = [f"Session {d['d']}: daily bias {bias}."]
    if not d["ev"]:
        lines.append("No level was swept in the 9:30-11:30 ET window. No setup.")
    for e in d["ev"]:
        side = "long" if e["side"] == 1 else "short"
        t = e.get("sweep")
        line = f"- {e['level']} ({side}): {TEXT[e['stage']]}"
        if e["stage"] in ("limit not filled", "traded", "target under 2R") and "entry" in e:
            line += f"; entry {e['entry']:,.2f}, stop {e['stop']:,.2f}"
            if "target" in e:
                line += f", target {e['target']:,.2f}"
        if e["stage"] == "traded":
            line += f"; result {e['R']:+.2f}R"
        lines.append(line)
    tr = [e for x in days for e in x["ev"] if e["stage"] == "traded"]
    recent = [e for x in days if x["d"] >= "2024-01-01" for e in x["ev"] if e["stage"] == "traded"]
    lines.append(f"Running totals: {len(tr)} trades since 2020, avg {sum(e['R'] for e in tr)/len(tr):+.2f}R; "
                 f"since 2024: {len(recent)} trades, avg {sum(e['R'] for e in recent)/max(len(recent),1):+.2f}R.")
    return "\n".join(lines)


if __name__ == "__main__":
    df = merged_15m()
    days = build(df)
    data = json.dumps(days, separators=(",", ":"))
    open(P("replay_data.json"), "w").write(data)
    t = open(P("replay_template.html")).read()
    open(P("replay.html"), "w").write(t.replace("__DATA__", data))
    s = summary(days)
    open(P("summary.md"), "w").write(s)
    print(s)
