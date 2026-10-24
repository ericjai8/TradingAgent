"""Simulated Lucid Flex accounts for the paper bot (25K and 50K, 50% consistency).

Eval rules (Flex, as researched Oct 2026). Prices from Eric's checkout with the
50% consistency add-on and a 30% coupon: 25K $50.40 (reset $50), 50K $90.30 (reset $90).
  25K: target $1,250, max loss $1,000 trailing the highest END-OF-DAY balance,
       daily loss limit $600 (soft: ends the day), up to 20 micros
  50K: target $3,000, max loss $2,000 EOD trailing, daily loss limit $1,200
  pass when balance >= target AND best day <= 50% of total profit
  a breach is checked against intraday equity (worst point of each trade)
Funded (after a pass): fresh balance, same max loss; the floor stops trailing
at +$100 and locks there after the first payout. Payout when 5 profit days
(>= $100 on 25K / $150 on 50K) since the last payout and 50% of profit >= $500:
request min(cap, 50% of profit), you keep 90%. Caps: $1,000 / $2,000; after 5
payouts the account moves to live (end of the simulation for that account).
A blown account restarts as a new eval the next session (reset fee counted).
"""
import json
import os

FLEX = {
    "25K": dict(target=1250, mll=1000, dll=600, cons=0.50, fee=50.40, reset=50, minday=100, cap=1000),
    "50K": dict(target=3000, mll=2000, dll=1200, cons=0.50, fee=90.30, reset=90, minday=150, cap=2000),
}
START = "2026-10-26"
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accounts.json")


def _fresh(stage, attempt, fees):
    return dict(stage=stage, attempt=attempt, balance=0.0, peak_eod=0.0, best_day=0.0,
                days=0, profit_days=0, payouts=0, locked=False, fees=fees, paid=0.0)


def load():
    try:
        return json.load(open(PATH))
    except Exception:
        return {"start": START, "booked": [], "history": [],
                "accounts": {a: _fresh("eval", 1, FLEX[a]["fee"]) for a in FLEX}}


def save(db):
    json.dump(db, open(PATH, "w"), indent=1)


def floor(a, s):
    f = FLEX[a]
    if s["stage"] == "funded":
        return 100.0 if s["locked"] else min(s["peak_eod"] - f["mll"], 100.0)
    return s["peak_eod"] - f["mll"]


def cushion(a, s):
    """Dollars between the current balance and the max-loss floor."""
    return s["balance"] - floor(a, s)


def book_day(db, date, pnl, worst):
    """Apply one session. pnl/worst: {acct: dollars}; worst = lowest intraday
    equity change that day (<= 0). Returns Discord lines."""
    if date in db["booked"] or date < db["start"]:
        return []
    lines = []
    for a, s in db["accounts"].items():
        if s["stage"] == "live":
            continue
        f = FLEX[a]
        p, w = pnl.get(a, 0.0), min(worst.get(a, 0.0), pnl.get(a, 0.0), 0.0)
        fl = floor(a, s)
        if s["balance"] + w <= fl:
            lines.append(f"**{a} {'eval' if s['stage'] == 'eval' else 'funded'} #{s['attempt']} BLOWN** on {date}: "
                         f"equity fell to ${s['balance'] + w:+,.0f} vs floor ${fl:+,.0f}. Restarting as a new eval "
                         f"(reset ${f['reset']}).")
            db["history"].append(dict(date=date, acct=a, stage=s["stage"], attempt=s["attempt"], pnl=p, event="blown"))
            db["accounts"][a] = _fresh("eval", s["attempt"] + 1, s["fees"] + f["reset"])
            db["accounts"][a]["paid"] = s["paid"]
            continue
        s["balance"] += p
        if p != 0:
            s["days"] += 1
        s["best_day"] = max(s["best_day"], p)
        s["peak_eod"] = max(s["peak_eod"], s["balance"])
        event = ""
        if s["stage"] == "eval":
            if s["balance"] >= f["target"] and s["best_day"] <= f["cons"] * s["balance"]:
                event = "passed"
                lines.append(f"**{a} eval #{s['attempt']} PASSED** on {date}: balance ${s['balance']:+,.0f} in "
                             f"{s['days']} trading days, best day ${s['best_day']:,.0f}. Now trading the funded account.")
                db["history"].append(dict(date=date, acct=a, stage="eval", attempt=s["attempt"], pnl=p, event=event))
                nf = _fresh("funded", s["attempt"], s["fees"]); nf["paid"] = s["paid"]
                db["accounts"][a] = nf
                continue
        else:
            if p >= f["minday"]:
                s["profit_days"] += 1
            req = min(f["cap"], 0.5 * s["balance"])
            if s["profit_days"] >= 5 and req >= 500:
                s["paid"] += 0.9 * req; s["balance"] -= req; s["payouts"] += 1
                s["profit_days"] = 0; s["locked"] = True; event = "payout"
                lines.append(f"**{a} funded PAYOUT #{s['payouts']}** on {date}: requested ${req:,.0f}, "
                             f"you keep ${0.9 * req:,.0f}. Total paid so far ${s['paid']:,.0f}.")
                if s["payouts"] >= 5:
                    s["stage"] = "live"
                    lines.append(f"**{a}: 5 payouts reached, account moves to live.** Simulation ends for this account.")
        db["history"].append(dict(date=date, acct=a, stage=s["stage"], attempt=s["attempt"], pnl=p, event=event))
    db["booked"].append(date)
    return lines


def status_lines(db):
    out = []
    for a, s in db["accounts"].items():
        f = FLEX[a]
        if s["stage"] == "eval":
            need = max(f["target"], s["best_day"] / f["cons"] if s["best_day"] > 0 else 0)
            out.append(f"{a} eval #{s['attempt']}: balance ${s['balance']:+,.0f} / target ${f['target']:,} "
                       f"(need ${need:,.0f} with best day ${s['best_day']:,.0f}) | floor ${floor(a, s):+,.0f} "
                       f"(cushion ${cushion(a, s):,.0f}) | {s['days']} trading days | fees so far ${s['fees']}")
        elif s["stage"] == "funded":
            out.append(f"{a} funded: balance ${s['balance']:+,.0f} | floor ${floor(a, s):+,.0f} | profit days "
                       f"{s['profit_days']}/5 | payouts {s['payouts']} (${s['paid']:,.0f} kept) | fees ${s['fees']}")
        else:
            out.append(f"{a}: moved to live after 5 payouts. Kept ${s['paid']:,.0f}, fees ${s['fees']}.")
    return out
