import json, re, csv, os, glob
from collections import defaultdict

def load_bot_log(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"- INFO - (\{.+\})$", line)
            if m:
                try:
                    entries.append(json.loads(m.group(1)))
                except:
                    pass
    return entries

def build_hands(events):
    hands = defaultdict(lambda: {
        "preflop": None, "discard": None,
        "postflop": [], "result": None,
        "cards": [], "community": []
    })
    for e in events:
        h  = e.get("hand", -1)
        ev = e.get("event")
        if ev == "hand_start":
            hands[h]["cards"] = e.get("hole_cards", [])
        elif ev == "preflop_decision": hands[h]["preflop"]  = e
        elif ev == "discard_decision": hands[h]["discard"]  = e
        elif ev == "postflop_decision":
            hands[h]["postflop"].append(e)
            if "community" in e:
                hands[h]["community"] = e["community"]
        elif ev == "hand_result":
            hands[h]["result"] = e
            if "community" in e and e["community"]:
                hands[h]["community"] = e["community"]
    return hands

def get_streets(postflop):
    by = {}
    for p in postflop:
        sn = (p.get("street_name") or p.get("street") or "?").lower()
        by[sn] = p
    return by.get("flop"), by.get("turn"), by.get("river")

loss_cases = []

for fpath in glob.glob("match logs/match_*_bot.txt"):
    events = load_bot_log(fpath)
    hands  = build_hands(events)
    for hnum, h in hands.items():
        r = h["result"]
        if not r or not r.get("showdown"): continue
        if r.get("outcome") != "loss": continue
        
        fl, tu, ri = get_streets(h["postflop"])
        
        for street, name in [(ri, "river"), (tu, "turn"), (fl, "flop")]:
            if street and street.get("adj_equity") is not None:
                eq = street.get("adj_equity")
                if eq >= 0.70:
                    loss_cases.append({
                        "match": fpath,
                        "hand": hnum,
                        "street": name,
                        "cards": r.get("our_kept_cards", h["cards"]),
                        "opp_cards": r.get("opp_kept_cards", []),
                        "community": r.get("community", h["community"]),
                        "equity": eq,
                        "pnl": r.get("pnl", 0),
                        "street_comm": street.get("community", [])
                    })
                    break

print(f"Found {len(loss_cases)} bad beats where we had >=70% equity late in the hand.")
loss_cases.sort(key=lambda x: x["pnl"])
for lc in loss_cases[:20]:
    print(f"Match {lc['match']} Hand {lc['hand']:4d} | {lc['street']} Eq: {lc['equity']:.3f} | PnL: {lc['pnl']} | We: {lc['cards']} | Comm({lc['street']}): {lc['street_comm']} | FinalComm: {lc['community']} | Opp: {lc['opp_cards']}")
