#!/usr/bin/env python3
"""One-off analysis for match 77659."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
CSV_PATH = BASE / "match_77659_csv.txt"
BOT_PATH = BASE / "match_77659_bot.txt"

OUR_TEAM = 1
OPP_TEAM = 0


def parse_csv():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(next(csv.reader([line])))
    header = "hand_number,street,active_team,team_0_bankroll,team_1_bankroll,action_type,action_amount,action_keep_1,action_keep_2,team_0_cards,team_1_cards,board_cards,team_0_discarded,team_1_discarded,team_0_bet,team_1_bet".split(",")
    # first data row might be header duplicate
    out = []
    for r in rows:
        if r[0] == "hand_number":
            continue
        out.append(dict(zip(header, r)))
    return out


def hand_pnl(by_hand, last_hand):
    """Per-hand team_1 PnL. CSV updates bankroll at first row of next hand; use start-of-hand bankrolls."""
    def s1(h):
        return int(by_hand[h][0]["team_1_bankroll"])

    def e1(h):
        return int(by_hand[h][-1]["team_1_bankroll"])

    pnls = {}
    for h in range(last_hand + 1):
        if h < last_hand:
            pnls[h] = s1(h + 1) - s1(h)
        else:
            pnls[h] = e1(h) - s1(h)
    return pnls


def build_by_hand(rows):
    by_hand = defaultdict(list)
    for r in rows:
        by_hand[int(r["hand_number"])].append(r)
    return by_hand


def opp_preflop_folded(by_hand, h):
    """Opponent (team 0) took FOLD on Pre-Flop."""
    for r in by_hand[h]:
        if r["street"] == "Pre-Flop" and int(r["active_team"]) == OPP_TEAM and r["action_type"] == "FOLD":
            return True
    return False


def find_bleed_start(by_hand, last_hand):
    """First hand H of longest suffix where opp folds preflop every hand; else None."""
    h = last_hand
    while h >= 0 and opp_preflop_folded(by_hand, h):
        h -= 1
    # h is last index before streak, or -1 if full match is streak
    if h == last_hand:
        return None  # last hand: opp did not fold preflop
    return h + 1


def parse_bot():
    preflop = []
    postflop = []
    hand_results = []
    hand_start = []
    for line in open(BOT_PATH, errors="ignore"):
        if '{"event":' not in line:
            continue
        m = re.search(r"\{.*\}\s*$", line)
        if not m:
            continue
        try:
            j = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        ev = j.get("event")
        if ev == "preflop_decision":
            preflop.append(j)
        elif ev == "postflop_decision":
            postflop.append(j)
        elif ev == "hand_result":
            hand_results.append(j)
        elif ev == "hand_start":
            hand_start.append(j)
    return preflop, postflop, hand_results, hand_start


def main():
    rows = parse_csv()
    by_hand = build_by_hand(rows)
    last_hand = max(by_hand)
    pnls = hand_pnl(by_hand, last_hand)
    bleed_H = find_bleed_start(by_hand, last_hand)
    if bleed_H is None:
        bleed_set = set()
        bleed_label = "none"
    else:
        bleed_set = set(range(bleed_H, last_hand + 1))
        bleed_label = str(bleed_H)

    # Real-play hands: exclude bleed-out; optionally exclude hand 0 (all-in dummy)
    real_hands = [h for h in range(0, last_hand + 1) if h not in bleed_set]
    # User asked "real-play" — exclude all-in hand 0 as not standard
    real_hands_no0 = [h for h in real_hands if h != 0]

    preflop_logs, postflop_logs, hand_results, hand_start = parse_bot()
    pos_by_hand = {j["hand"]: j.get("position") for j in hand_start}
    hr_by_hand = {j["hand"]: j for j in hand_results}

    def position_for_hand(h):
        if h >= 1:
            p = pos_by_hand.get(h)
            if p in ("SB", "BB"):
                return p
            # Bot log often omits hand_start; HU alternates from hand 1 = SB
            return "SB" if h % 2 == 1 else "BB"
        return None

    # CSV: our actions
    def normalize_action(a):
        if a in ("CHECK", "CALL", "FOLD", "RAISE"):
            return a
        return None

    pre_counts = defaultdict(int)
    post_counts = defaultdict(int)
    for h, rs in by_hand.items():
        if h in bleed_set:
            continue
        for r in rs:
            if int(r["active_team"]) != OUR_TEAM:
                continue
            st = r["street"]
            act = normalize_action(r["action_type"])
            if act is None:
                continue
            if st == "Pre-Flop":
                pre_counts[act] += 1
            else:
                post_counts[act] += 1

    net_pnl = sum(pnls[h] for h in real_hands_no0)
    wins = sum(1 for h in real_hands_no0 if pnls[h] > 0)
    losses = sum(1 for h in real_hands_no0 if pnls[h] < 0)
    ties = sum(1 for h in real_hands_no0 if pnls[h] == 0)

    sd_hands = [h for h in real_hands_no0 if hr_by_hand.get(h, {}).get("showdown")]
    sd_count = len(sd_hands)
    sd_wins = sum(1 for h in sd_hands if pnls[h] > 0)
    sd_wr = sd_wins / sd_count if sd_count else 0

    # Pot at showdown: use last postflop pot from bot for SD hands, or CSV team bets
    def pot_from_csv(h):
        rs = by_hand[h]
        last = rs[-1]
        return int(last["team_0_bet"]) + int(last["team_1_bet"])

    sd_pots = [pot_from_csv(h) for h in sd_hands]
    avg_sd_pot = sum(sd_pots) / len(sd_pots) if sd_pots else 0

    big_loss = [h for h in real_hands_no0 if pnls[h] <= -50]
    big_win = [h for h in real_hands_no0 if pnls[h] >= 50]

    pnl_sb = sum(pnls[h] for h in real_hands_no0 if position_for_hand(h) == "SB")
    pnl_bb = sum(pnls[h] for h in real_hands_no0 if position_for_hand(h) == "BB")

    # --- Bot log (exclude bleed) ---
    def in_real(h):
        return h not in bleed_set and h != 0  # align with real_hands_no0 for postflop stats

    # Last postflop decision per hand -> strength label + pnl
    last_pf_by_hand = defaultdict(list)
    for j in postflop_logs:
        h = j["hand"]
        if not in_real(h):
            continue
        last_pf_by_hand[h].append(j)
    last_post = {}
    for h, lst in last_pf_by_hand.items():
        lst.sort(key=lambda x: (x.get("street", 0), x.get("ts", 0)))
        last_post[h] = lst[-1]

    strength_pnl = defaultdict(lambda: [0, 0])  # sum, count
    for h, j in last_post.items():
        st = j.get("strength", "?")
        strength_pnl[st][0] += pnls.get(h, 0)
        strength_pnl[st][1] += 1

    # Strength distribution + avg adj_equity (postflop decisions, all streets)
    str_dist = defaultdict(lambda: [0, 0.0])
    for j in postflop_logs:
        h = j["hand"]
        if h in bleed_set:
            continue
        st = j.get("strength", "?")
        str_dist[st][0] += 1
        str_dist[st][1] += j.get("adj_equity") or 0

    pre_reason = defaultdict(int)
    for j in preflop_logs:
        if j["hand"] in bleed_set:
            continue
        pre_reason[j.get("reason", "?")] += 1

    # Avg raw/adj by street (postflop)
    street_eq = defaultdict(lambda: [[], []])
    for j in postflop_logs:
        if j["hand"] in bleed_set:
            continue
        sn = j.get("street_name", "?")
        street_eq[sn][0].append(j.get("raw_equity") or 0)
        street_eq[sn][1].append(j.get("adj_equity") or 0)

    monster_low = sum(
        1
        for j in postflop_logs
        if j["hand"] not in bleed_set
        and j.get("strength") == "monster"
        and (j.get("adj_equity") or 0) < 0.50
    )

    raise_low = sum(
        1
        for j in postflop_logs
        if j["hand"] not in bleed_set
        and j.get("final_action") == "RAISE"
        and (j.get("adj_equity") or 0) < 0.30
    )

    river_tex = [
        j.get("texture_adj")
        for j in postflop_logs
        if j["hand"] not in bleed_set and j.get("street_name") == "river" and j.get("texture_adj") is not None
    ]
    avg_river_tex = sum(river_tex) / len(river_tex) if river_tex else 0

    # Showdown win/loss pot
    sd_win_pots = [pot_from_csv(h) for h in sd_hands if pnls[h] > 0]
    sd_loss_pots = [pot_from_csv(h) for h in sd_hands if pnls[h] < 0]

    sd_loss_monster = [
        h
        for h in sd_hands
        if pnls[h] < 0 and last_post.get(h, {}).get("strength") == "monster"
    ]

    # Semi-bluff (postflop decisions)
    semi_fire = [j for j in postflop_logs if j["hand"] not in bleed_set and j["hand"] >= 1 and j.get("semi_bluff_fired")]
    semi_true = [j for j in semi_fire if j.get("final_action") == "RAISE"]
    semi_hands = sorted({j["hand"] for j in semi_fire})
    semi_pnl = sum(pnls[h] for h in semi_hands)
    semi_wins = sum(1 for h in semi_hands if pnls[h] > 0)

    # Large pots breakdown
    large = [h for h in real_hands_no0 if abs(pnls[h]) >= 50]

    # Equity calibration at showdown: adj_equity from last postflop before SD? Use hand_result time - use last postflop adj_equity
    buckets = {"0-0.3": [0, 0], "0.3-0.5": [0, 0], "0.5-0.7": [0, 0], "0.7-1.0": [0, 0]}
    for h in sd_hands:
        j = last_post.get(h)
        if not j:
            continue
        ae = j.get("adj_equity")
        if ae is None:
            continue
        win = 1 if pnls[h] > 0 else 0
        if ae < 0.3:
            buckets["0-0.3"][0] += win
            buckets["0-0.3"][1] += 1
        elif ae < 0.5:
            buckets["0.3-0.5"][0] += win
            buckets["0.3-0.5"][1] += 1
        elif ae < 0.7:
            buckets["0.5-0.7"][0] += win
            buckets["0.5-0.7"][1] += 1
        else:
            buckets["0.7-1.0"][0] += win
            buckets["0.7-1.0"][1] += 1

    # Opponent fold to raise: when we raise, opp next folds on same street
    # Track sequences in by_hand
    ftr = {"Pre-Flop": [0, 0], "Flop": [0, 0], "Turn": [0, 0], "River": [0, 0]}
    street_order = ["Pre-Flop", "Flop", "Turn", "River"]
    for h, rs in by_hand.items():
        if h in bleed_set:
            continue
        for i, r in enumerate(rs):
            if int(r["active_team"]) != OUR_TEAM or r["action_type"] != "RAISE":
                continue
            st = r["street"]
            if st not in ftr:
                continue
            # find next action by opp on same street
            folded = False
            for r2 in rs[i + 1 :]:
                if r2["street"] != st:
                    break
                if int(r2["active_team"]) == OPP_TEAM and r2["action_type"] == "FOLD":
                    folded = True
                    break
                if int(r2["active_team"]) == OPP_TEAM:
                    break
            ftr[st][1] += 1
            if folded:
                ftr[st][0] += 1

    # Bet sizing: avg raise amount by street (our raises)
    raise_amt = defaultdict(list)
    for h, rs in by_hand.items():
        if h in bleed_set:
            continue
        for r in rs:
            if int(r["active_team"]) != OUR_TEAM or r["action_type"] != "RAISE":
                continue
            st = r["street"]
            if st in street_order:
                amt = int(r["action_amount"])
                if amt > 0:
                    raise_amt[st].append(amt)

    # Comeback mode
    cm_hands = sum(1 for j in hand_start if j["hand"] not in bleed_set and j.get("comeback_mode"))
    cm_pnl = sum(pnls[j["hand"]] for j in hand_start if j["hand"] not in bleed_set and j.get("comeback_mode"))
    cm_pf = [j for j in preflop_logs if j["hand"] not in bleed_set and j.get("comeback_mode")]
    cm_post = [j for j in postflop_logs if j["hand"] not in bleed_set and j.get("comeback_mode")]

    # Top 5 losing hands
    losing = sorted([(h, pnls[h]) for h in real_hands_no0 if pnls[h] < 0], key=lambda x: x[1])[:5]

    # Print everything as structured output
    out = []
    out.append("===A===")
    out.append(f"team_0_final,55\nteam_1_final,-55\nwinner,ALL IN (team 0)\nour_bot_team,1")

    out.append("===B===")
    out.append(f"bleed_start_H,{bleed_label}")

    out.append("===C===")
    out.append(f"real_play_hands_excl_bleed_and_h0,{len(real_hands_no0)}")
    out.append(f"net_pnl,{net_pnl}")
    out.append(f"win_rate,{wins/(len(real_hands_no0) or 1):.6f}")
    out.append(f"showdown_count,{sd_count}")
    out.append(f"showdown_win_rate,{sd_wr:.6f}")
    out.append(f"freq_preflop_FOLD,{pre_counts['FOLD']}")
    out.append(f"freq_preflop_CHECK,{pre_counts['CHECK']}")
    out.append(f"freq_preflop_CALL,{pre_counts['CALL']}")
    out.append(f"freq_preflop_RAISE,{pre_counts['RAISE']}")
    out.append(f"freq_postflop_FOLD,{post_counts['FOLD']}")
    out.append(f"freq_postflop_CHECK,{post_counts['CHECK']}")
    out.append(f"freq_postflop_CALL,{post_counts['CALL']}")
    out.append(f"freq_postflop_RAISE,{post_counts['RAISE']}")
    out.append(f"avg_pot_showdown_hands,{avg_sd_pot:.4f}")
    out.append(f"hands_pnl_lte_-50,{len(big_loss)}")
    out.append(f"hands_pnl_gte_+50,{len(big_win)}")
    out.append(f"pnl_as_SB,{pnl_sb}")
    out.append(f"pnl_as_BB,{pnl_bb}")

    out.append("===D===")
    for st in sorted(str_dist, key=lambda x: -str_dist[x][0]):
        c, s = str_dist[st]
        out.append(f"strength_{st}_count,{c},avg_adj_eq,{s/max(c,1):.6f}")
    out.append(f"preflop_reasons:{dict(pre_reason)}")
    for sn in ["flop", "turn", "river"]:
        rw, ad = street_eq.get(sn, ([], []))
        out.append(f"street_{sn}_avg_raw,{sum(rw)/max(len(rw),1):.6f},avg_adj,{sum(ad)/max(len(ad),1):.6f},n,{len(rw)}")
    out.append(f"monster_adj_lt_0.50_count,{monster_low}")
    out.append(f"raise_adj_lt_0.30_count,{raise_low}")
    out.append(f"avg_texture_adj_river,{avg_river_tex:.6f}")

    out.append("===E===")
    for st in sorted(strength_pnl):
        s, c = strength_pnl[st]
        out.append(f"last_pf_strength,{st},total_pnl,{s},hands,{c},avg,{s/max(c,1):.4f}")

    out.append("===F===")
    out.append(f"sd_avg_pot_win,{sum(sd_win_pots)/max(len(sd_win_pots),1):.4f}")
    out.append(f"sd_avg_pot_loss,{sum(sd_loss_pots)/max(len(sd_loss_pots),1):.4f}")
    out.append(f"sd_losses_last_strength_monster_count,{len(sd_loss_monster)}")
    out.append(f"sd_losses_last_strength_monster_hands,{sd_loss_monster}")

    out.append("===G===")
    out.append(f"semi_bluff_decisions_total,{len(semi_fire)}")
    out.append(f"semi_bluff_decisions_with_RAISE,{len(semi_true)}")
    out.append(f"distinct_hands_with_semi_bluff,{len(semi_hands)}")
    out.append(f"total_pnl_on_those_hands,{semi_pnl}")
    out.append(f"wins_on_those_hands,{semi_wins},win_rate,{semi_wins/max(len(semi_hands),1):.6f}")

    out.append("===H===")
    for h in sorted(large, key=lambda x: -abs(pnls[x])):
        out.append(f"large_hand,{h},pnl,{pnls[h]},showdown,{hr_by_hand.get(h,{}).get('showdown')},outcome,{hr_by_hand.get(h,{}).get('outcome')}")

    out.append("===I===")
    for bk, (w, n) in buckets.items():
        out.append(f"bucket_{bk},win_rate,{w/max(n,1):.6f},n,{n}")

    out.append("===J===")
    for st in ["Pre-Flop", "Flop", "Turn", "River"]:
        folds, opps = ftr[st]
        out.append(f"street_{st},fold_to_raise,{folds},opportunities,{opps},rate,{folds/max(opps,1):.6f}")

    out.append("===K===")
    for st in ["Pre-Flop", "Flop", "Turn", "River"]:
        amts = raise_amt.get(st, [])
        out.append(f"street_{st},avg_raise_amt,{sum(amts)/max(len(amts),1):.4f},n_raises,{len(amts)}")

    out.append("===L===")
    out.append(f"comeback_mode_hand_starts,{cm_hands}")
    out.append(f"comeback_mode_total_pnl,{cm_pnl}")
    out.append(f"comeback_preflop_decisions,{len(cm_pf)}")
    out.append(f"comeback_postflop_decisions,{len(cm_post)}")

    out.append("===M===")
    for h, pnl in losing:
        hr = hr_by_hand.get(h, {})
        lp = last_post.get(h, {})
        out.append(
            f"hand,{h},pnl,{pnl},showdown,{hr.get('showdown')},outcome,{hr.get('outcome')},last_strength,{lp.get('strength')},last_adj_eq,{lp.get('adj_equity')},position,{position_for_hand(h)}"
        )

    print("\n".join(out))


if __name__ == "__main__":
    main()
