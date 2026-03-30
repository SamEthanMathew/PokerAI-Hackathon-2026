"""
Compare our OPP_RECON (read) vs actual opponent behavior from CSV.
Actual VPIP/PFR/AF/fold rates: from CSV, opponent = the other team; trailing window.
"""


def _actual_vpip_pfr_from_hands(hands, our_team_id, window_hands):
    """Compute actual VPIP and PFR for opponent over the given hand numbers."""
    opp = 1 - our_team_id
    vpip_count = pfr_count = opportunities = 0
    hand_set = set(window_hands)
    for h in hands:
        if h["hand_number"] not in hand_set:
            continue
        opportunities += 1
        vpip = pfr = False
        for a in h["actions"]:
            if int(a.get("active_team", -1)) != opp:
                continue
            street = (a.get("street") or "").lower()
            if "pre-flop" not in street and "preflop" not in street:
                continue
            act = (a.get("action_type") or "").upper()
            if act == "RAISE":
                vpip = True
                pfr = True
            elif act == "CALL":
                vpip = True
        if vpip:
            vpip_count += 1
        if pfr:
            pfr_count += 1
    n = max(1, opportunities)
    return vpip_count / n, pfr_count / n


def _is_preflop(street):
    s = (street or "").lower()
    return "pre-flop" in s or "preflop" in s


def _is_river(street):
    s = (street or "").lower()
    return "river" in s


def _actual_af_fold_rates_from_hands(hands, our_team_id, window_hands):
    """
    From CSV actions: opponent raise/call on Flop/Turn/River -> AF.
    Opponent faced our bet (we RAISE, then they act) and FOLD -> fold_non_river / fold_river.
    Opponent had opportunity to bet (we CHECK, then they act) and they RAISE -> non_river_bet_pct.
    """
    opp = 1 - our_team_id
    hand_set = set(window_hands)
    raise_n = call_n = 0
    non_river_faced = non_river_folds = 0
    river_faced = river_folds = 0
    non_river_bet_opps = non_river_bet_n = 0

    for h in hands:
        if h["hand_number"] not in hand_set:
            continue
        actions = h["actions"]
        for i, a in enumerate(actions):
            act = (a.get("action_type") or "").upper()
            street = a.get("street") or ""
            active = int(a.get("active_team", -1))

            if active == opp:
                if not _is_preflop(street):
                    if act == "RAISE":
                        raise_n += 1
                    elif act == "CALL":
                        call_n += 1

                # Did they face our bet? (previous action in same street was us with RAISE)
                if i > 0:
                    prev = actions[i - 1]
                    if prev.get("street") == street and int(prev.get("active_team", -1)) == our_team_id:
                        prev_act = (prev.get("action_type") or "").upper()
                        if prev_act == "RAISE":
                            if _is_river(street):
                                river_faced += 1
                                if act == "FOLD":
                                    river_folds += 1
                            else:
                                non_river_faced += 1
                                if act == "FOLD":
                                    non_river_folds += 1
                        elif prev_act == "CHECK":
                            if not _is_river(street):
                                non_river_bet_opps += 1
                                if act == "RAISE":
                                    non_river_bet_n += 1

    af = raise_n / max(1, call_n)
    fold_non_river = non_river_folds / max(1, non_river_faced) if non_river_faced else 0.5
    fold_river = river_folds / max(1, river_faced) if river_faced else 0.5
    non_river_bet_pct = non_river_bet_n / max(1, non_river_bet_opps) if non_river_bet_opps else 0.5
    return af, fold_non_river, fold_river, non_river_bet_pct


def compute_read_accuracy(
    opp_recon,
    csv_hands,
    our_team_id,
    window=50,
):
    """
    For each hand where we have OPP_RECON, compare our read to actual (trailing window from CSV).
    Returns list of {
        hand_number, our_vpip, our_pfr, actual_vpip, actual_pfr, our_opp_type,
        our_af, actual_af, our_fold_non_river, actual_fold_non_river,
        our_fold_river, actual_fold_river, our_non_river_bet_pct, actual_non_river_bet_pct
    }.
    """
    if not opp_recon or not csv_hands:
        return []

    by_hand = {r["hand"]: r for r in opp_recon if r.get("hand") is not None}
    hand_nums = sorted(by_hand.keys())
    csv_hand_nums = {h["hand_number"] for h in csv_hands}
    out = []
    for h in hand_nums:
        start = max(0, h - window + 1)
        window_hands = list(range(start, h + 1))
        if not csv_hand_nums.intersection(window_hands):
            continue
        actual_vpip, actual_pfr = _actual_vpip_pfr_from_hands(csv_hands, our_team_id, window_hands)
        actual_af, actual_fold_nr, actual_fold_r, actual_nr_bet = _actual_af_fold_rates_from_hands(
            csv_hands, our_team_id, window_hands
        )
        r = by_hand[h]

        def safe_float(x, default=0.0):
            if x is None:
                return default
            try:
                return float(x)
            except (TypeError, ValueError):
                return default

        our_vpip = safe_float(r.get("vpip"), 0.0)
        our_pfr = safe_float(r.get("pfr"), 0.0)
        our_af = safe_float(r.get("af"), 1.0)
        our_fold_nr = safe_float(r.get("fold_non_river"), 0.5)
        our_fold_r = safe_float(r.get("fold_river"), 0.5)
        our_nr_bet = safe_float(r.get("non_river_bet_pct"), 0.5)

        out.append({
            "hand_number": h,
            "our_vpip": round(our_vpip, 3),
            "our_pfr": round(our_pfr, 3),
            "actual_vpip": round(actual_vpip, 3),
            "actual_pfr": round(actual_pfr, 3),
            "our_opp_type": r.get("opp_type") or "",
            "our_af": round(our_af, 3),
            "actual_af": round(actual_af, 3),
            "our_fold_non_river": round(our_fold_nr, 3),
            "actual_fold_non_river": round(actual_fold_nr, 3),
            "our_fold_river": round(our_fold_r, 3),
            "actual_fold_river": round(actual_fold_r, 3),
            "our_non_river_bet_pct": round(our_nr_bet, 3),
            "actual_non_river_bet_pct": round(actual_nr_bet, 3),
        })
    return out
