"""
Compare our OPP_RECON (read) vs actual opponent behavior from CSV.
Actual VPIP/PFR: from CSV, opponent = the other team; trailing window.
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


def compute_read_accuracy(
    opp_recon,
    csv_hands,
    our_team_id,
    window=50,
):
    """
    For each hand where we have OPP_RECON, compare our vpip/pfr to actual (trailing window from CSV).
    Returns list of {hand_number, our_vpip, our_pfr, actual_vpip, actual_pfr, our_opp_type}.
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
        r = by_hand[h]
        our_vpip = r.get("vpip")
        our_pfr = r.get("pfr")
        if our_vpip is None:
            our_vpip = 0.0
        if our_pfr is None:
            our_pfr = 0.0
        try:
            our_vpip = float(our_vpip)
            our_pfr = float(our_pfr)
        except (TypeError, ValueError):
            our_vpip = our_pfr = 0.0
        out.append({
            "hand_number": h,
            "our_vpip": round(our_vpip, 3),
            "our_pfr": round(our_pfr, 3),
            "actual_vpip": round(actual_vpip, 3),
            "actual_pfr": round(actual_pfr, 3),
            "our_opp_type": r.get("opp_type") or "",
        })
    return out
