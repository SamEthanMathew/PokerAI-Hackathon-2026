"""
Extract mistake highlights from HAND_RESULT data.
"""


def extract_mistakes(
    hand_results,
    big_pot_threshold=50,
):
    """
    From HAND_RESULT list, extract:
      - invalid_action_hands: list of hand numbers where we had invalid_action=True
      - we_fold_big_pot: list of {hand_number, pot, ...} where end_type=we_fold and pot >= threshold
      - loss_breakdown: counts by street_ended, end_type, position
    """
    invalid_hands = []
    we_fold_big = []
    loss_breakdown = {}  # (street_ended, end_type, position) -> {count, total_reward}

    for r in hand_results:
        hand = r.get("hand")
        if hand is None:
            continue
        if r.get("invalid_action"):
            invalid_hands.append(hand)
        end_type = (r.get("end_type") or "").strip()
        pot = r.get("pot") or 0
        try:
            pot = int(pot)
        except (TypeError, ValueError):
            pot = 0
        if end_type == "we_fold" and pot >= big_pot_threshold:
            we_fold_big.append({"hand_number": hand, "pot": pot, **{k: r.get(k) for k in ("reward", "position", "our_discard_class")}})

        street_ended = r.get("street_ended")
        position = r.get("position") or ""
        key = (street_ended, end_type, position)
        if key not in loss_breakdown:
            loss_breakdown[key] = {"count": 0, "total_reward": 0.0}
        reward = r.get("reward") or 0
        try:
            reward = float(reward)
        except (TypeError, ValueError):
            reward = 0.0
        loss_breakdown[key]["count"] += 1
        loss_breakdown[key]["total_reward"] += reward

    return {
        "invalid_action_hands": invalid_hands,
        "we_fold_big_pot": we_fold_big,
        "loss_breakdown": loss_breakdown,
    }
