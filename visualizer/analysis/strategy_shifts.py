# Strategy shift detection from OPP_RECON series

def detect_opponent_shifts(opp_recon, window=50, vpip_threshold=0.15, pfr_threshold=0.15, af_threshold=0.5):
    if not opp_recon or len(opp_recon) < window * 2:
        return []
    shifts = []
    by_hand = sorted(opp_recon, key=lambda x: x.get("hand", 0))
    for i in range(window, len(by_hand) - window):
        prev_window = by_hand[i - window : i]
        next_window = by_hand[i : i + window]
        def avg(lst, key):
            vals = [x.get(key) for x in lst if x.get(key) is not None]
            if not vals:
                return 0.0
            try:
                return sum(float(v) for v in vals) / len(vals)
            except (TypeError, ValueError):
                return 0.0
        vpip_before = avg(prev_window, "vpip")
        vpip_after = avg(next_window, "vpip")
        if abs(vpip_after - vpip_before) >= vpip_threshold:
            shifts.append({"hand_number": by_hand[i].get("hand"), "reason": "vpip_change", "metric": "vpip", "before_value": round(vpip_before, 3), "after_value": round(vpip_after, 3)})
        pfr_before = avg(prev_window, "pfr")
        pfr_after = avg(next_window, "pfr")
        if abs(pfr_after - pfr_before) >= pfr_threshold:
            shifts.append({"hand_number": by_hand[i].get("hand"), "reason": "pfr_change", "metric": "pfr", "before_value": round(pfr_before, 3), "after_value": round(pfr_after, 3)})
        af_before = avg(prev_window, "af")
        af_after = avg(next_window, "af")
        if abs(af_after - af_before) >= af_threshold:
            shifts.append({"hand_number": by_hand[i].get("hand"), "reason": "af_change", "metric": "af", "before_value": round(af_before, 3), "after_value": round(af_after, 3)})
        type_before = prev_window[-1].get("opp_type") if prev_window else ""
        type_after = next_window[0].get("opp_type") if next_window else ""
        if type_before and type_after and type_before != type_after:
            shifts.append({"hand_number": by_hand[i].get("hand"), "reason": "opp_type_change", "metric": "opp_type", "before_value": type_before, "after_value": type_after})
    return shifts
