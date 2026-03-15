"""
Betting policy table construction.
Builds a mixed-strategy policy table keyed by (street, position, strength, board, to_call).
Output is a Python dict suitable for embedding in libratus_tables.py.
"""

STREETS = [0, 1, 2, 3]
POSITIONS = ["sb", "bb"]
STRENGTH_BUCKETS = ["monster", "strong", "good", "marginal", "weak"]
BOARD_BUCKETS = ["wet", "medium", "dry"]
TO_CALL_BUCKETS = ["none", "small", "medium", "large"]

ACTIONS = ["fold", "check_call", "small_bet", "medium_bet", "large_bet", "jam"]


def _normalize(d):
    t = sum(d.values())
    if t == 0:
        return {k: 1.0 / len(d) for k in d}
    return {k: round(v / t, 3) for k, v in d.items()}


def build_preflop_policy():
    """Preflop (street 0) policy: loose-aggressive."""
    table = {}
    for pos in POSITIONS:
        for tc in TO_CALL_BUCKETS:
            facing_raise = tc in ("small", "medium", "large")

            # Monster / strong preflop
            for strength in ["monster", "strong"]:
                if facing_raise:
                    p = {"fold": 0, "check_call": 0.10, "small_bet": 0.20,
                         "medium_bet": 0.35, "large_bet": 0.25, "jam": 0.10}
                else:
                    p = {"fold": 0, "check_call": 0.05, "small_bet": 0.35,
                         "medium_bet": 0.35, "large_bet": 0.15, "jam": 0.10}
                table[(0, pos, strength, "any", tc)] = _normalize(p)

            # Good preflop
            if facing_raise:
                p = {"fold": 0.10, "check_call": 0.50, "small_bet": 0.20,
                     "medium_bet": 0.15, "large_bet": 0.05, "jam": 0}
            else:
                p = {"fold": 0, "check_call": 0.20, "small_bet": 0.45,
                     "medium_bet": 0.25, "large_bet": 0.10, "jam": 0}
            table[(0, pos, "good", "any", tc)] = _normalize(p)

            # Marginal preflop
            if facing_raise:
                p = {"fold": 0.30, "check_call": 0.50, "small_bet": 0.15,
                     "medium_bet": 0.05, "large_bet": 0, "jam": 0}
            else:
                p = {"fold": 0.05, "check_call": 0.55, "small_bet": 0.30,
                     "medium_bet": 0.10, "large_bet": 0, "jam": 0}
            table[(0, pos, "marginal", "any", tc)] = _normalize(p)

            # Weak preflop: occasional bluff
            if facing_raise and tc == "large":
                p = {"fold": 0.80, "check_call": 0.15, "small_bet": 0.05,
                     "medium_bet": 0, "large_bet": 0, "jam": 0}
            elif facing_raise:
                p = {"fold": 0.60, "check_call": 0.25, "small_bet": 0.10,
                     "medium_bet": 0.05, "large_bet": 0, "jam": 0}
            else:
                p = {"fold": 0.15, "check_call": 0.50, "small_bet": 0.25,
                     "medium_bet": 0.10, "large_bet": 0, "jam": 0}
            table[(0, pos, "weak", "any", tc)] = _normalize(p)

    return table


def build_postflop_policy():
    """Post-flop (streets 1-3) policy: value-heavy, semi-bluff capable."""
    table = {}
    for street in [1, 2, 3]:
        for pos in POSITIONS:
            for board in BOARD_BUCKETS:
                for tc in TO_CALL_BUCKETS:
                    facing_bet = tc in ("small", "medium", "large")
                    is_river = street == 3

                    # Monster
                    if facing_bet:
                        p = {"fold": 0, "check_call": 0.15, "small_bet": 0.10,
                             "medium_bet": 0.30, "large_bet": 0.30, "jam": 0.15}
                    else:
                        p = {"fold": 0, "check_call": 0.10, "small_bet": 0.10,
                             "medium_bet": 0.30, "large_bet": 0.35, "jam": 0.15}
                    table[(street, pos, "monster", board, tc)] = _normalize(p)

                    # Strong
                    if facing_bet:
                        p = {"fold": 0.03, "check_call": 0.25, "small_bet": 0.20,
                             "medium_bet": 0.30, "large_bet": 0.17, "jam": 0.05}
                    else:
                        p = {"fold": 0, "check_call": 0.15, "small_bet": 0.25,
                             "medium_bet": 0.35, "large_bet": 0.20, "jam": 0.05}
                    table[(street, pos, "strong", board, tc)] = _normalize(p)

                    # Good
                    wet_adj = 0.05 if board == "wet" else 0
                    if facing_bet:
                        p = {"fold": 0.10 + wet_adj, "check_call": 0.40,
                             "small_bet": 0.25, "medium_bet": 0.15,
                             "large_bet": 0.05, "jam": 0}
                    else:
                        p = {"fold": 0, "check_call": 0.35, "small_bet": 0.35,
                             "medium_bet": 0.20, "large_bet": 0.10, "jam": 0}
                    if is_river and facing_bet:
                        p["fold"] += 0.10
                        p["check_call"] -= 0.05
                        p["small_bet"] -= 0.05
                    table[(street, pos, "good", board, tc)] = _normalize(p)

                    # Marginal
                    if facing_bet:
                        p = {"fold": 0.30, "check_call": 0.45, "small_bet": 0.15,
                             "medium_bet": 0.10, "large_bet": 0, "jam": 0}
                    else:
                        p = {"fold": 0.05, "check_call": 0.50, "small_bet": 0.30,
                             "medium_bet": 0.15, "large_bet": 0, "jam": 0}
                    if is_river and facing_bet:
                        p["fold"] += 0.15
                        p["check_call"] -= 0.10
                        p["small_bet"] -= 0.05
                    table[(street, pos, "marginal", board, tc)] = _normalize(p)

                    # Weak: bluff mix on dry boards, fold more on wet
                    bluff_freq = 0.20 if board == "dry" else 0.10
                    if facing_bet:
                        p = {"fold": 0.55, "check_call": 0.20,
                             "small_bet": bluff_freq, "medium_bet": 0.05,
                             "large_bet": 0, "jam": 0}
                    else:
                        p = {"fold": 0.10, "check_call": 0.40,
                             "small_bet": bluff_freq + 0.10, "medium_bet": 0.10,
                             "large_bet": 0.05, "jam": 0}
                    if is_river:
                        p["fold"] += 0.15
                        p["small_bet"] = max(0, p["small_bet"] - 0.10)
                    table[(street, pos, "weak", board, tc)] = _normalize(p)

    return table


def build_full_policy():
    """Build the complete policy table."""
    policy = {}
    policy.update(build_preflop_policy())
    policy.update(build_postflop_policy())
    return policy


def policy_to_source(policy):
    """Convert policy dict to Python source code for embedding."""
    lines = ["POLICY = {"]
    for key in sorted(policy.keys()):
        lines.append(f"    {key!r}: {policy[key]!r},")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    policy = build_full_policy()
    print(f"Generated {len(policy)} policy entries.")
    print("\nSample entries:")
    for key in list(sorted(policy.keys()))[:10]:
        print(f"  {key} -> {policy[key]}")
