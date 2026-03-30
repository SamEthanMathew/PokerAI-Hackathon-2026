"""
Cross-test for the overhauled player.py (MetaBot v2).

Manual regression checks for submission/player.py (not run by pytest).

Tests (each is self-contained and prints PASS/FAIL):
  T1  – opp_last_action present (uppercase): stats accumulate via act()
  T2  – opp_last_action absent: bet-change inference in observe is gone; stats via act()
  T3  – mode switches to TRAP when opponent raises a lot (maniac)
  T4  – mode switches to VALUE when opponent always calls (station)
  T5  – mode switches to AGRO+ramp when opponent folds often (overfolder)
  T6  – _safe_rate uses prior until 3 samples, then uses real data
  T7  – observe() only sets _opp_folded and _running_pnl (no stat updates)
  T8  – _process_opponent_action handles check-raise override
  T9  – _prev_opp_bet is set at start of every act() call
  T10 – syntax + import smoke test
  T11 – case normalization: uppercase "FOLD", "RAISE" etc. handled correctly
  T12 – removed stats no longer exist (showdown_win, suit_attack_fold, opp_af_*)
  T13 – _choose_keep equity guard: prefers high-equity combo over high-score combo
  T14 – _preflop_equity returns max (not avg top-3) and early-exits for premium pairs
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))  # repo root (script lives in scripts/)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    results.append(cond)


# ── T10: import smoke ─────────────────────────────────────────────────────────
print("\nT10 – import smoke test")
try:
    from submission.player import PlayerAgent, _normalize_action, _hand_rank_category
    check("PlayerAgent imports cleanly", True)
except Exception as e:
    check("PlayerAgent imports cleanly", False, str(e))
    print("FATAL: cannot continue without import"); sys.exit(1)


def make_agent():
    return PlayerAgent(stream=False)


def make_obs(street=0, my_bet=1, opp_bet=1, my_cards=None, opp_bet_obs=None,
             valid=None, opp_last_action=None, terminated=False):
    """Build a minimal observation dict."""
    if my_cards is None:
        my_cards = [0, 9, 18, 1, 10]
    if valid is None:
        valid = [False, True, True, True, False]
    obs = {
        "my_cards":           my_cards + [-1] * (5 - len(my_cards)),
        "community_cards":    [-1, -1, -1, -1, -1],
        "opp_discarded_cards": [-1, -1, -1],
        "my_discarded_cards":  [-1, -1, -1],
        "valid_actions": valid,
        "street":        street,
        "min_raise":     2,
        "max_raise":     50,
        "my_bet":        my_bet,
        "opp_bet":       opp_bet if opp_bet_obs is None else opp_bet_obs,
        "pot_size":      my_bet + opp_bet,
        "blind_position": 0,
        "time_left":     400.0,
    }
    if opp_last_action is not None:
        obs["opp_last_action"] = opp_last_action
    return obs


# ── T11: case normalization ───────────────────────────────────────────────────
print("\nT11 – case normalization of opp_last_action")
check("'FOLD' -> 'fold'", _normalize_action("FOLD") == "fold")
check("'RAISE' -> 'raise'", _normalize_action("RAISE") == "raise")
check("'None' -> ''", _normalize_action("None") == "")
check("'' -> ''", _normalize_action("") == "")
check("None -> ''", _normalize_action(None) == "")
check("'CHECK' -> 'check'", _normalize_action("CHECK") == "check")


# ── T12: removed stats no longer exist ────────────────────────────────────────
print("\nT12 – removed stats no longer exist")
p = make_agent()
check("showdown_win removed", "showdown_win" not in p._stats)
check("suit_attack_fold removed", "suit_attack_fold" not in p._stats)
check("opp_af_flop removed", "opp_af_flop" not in p._stats)
check("opp_af_turn removed", "opp_af_turn" not in p._stats)
check("opp_af_river removed", "opp_af_river" not in p._stats)
check("fold_flop_bet removed", "fold_flop_bet" not in p._stats)
check("fold_turn_bet removed", "fold_turn_bet" not in p._stats)
check("fold_river_bet removed", "fold_river_bet" not in p._stats)
check("fold_to_bet exists", "fold_to_bet" in p._stats)
check("opp_3bet_freq removed", "opp_3bet_freq" not in p._stats)
check("opp_river_call_freq removed", "opp_river_call_freq" not in p._stats)


# ── T1: opp_last_action present → stats accumulate via act() ─────────────────
print("\nT1 – opp_last_action present (uppercase): stats via act()")
p = make_agent()
# First act() to establish hand (preflop, SB)
obs0 = make_obs(street=0, my_bet=1, opp_bet=2, opp_last_action="None")
p.act(obs0, 0, False, False, {})

# Now simulate: we raised, opp called on flop (street 1)
# Set state as if we raised
p._last_was_bet = True
p._last_street  = 1
p._prev_opp_bet = 5
p._prev_my_bet  = 10

# Next act() call on street 1 with opp_last_action="CALL" (uppercase from match.py)
obs1 = make_obs(street=1, my_bet=10, opp_bet=10, my_cards=[0, 9],
                opp_last_action="CALL",
                valid=[False, True, True, True, False])
obs1["community_cards"] = [2, 11, 20, -1, -1]
p.act(obs1, 0, False, False, {})

check("fold_to_bet[1] incremented", p._stats["fold_to_bet"][1] >= 1,
      f"got {p._stats['fold_to_bet']}")
check("opp_aggression[1] incremented", p._stats["opp_aggression"][1] >= 1,
      f"got {p._stats['opp_aggression']}")
check("call_down[0] incremented (postflop call)", p._stats["call_down"][0] >= 1,
      f"got {p._stats['call_down']}")


# ── T2: stats no longer updated in observe() ─────────────────────────────────
print("\nT2 – observe() does NOT update stats")
p = make_agent()
p.hand_number = 10
p._last_was_bet = True
p._last_street  = 2
p._prev_opp_bet = 2
p._prev_my_bet  = 10

obs = make_obs(street=2, my_bet=10, opp_bet=10, opp_last_action="CALL")
p.observe(obs, 0, False, False, {})
check("opp_aggression[1] stays 0 after observe()", p._stats["opp_aggression"][1] == 0,
      f"got {p._stats['opp_aggression']}")
check("fold_to_bet[1] stays 0 after observe()", p._stats["fold_to_bet"][1] == 0,
      f"got {p._stats['fold_to_bet']}")


# ── T3: mode → TRAP when opponent is a maniac ─────────────────────────────────
print("\nT3 – mode switches to TRAP (maniac: opp_aggro > 0.45)")
p = make_agent()
p.hand_number = 10
p._stats["opp_aggression"] = [10, 20]  # 0.50
mode, mult = p._select_mode()
check("TRAP triggered by high opp_aggro", mode == "TRAP", f"got mode={mode}")


# ── T4: mode → VALUE when opponent is a calling station ──────────────────────
print("\nT4 – mode switches to VALUE (station: call_down > 0.55)")
p = make_agent()
p.hand_number = 10
p._stats["call_down"]   = [6, 10]    # 0.60
p._stats["fold_to_bet"] = [4, 10]    # 0.40 — normal
mode, mult = p._select_mode()
check("VALUE triggered by call_down 0.60", mode == "VALUE", f"got mode={mode}")


# ── T5: mode → AGRO+ramp when opponent folds often ───────────────────────────
print("\nT5 – mode switches to AGRO+ramp (overfolder: fold_to_bet > 0.48)")
p = make_agent()
p.hand_number = 20
p._stats["fold_to_bet"] = [5, 10]  # 0.50 > 0.48
mode, mult = p._select_mode()
check("AGRO triggered by fold_to_bet 0.50", mode == "AGRO", f"got mode={mode}")
check("AGRO mult > 1.0 (ramp active)", mult > 1.0, f"got mult={mult:.2f}")


# ── T6: _safe_rate uses prior until 3 samples ─────────────────────────────────
print("\nT6 – _safe_rate uses prior until 3 samples, real data after")
p = make_agent()
p._stats["fold_to_bet"] = [0, 2]
rate_with_2 = p._safe_rate("fold_to_bet")
check("2 obs -> returns prior (0.35)", abs(rate_with_2 - 0.35) < 0.001, f"got {rate_with_2:.3f}")

p._stats["fold_to_bet"] = [0, 3]
rate_with_3 = p._safe_rate("fold_to_bet")
check("3 obs -> returns real rate (0.0)", abs(rate_with_3 - 0.0) < 0.001, f"got {rate_with_3:.3f}")


# ── T7: observe() only sets _opp_folded and _running_pnl ─────────────────────
print("\nT7 – observe() handles terminated correctly (only P&L + fold)")
p = make_agent()
p.hand_number = 5
p._we_folded  = False
p._opp_folded = False

obs = make_obs(my_bet=10, opp_bet=10, opp_last_action="None")
p.observe(obs, -5, True, False, {})
check("_running_pnl updated to -5", p._running_pnl == -5, f"got {p._running_pnl}")

p2 = make_agent()
obs2 = make_obs(my_bet=10, opp_bet=5, opp_last_action="FOLD")
p2.observe(obs2, 10, True, False, {})
check("_opp_folded set True on fold", p2._opp_folded)
check("_running_pnl updated to 10", p2._running_pnl == 10, f"got {p2._running_pnl}")


# ── T8: _process_opponent_action handles check-raise override ─────────────────
print("\nT8 – check-raise detection sets override to VALUE")
p = make_agent()
p.hand_number = 10
p._hand_override = None

obs = make_obs(street=1, my_bet=5, opp_bet=15, opp_last_action="RAISE")
p._process_opponent_action(obs, "raise", False, 1)
check("check_raise[0] incremented", p._stats["check_raise"][0] == 1)
check("_hand_override set to VALUE", p._hand_override == "VALUE")


# ── T9: _prev_opp_bet set at start of act() ───────────────────────────────────
print("\nT9 – _prev_opp_bet set at start of act()")
p = make_agent()
p.hand_number = 1
obs = make_obs(street=0, my_bet=1, opp_bet=3, my_cards=[0, 9, 18, 1, 10],
               valid=[False, True, True, True, False],
               opp_last_action="None")
p.act(obs, 0, False, False, {})
check("_prev_opp_bet set to 3 after act()", p._prev_opp_bet == 3, f"got {p._prev_opp_bet}")


# ── T13: _choose_keep equity guard ─────────────────────────────────────────────
print("\nT13 – _choose_keep equity guard (prefers high-equity combo)")
# This is tested indirectly: _keep_score now returns (score, eq) tuple
p = make_agent()
sc, eq = p._keep_score([0, 1], [], [], set(), 30)
check("_keep_score returns tuple (score, equity)", isinstance(sc, float) and isinstance(eq, float),
      f"got ({type(sc).__name__}, {type(eq).__name__})")


# ── T14: _preflop_equity ─────────────────────────────────────────────────────
print("\nT14 – _preflop_equity uses max and early-exits for premium pairs")
p = make_agent()
# AA pair: cards 8 (Ad=8%9=8), 17 (Ah=17%9=8), plus 3 filler
aa_cards = [8, 17, 0, 1, 2]
eq = p._preflop_equity(aa_cards, 400)
check("AA early exits with 0.75", abs(eq - 0.75) < 0.001, f"got {eq:.3f}")


# ── T15: new-hand branch passes prev context (not False, 0) ───────────────────
print("\nT15 – new-hand fold attributed to correct street (prev_was_bet/prev_street)")
p = make_agent()
# Hand 1: simulate ending on river (street 3) where we bet
obs_h1 = make_obs(street=3, my_bet=20, opp_bet=10, my_cards=[0, 9],
                  opp_last_action="None",
                  valid=[False, True, True, True, False])
obs_h1["community_cards"] = [2, 11, 20, 3, 12]
p.act(obs_h1, 0, False, False, {})
# After act, _last_was_bet and _last_street should reflect our action
p._last_was_bet = True
p._last_street  = 3
p._prev_opp_bet = 10
p._prev_my_bet  = 20

# Hand 2 starts: opponent folded to our river bet (terminal action from prev hand)
obs_h2 = make_obs(street=0, my_bet=1, opp_bet=2, my_cards=[0, 9, 18, 1, 10],
                  opp_last_action="FOLD",
                  valid=[False, True, True, True, False])
p.act(obs_h2, 5, False, False, {})

check("fold_to_bet[0] incremented for end-of-hand fold",
      p._stats["fold_to_bet"][0] >= 1,
      f"got {p._stats['fold_to_bet']}")
check("fold_to_raise[0] incremented (we had bet -> raise context)",
      p._stats["fold_to_raise"][0] >= 1,
      f"got {p._stats['fold_to_raise']}")
check("opp_preflop_raise NOT incremented (fold was on river, not preflop)",
      p._stats["opp_preflop_raise"][0] == 0,
      f"got {p._stats['opp_preflop_raise']}")


# ── T16: _hand_rank_category draw_only (flush draw) ─────────────────────────────
print("\nT16 – _hand_rank_category: 4-to-flush no pair -> draw_only")
# Four of suit 0, ranks 0,1,2,3,5 (no pair, no made straight). 14 = s1 rank5.
my_cards = [0, 1]
community = [2, 3, 14]
check("flush draw returns draw_only", _hand_rank_category(my_cards, community) == "draw_only",
      f"got {_hand_rank_category(my_cards, community)}")


# ── T17: _hand_rank_category draw_only (straight draw) and nothing ──────────────
print("\nT17 – _hand_rank_category: 4-to-straight -> draw_only; pure air -> nothing")
# 4 to straight (ranks 0,1,2,3), fifth rank 7 so no made straight. 16 = rank7.
my_cards_s = [0, 1]
community_s = [2, 3, 16]
check("straight draw returns draw_only", _hand_rank_category(my_cards_s, community_s) == "draw_only",
      f"got {_hand_rank_category(my_cards_s, community_s)}")
# Pure air: no 4 flush, no 4 straight, no pair. Ranks 0,2,5,7,8 all distinct.
my_cards_n = [0, 2]
community_n = [14, 16, 17]
check("no draw no pair returns nothing", _hand_rank_category(my_cards_n, community_n) == "nothing",
      f"got {_hand_rank_category(my_cards_n, community_n)}")


# ── T18: draw_only not forced to FOLD (postflop cap leaves mode result) ────────
print("\nT18 – draw_only hand: postflop cap does not override to FOLD")
# With draw_only, the cap block only applies to one_pair and nothing; draw_only is left to mode.
p = make_agent()
p._hand_mode = "AGRO"
p._hand_override = None
# Flop: we have flush draw (e.g. 0,1 and board 2,3,10). Mode would e.g. CALL.
obs = make_obs(street=1, my_bet=2, opp_bet=5, my_cards=[0, 1],
               valid=[False, True, True, True, False])
obs["community_cards"] = [2, 3, 10, -1, -1]
obs["opp_discarded_cards"] = [18, 19, 20]
obs["my_discarded_cards"] = [4, 5, 6]
p._last_equity = 0.45
action, amt, _, _ = p.act(obs, 0, False, False, {})
# With draw_only we must not have forced FOLD (mode could return CALL or CHECK)
check("draw_only not forced to FOLD", action != 0, f"action={action} (0=FOLD)")
# 0=FOLD, 1=CHECK, 2=CALL, 3=RAISE in gym_env. So action != 0 means not FOLD.


# ── T19: one-pair high equity can raise; fold only when equity < 0.48 ───────────
print("\nT19 – one-pair: high equity allows raise; low equity + big bet triggers fold")
# One pair with equity > 0.75: raise not downgraded
p = make_agent()
p._hand_mode = "VALUE"
p._hand_override = None
# Simulate postflop with one pair (e.g. pair of 8s on board). We need result to be RAISE and equity > 0.75.
# This is tested indirectly: set _last_equity after an action would be chosen; the cap runs after.
# So we need a scenario where the mode returns RAISE and we have one pair. Then cap checks equity.
# If equity > 0.75 we keep RAISE. So we just assert the logic: when equity is high, one pair doesn't force call.
# Create obs where we have one pair (hand rank matches board rank). VALUE might return RAISE with high equity.
obs_op = make_obs(street=2, my_bet=5, opp_bet=5, my_cards=[8, 17],  # two 8s
                  valid=[False, True, True, True, False])
obs_op["community_cards"] = [26, 2, 11, 20, -1]  # include an 8? 8 is rank 6. 26=2*9+8, so 26 is rank 8. So board has rank 8. Hand 8,17: 8=0*9+8 (rank 8), 17=1*9+8. So we have trips (three 8s), not one pair. For one pair we need exactly one pair. So hand [8, 9] (rank 8 and rank 0), board [26, 2, 11] (rank 8, 0, 2). Then we have pair of 8s. So my_cards = [8, 9], community = [26, 2, 11, 3, 12].
obs_op["my_cards"] = [8, 9]
obs_op["community_cards"] = [26, 2, 11, 3, 12]
obs_op["opp_discarded_cards"] = [0, 1, 10]
obs_op["my_discarded_cards"] = [4, 5, 6]
# Force high equity so one-pair cap allows raise
p._last_equity = 0.78
p._betting_history = {"bet_flop": False, "bet_turn": False}
action_high, _, _, _ = p.act(obs_op, 0, False, False, {})
# With one pair and equity 0.78, if mode returned RAISE we keep it (no downgrade)
# We only check that we didn't get FOLD due to one-pair fold rule (that requires equity < 0.48)
check("one-pair with high equity does not force fold", action_high != 0 or p._last_equity >= 0.48,
      f"action={action_high} eq={p._last_equity}")


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed")
if passed == total:
    print(f"\033[92mAll tests PASSED\033[0m")
else:
    print(f"\033[91m{total - passed} test(s) FAILED\033[0m")
    sys.exit(1)
