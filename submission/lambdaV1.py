# LambdaV1: Meta chassis × Alpha killer instinct
#
# METAV5's adaptive opponent profiling + 4-mode decision tree,
# enhanced with Alpha's strongest proven edges:
#
#   1. Early-phase aggression (hands 1-EARLY_PHASE_HANDS: AGRO-lite default)
#   2. Bleed-out lock (fold out rest of match if chip lead > max remaining blind cost)
#   3. Semi-bluff band in AGRO mode (equity 0.38-0.52 → small raise at %)
#   4. Softened one-pair cap (allow strong top-pair raises vs overfolders on dry boards)
#   5. Alpha-style lightweight discard threat delta on equity
#   6. External profile JSON (logs/lambda_profile.json) for rapid tuning
#
# Mode routing (per hand, priority order):
#   hand <= EARLY_PHASE_HANDS       -> AGRO-lite (unless extreme evidence)
#   total_obs < 5                   -> GTO_DEFAULT
#   opp_aggression > 0.45           -> TRAP_MODE    (Maniac)
#   fold_to_bet > 0.48              -> AGRO + mult  (Overfolder)
#   fold_to_bet < 0.30 or
#     call_down > 0.55              -> VALUE_GRIND  (Station)
#   check_raise > 0.10              -> TRAP_MODE    (Trapper)
#   opp_aggression < 0.15           -> AGRO + mult  (Passive)
#   opp_preflop_raise > 0.50        -> AGRO         (Loose PF)
#   opp_preflop_raise < 0.12        -> VALUE_GRIND  (Limper)
#   default                         -> AGRO x 1.0

import json
import os
import random
from collections import Counter
from itertools import combinations

from agents.agent import Agent
from gym_env import PokerEnv

try:
    from submission.libratus_tables import POLICY, KEEP_EQUITY, POSTERIOR, MATCHUPS
except ImportError:
    from libratus_tables import POLICY, KEEP_EQUITY, POSTERIOR, MATCHUPS

try:
    from submission.discard_engine import choose_keep_postflop as _choose_keep_postflop
except ImportError:
    from discard_engine import choose_keep_postflop as _choose_keep_postflop

# ── Profile loading ────────────────────────────────────────────────────────────

_PROFILE = {}
_profile_path = os.path.join(os.path.dirname(__file__), "..", "logs", "lambda_profile.json")
if os.path.isfile(_profile_path):
    try:
        with open(_profile_path, "r", encoding="utf-8") as f:
            _PROFILE = json.load(f)
    except (json.JSONDecodeError, OSError):
        pass

# ── Tunable constants (override via lambda_profile.json) ──────────────────────

TOTAL_HANDS          = _PROFILE.get("total_hands",              1000)
EARLY_PHASE_HANDS    = _PROFILE.get("early_phase_hands",          50)
EARLY_AGRO_MULT      = _PROFILE.get("early_agro_mult",           1.2)
EARLY_EXTREME_AGGRO  = _PROFILE.get("early_extreme_aggro",      0.55)  # force TRAP
EARLY_EXTREME_FOLD   = _PROFILE.get("early_extreme_fold",        0.20)  # force VALUE

SEMI_BLUFF_LO        = _PROFILE.get("semi_bluff_lo",            0.38)
SEMI_BLUFF_HI        = _PROFILE.get("semi_bluff_hi",            0.52)
SEMI_BLUFF_FREQ      = _PROFILE.get("semi_bluff_freq",          0.25)   # was 0.50 — reduced to avoid over-bluffing
SEMI_BLUFF_FOLD_MIN  = _PROFILE.get("semi_bluff_fold_min",       0.38)  # kept for profile compat but no longer a hard gate

ONE_PAIR_RAISE_EQ    = _PROFILE.get("one_pair_raise_equity",      0.62)
ONE_PAIR_RAISE_FOLD  = _PROFILE.get("one_pair_raise_fold_thresh",  0.42)
ONE_PAIR_RAISE_AGGRO = _PROFILE.get("one_pair_raise_aggro_cap",    0.35)  # max opp_aggression to allow one-pair raise
ONE_PAIR_FOLD_PRESS  = _PROFILE.get("one_pair_fold_pressure",      0.50)

RIVER_CALL_POT_RATIO_MAX = _PROFILE.get("river_call_pot_ratio_max", 0.35)  # Alpha feature: cap river calls

# Keep-score weights (AGRO mode) — tunable without touching logic
_KW_EQ   = _PROFILE.get("keep_eq_w",   3.0)
_KW_PP   = _PROFILE.get("keep_pp_w",   2.0)
_KW_DD   = _PROFILE.get("keep_dd_w",   1.7)
_KW_MHV  = _PROFILE.get("keep_mhv_w",  1.2)
_KW_BV   = _PROFILE.get("keep_bv_w",   0.8)
_KW_SF   = _PROFILE.get("keep_sf_w",   0.6)

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_RANKS = 9
DECK_SIZE = 27
RANK_A    = 8
RANK_9    = 7
RANK_8    = 6

FOLD    = PokerEnv.ActionType.FOLD.value
RAISE   = PokerEnv.ActionType.RAISE.value
CHECK   = PokerEnv.ActionType.CHECK.value
CALL    = PokerEnv.ActionType.CALL.value
DISCARD = PokerEnv.ActionType.DISCARD.value

_int_to_card = PokerEnv.int_to_card

MODE_GTO   = "GTO"
MODE_AGRO  = "AGRO"
MODE_VALUE = "VALUE"
MODE_TRAP  = "TRAP"

# ── Premium hand definitions ───────────────────────────────────────────────────

PREM_PAIRS = frozenset([
    frozenset([RANK_A, RANK_A]),
    frozenset([RANK_9, RANK_9]),
    frozenset([RANK_8, RANK_8]),
])

PREM_ANY = frozenset([
    frozenset([RANK_A, RANK_9]),
    frozenset([RANK_A, RANK_8]),
])

PREM_SUITED = frozenset([
    frozenset([RANK_9, RANK_8]),
    frozenset([RANK_8, 5]),
    frozenset([5, 4]),
    frozenset([5, RANK_9]),
])

# ── Card helpers ───────────────────────────────────────────────────────────────

def _rank(c):
    return c % NUM_RANKS

def _suit(c):
    return c // NUM_RANKS

def _effective_gap(c1, c2):
    g = abs(_rank(c1) - _rank(c2))
    return g if g <= 4 else NUM_RANKS - g

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _max_connectivity(ranks):
    unique = sorted(set(ranks))
    if not unique:
        return 0
    best = cur = 1
    for i in range(1, len(unique)):
        if unique[i] - unique[i - 1] == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    if RANK_A in unique and 0 in unique:
        best = max(best, 2)
    return best

def _is_premium(c1, c2):
    r1, r2 = _rank(c1), _rank(c2)
    s1, s2 = _suit(c1), _suit(c2)
    ranks = frozenset([r1, r2])
    if r1 == r2 and frozenset([r1, r1]) in PREM_PAIRS:
        return True
    if ranks in PREM_ANY:
        return True
    if s1 == s2 and ranks in PREM_SUITED:
        return True
    return False

def _has_any_premium(cards):
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _is_premium(cards[i], cards[j]):
                return True
    return False

def _is_premium_pair(c1, c2):
    r1, r2 = _rank(c1), _rank(c2)
    return r1 == r2 and frozenset([r1, r1]) in PREM_PAIRS

def _has_premium_pair(cards):
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _is_premium_pair(cards[i], cards[j]):
                return True
    return False

def _board_paired_and_we_weak(my_cards, community):
    if len(my_cards) < 2 or len(community) < 3:
        return False
    b_ranks = [_rank(c) for c in community]
    rc = Counter(b_ranks)
    bp_rank = next((r for r, cnt in rc.items() if cnt >= 2), None)
    if bp_rank is None:
        return False
    my_ranks = [_rank(c) for c in my_cards[:2]]
    if bp_rank in my_ranks:
        return False
    if my_ranks[0] == my_ranks[1] and frozenset([my_ranks[0], my_ranks[0]]) in PREM_PAIRS:
        return False
    return True

def _board_is_wet(community):
    """True if board is flush-heavy (monotone) or highly connected (3+ run)."""
    if len(community) < 3:
        return False
    sc = Counter(_suit(c) for c in community)
    conn = _max_connectivity([_rank(c) for c in community])
    return sc.most_common(1)[0][1] >= 3 or conn >= 3

def _normalize_action(raw):
    if not raw:
        return ""
    s = raw.strip().lower()
    return "" if s == "none" else s

# ── Bucket helpers ─────────────────────────────────────────────────────────────

def _bucket_keep(keep2):
    r1, r2 = _rank(keep2[0]), _rank(keep2[1])
    s1, s2 = _suit(keep2[0]), _suit(keep2[1])
    suited = (s1 == s2)
    if r1 == r2:
        if r1 >= RANK_9:   return "premium_pair"
        elif r1 >= RANK_8: return "medium_pair"
        else:              return "low_pair"
    eg = _effective_gap(keep2[0], keep2[1])
    if suited:
        if eg <= 1:   return "suited_connector"
        elif eg <= 3: return "suited_semi"
        else:         return "suited_gapper"
    else:
        if eg <= 1: return "offsuit_connector"
        else:       return "offsuit_other"

def _bucket_flop_simple(community):
    if not community or len(community) < 3:
        return "medium"
    suits  = [_suit(c) for c in community]
    ranks  = [_rank(c) for c in community]
    sc     = Counter(suits)
    rc     = Counter(ranks)
    score  = 0
    if sc.most_common(1)[0][1] >= 3:   score += 3
    elif sc.most_common(1)[0][1] >= 2: score += 1
    conn = _max_connectivity(ranks)
    if conn >= 3:   score += 2
    elif conn >= 2: score += 1
    if rc.most_common(1)[0][1] >= 2: score += 1
    if score >= 3: return "wet"
    if score >= 1: return "medium"
    return "dry"

def _bucket_strength(equity):
    if equity > 0.80:   return "monster"
    elif equity > 0.65: return "strong"
    elif equity > 0.50: return "good"
    elif equity > 0.35: return "marginal"
    return "weak"

def _bucket_to_call(to_call, pot_size):
    if to_call <= 0: return "none"
    ratio = to_call / max(pot_size + to_call, 1)
    if ratio <= 0.15: return "small"
    elif ratio <= 0.40: return "medium"
    return "large"

def _bucket_opp_discard(opp_discards):
    if len(opp_discards) < 3:
        return "unknown"
    ranks    = [_rank(c) for c in opp_discards]
    suits    = [_suit(c) for c in opp_discards]
    sc       = Counter(suits)
    rc       = Counter(ranks)
    sorted_r = sorted(ranks)
    conn = sum(1 for i in range(len(sorted_r) - 1)
               if sorted_r[i + 1] - sorted_r[i] == 1)
    if RANK_A in sorted_r and 0 in sorted_r:
        conn += 1
    if rc.most_common(1)[0][1] >= 2:  return "discarded_pair"
    if sc.most_common(1)[0][1] >= 2:  return "suited_cluster"
    if conn >= 2:                      return "connected_cluster"
    if RANK_A in ranks:                return "high_junk"
    if max(ranks) <= 5:                return "low_junk"
    return "mixed_discard"

def _hand_rank_category(my_cards, community):
    """Returns one of: flush | straight | full_house | trips | two_pair |
    one_pair | flush_draw | oesd | gutshot | nothing.
    Backward-compatible: callers only check 'one_pair' and 'nothing'.
    Draw categories are used by river logic to fold dead draws.
    """
    if len(my_cards) < 2 or len(community) < 3:
        return "nothing"
    all_cards = list(my_cards[:2]) + list(community)
    all_ranks = [_rank(c) for c in all_cards]
    all_suits = [_suit(c) for c in all_cards]
    rc = Counter(all_ranks)
    sc = Counter(all_suits)

    # Flush: 5 cards of same suit (2 hole + 3 community = 5 total)
    if sc.most_common(1)[0][1] >= 5:
        return "flush"

    # Straight: 5 consecutive unique ranks (Ace can be low as -1 or high as 8)
    unique_r = sorted(set(all_ranks))
    # Extend with Ace-low (-1) if Ace present
    unique_r_ext = ([-1] + unique_r) if (RANK_A in unique_r and 0 in unique_r) else unique_r
    best_run = cur_run = 1
    for i in range(1, len(unique_r_ext)):
        if unique_r_ext[i] - unique_r_ext[i - 1] == 1:
            cur_run += 1
            best_run = max(best_run, cur_run)
        else:
            cur_run = 1
    if best_run >= 5:
        return "straight"

    counts = sorted(rc.values(), reverse=True)
    # Full house: trips + pair
    if counts[0] >= 3 and len(counts) >= 2 and counts[1] >= 2:
        return "full_house"
    # Trips
    if counts[0] >= 3:
        return "trips"

    pairs = [r for r, cnt in rc.items() if cnt >= 2]
    if len(pairs) >= 2:
        return "two_pair"
    if len(pairs) == 1:
        return "one_pair"

    # Draw detection (meaningful on flop/turn; on river these are dead)
    # Flush draw: 4 cards same suit
    if sc.most_common(1)[0][1] == 4:
        return "flush_draw"
    # OESD: 4 consecutive ranks (open-ended straight draw)
    if best_run == 4:
        return "oesd"
    # Gutshot: 4 cards with exactly 1 gap in any 5-rank window
    for start in range(NUM_RANKS - 4):
        window = set(range(start, start + 5))
        if len(window & set(all_ranks)) == 4:
            return "gutshot"
    return "nothing"

# ── Alpha-style lightweight discard threat ─────────────────────────────────────

def _alpha_discard_threat(opp_discards, community):
    """Equity penalty when opponent's discards imply a strong kept hand.
    Upgraded to ALPHANiTV7 logic with stronger increments. Max penalty: 0.30."""
    if len(opp_discards) < 3:
        return 0.0
    threat = 0.0
    d_suits = [_suit(c) for c in opp_discards]
    d_ranks = sorted(_rank(c) for c in opp_discards)
    unique_d_suits = set(d_suits)

    # Rainbow discards (3 different suits) → diverse dump implies kept flush/high structure
    if len(unique_d_suits) == 3:
        threat += 0.08  # was 0.04

    # All low cards discarded (rank ≤ 5) → kept high/strong structure
    if all(r <= 5 for r in d_ranks):
        threat += 0.07  # was 0.05, threshold 4→5

    # All unique ranks in discards → implies kept pair or flush (no repeats dumped)
    if len(set(d_ranks)) == 3:
        threat += 0.05  # NEW

    # Board has a flush suit (2+) that they did NOT discard → likely kept that suit
    if community:
        comm_suits = [_suit(c) for c in community]
        for s in range(3):
            if comm_suits.count(s) >= 2 and s not in unique_d_suits:
                threat += 0.10  # was 0.09
                break

    return min(0.30, threat)  # was 0.18

# ══════════════════════════════════════════════════════════════════════════════
# PlayerAgent
# ══════════════════════════════════════════════════════════════════════════════

class PlayerAgent(Agent):

    _STAT_PRIORS = {
        "fold_to_bet":       0.35,
        "fold_to_raise":     0.35,
        "check_raise":       0.05,
        "call_down":         0.40,
        "opp_aggression":    0.25,
        "opp_avg_bet_frac":  0.50,
        "opp_preflop_raise": 0.30,
    }

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self._env      = PokerEnv()
        self.evaluator = self._env.evaluator
        self.hand_number = 0
        self._rng        = random.Random(42)

        # ── Version stamp ─────────────────────────────────────────────────
        _profile_name = os.path.basename(_profile_path) if os.path.isfile(_profile_path) else "defaults"
        self.logger.info(
            f"[LambdaV1 INIT] profile={_profile_name} "
            f"early_phase={EARLY_PHASE_HANDS} agro_mult={EARLY_AGRO_MULT} "
            f"semi_bluff={SEMI_BLUFF_FREQ} semi_lo={SEMI_BLUFF_LO} semi_hi={SEMI_BLUFF_HI} "
            f"one_pair_eq={ONE_PAIR_RAISE_EQ} one_pair_fold={ONE_PAIR_RAISE_FOLD} "
            f"river_call_cap={RIVER_CALL_POT_RATIO_MAX}"
        )

        self._stats = {
            "fold_to_bet":       [0, 0],
            "fold_to_raise":     [0, 0],
            "check_raise":       [0, 0],
            "call_down":         [0, 0],
            "opp_aggression":    [0, 0],
            "opp_avg_bet_frac":  [0.0, 0],
            "opp_preflop_raise": [0, 0],
        }

        self._hand_mode       = MODE_GTO
        self._hand_override   = None
        self._agro_mult       = 1.0
        self._betting_history = {"bet_flop": False, "bet_turn": False}
        self._last_was_bet    = False
        self._last_street     = 0
        self._opp_folded      = False
        self._we_folded       = False
        self._prev_opp_bet    = 0
        self._prev_my_bet     = 0

        self._running_pnl  = 0
        self._mode_reason  = "warmup"
        self._prev_mode    = MODE_GTO
        self._last_equity  = 0.5
        self._last_aggro   = 0.5

    def __name__(self):
        return "LambdaV1"

    # ── Time-aware sim budget ─────────────────────────────────────────────────

    def _sim_budget(self, time_left):
        """Returns (discard_sims_per_combo, postflop_sims)."""
        if time_left > 600:   return 200, 400
        elif time_left > 300: return 150, 300
        elif time_left > 150: return 80,  180
        elif time_left > 60:  return 50,  120
        return 30, 80

    # ── MC Equity ─────────────────────────────────────────────────────────────

    def _mc_equity(self, my2, community, dead, num_sims=200):
        known     = set(my2) | set(community) | dead
        remaining = [c for c in range(DECK_SIZE) if c not in known]
        needed    = 2 + (5 - len(community))
        if needed > len(remaining):
            return 0.5
        wins  = 0.0
        total = 0
        for _ in range(num_sims):
            sample     = self._rng.sample(remaining, needed)
            opp        = sample[:2]
            full_board = list(community) + sample[2:]
            my_hand    = [_int_to_card(c) for c in my2]
            opp_hand   = [_int_to_card(c) for c in opp]
            board      = [_int_to_card(c) for c in full_board]
            mr         = self.evaluator.evaluate(my_hand, board)
            orr        = self.evaluator.evaluate(opp_hand, board)
            if mr < orr:    wins += 1.0
            elif mr == orr: wins += 0.5
            total += 1
        return wins / total if total > 0 else 0.5

    def _range_equity(self, my2, community, dead, opp_b, num_sims):
        """Blend MC equity (70%) with Libratus range-vs-range table (30%)."""
        mc_eq = self._mc_equity(my2, community, dead, num_sims)
        if opp_b == "unknown" or len(community) < 3:
            return mc_eq
        board_b  = _bucket_flop_simple(community)
        opp_dist = POSTERIOR.get(f"{opp_b}|{board_b}", {})
        if not opp_dist:
            return mc_eq
        my_kb      = _bucket_keep(my2)
        matchup_eq = 0.0
        wsum       = 0.0
        for opp_kb, prob in opp_dist.items():
            matchup_eq += prob * MATCHUPS.get(f"{my_kb}|{opp_kb}", 0.5)
            wsum += prob
        if wsum > 0:
            return 0.7 * mc_eq + 0.3 * (matchup_eq / wsum)
        return mc_eq

    def _preflop_equity(self, my5, p_sims):
        """Best 2-card keep equity from a 5-card preflop hand."""
        if _has_premium_pair(my5):
            return 0.75
        sims = max(p_sims // 10, 15)
        best = 0.0
        for i, j in combinations(range(len(my5)), 2):
            keep = [my5[i], my5[j]]
            toss = set(my5[k] for k in range(len(my5)) if k not in (i, j))
            eq = self._mc_equity(keep, [], toss, num_sims=sims)
            if eq > best:
                best = eq
        return best if best > 0 else 0.45

    # ── Stat helpers ──────────────────────────────────────────────────────────

    def _safe_rate(self, key):
        """Confidence-aware stat: blend observed rate with prior based on sample size.
        Low-count observations are pulled toward the prior to reduce early misclassification."""
        folds, total = self._stats.get(key, [0, 0])
        if total == 0:
            return self._STAT_PRIORS.get(key, 0.35)
        # Confidence ramp: full trust at 15+ observations; scales linearly below
        confidence   = min(1.0, total / 15.0)
        observed     = folds / total
        prior        = self._STAT_PRIORS.get(key, 0.35)
        return confidence * observed + (1.0 - confidence) * prior

    def _fmt_stat(self, key):
        num, den = self._stats.get(key, [0, 0])
        rate = self._safe_rate(key)
        return f"{rate:.2f}({num}/{den})"

    def _avg_bet_frac(self):
        total_sum, count = self._stats["opp_avg_bet_frac"]
        if count < 5:
            return self._STAT_PRIORS["opp_avg_bet_frac"]
        return total_sum / count

    def _total_obs(self):
        return self.hand_number

    # ── Opponent action processing ────────────────────────────────────────────

    def _process_opponent_action(self, observation, opp_action, last_was_bet, last_street):
        if not opp_action:
            return
        if opp_action == "fold":
            self._opp_folded = True
        if last_was_bet:
            if opp_action == "fold":
                self._stats["fold_to_bet"][0]  += 1
                self._stats["fold_to_bet"][1]  += 1
                self._stats["fold_to_raise"][0] += 1
                self._stats["fold_to_raise"][1] += 1
            elif opp_action in ("call", "check", "raise"):
                self._stats["fold_to_bet"][1]  += 1
                self._stats["fold_to_raise"][1] += 1
                if opp_action == "call" and last_street >= 1:
                    self._stats["call_down"][0] += 1
                    self._stats["call_down"][1] += 1
                elif last_street >= 1:
                    self._stats["call_down"][1] += 1
        if not last_was_bet and opp_action == "raise":
            self._stats["check_raise"][0] += 1
            self._stats["check_raise"][1] += 1
            if self._hand_override is None:
                self._hand_override = MODE_VALUE
        elif not last_was_bet and opp_action in ("check", "call", "fold"):
            self._stats["check_raise"][1] += 1
        elif last_was_bet and opp_action == "raise":
            if self._hand_override is None:
                self._hand_override = MODE_VALUE
        if opp_action in ("raise", "call", "check"):
            self._stats["opp_aggression"][1] += 1
            if opp_action == "raise":
                self._stats["opp_aggression"][0] += 1
                opp_bet_obs = observation.get("opp_bet", 0)
                my_bet_obs  = observation.get("my_bet", 0)
                pot_obs     = observation.get("pot_size", opp_bet_obs + my_bet_obs)
                raise_size  = max(0, opp_bet_obs - my_bet_obs)
                if pot_obs > 0 and raise_size > 0:
                    frac = raise_size / pot_obs
                    self._stats["opp_avg_bet_frac"][0] += frac
                    self._stats["opp_avg_bet_frac"][1] += 1
        if last_street == 0 and opp_action in ("raise", "call", "check", "fold"):
            self._stats["opp_preflop_raise"][1] += 1
            if opp_action == "raise":
                self._stats["opp_preflop_raise"][0] += 1

    # ── Mode selection ────────────────────────────────────────────────────────

    def _select_mode(self):
        """Returns (mode, agro_mult).
        Hands 1-EARLY_PHASE_HANDS: AGRO-lite unless extreme opponent evidence.
        After warmup: full opponent-model routing."""
        fold_to_bet = self._safe_rate("fold_to_bet")
        opp_aggro   = self._safe_rate("opp_aggression")
        call_down   = self._safe_rate("call_down")
        cr_rate     = self._safe_rate("check_raise")
        pf_raise    = self._safe_rate("opp_preflop_raise")

        # ── Early phase: AGRO-lite with growing multiplier ────────────────────
        if self.hand_number <= EARLY_PHASE_HANDS:
            ramp = min(1.0, self.hand_number / float(EARLY_PHASE_HANDS))
            mult = round(EARLY_AGRO_MULT * (1.0 + 0.15 * ramp), 2)
            # Override only on strong, early evidence (min 5 observations)
            if self._stats["opp_aggression"][1] >= 5 and opp_aggro > EARLY_EXTREME_AGGRO:
                self._mode_reason = f"early_trap(agg={opp_aggro:.2f})"
                return MODE_TRAP, 1.0
            if self._stats["fold_to_bet"][1] >= 5 and fold_to_bet < EARLY_EXTREME_FOLD:
                self._mode_reason = f"early_value(fb={fold_to_bet:.2f})"
                return MODE_VALUE, 1.0
            self._mode_reason = f"early_agro(h={self.hand_number},x{mult:.1f})"
            return MODE_AGRO, mult

        # ── Post-warmup: GTO for first handful of hands ───────────────────────
        if self._total_obs() < 5:
            self._mode_reason = "warmup"
            return MODE_GTO, 1.0

        ramp = min(1.0, (self.hand_number - 5) / 7.0)

        # ── Soft blending: graduated multipliers instead of binary switches ──
        # Maniac: hard TRAP only at extreme aggression; both thresholds return TRAP
        if opp_aggro > 0.55:
            self._mode_reason = f"maniac_hard(agg={opp_aggro:.2f})"
            return MODE_TRAP, 1.0
        if opp_aggro > 0.45:
            self._mode_reason = f"maniac_soft(agg={opp_aggro:.2f})"
            return MODE_TRAP, 1.0

        # Overfolder: graduated multiplier based on fold rate magnitude
        if fold_to_bet > 0.55:
            mult = round(1.0 + 0.5 * ramp, 2)
            self._mode_reason = f"overfold_strong(fb={fold_to_bet:.2f})"
            return MODE_AGRO, mult
        if fold_to_bet > 0.42:
            mult = round(1.0 + 0.3 * ramp, 2)
            self._mode_reason = f"overfold_mild(fb={fold_to_bet:.2f})"
            return MODE_AGRO, mult

        # Station: hard vs mild distinction
        if fold_to_bet < 0.22 or call_down > 0.62:
            self._mode_reason = f"station_hard(fb={fold_to_bet:.2f},cd={call_down:.2f})"
            return MODE_VALUE, 1.0
        if fold_to_bet < 0.30 or call_down > 0.55:
            self._mode_reason = f"station_mild(fb={fold_to_bet:.2f},cd={call_down:.2f})"
            return MODE_VALUE, 1.0

        if cr_rate > 0.10:
            self._mode_reason = f"checkraise({cr_rate:.2f})"
            return MODE_TRAP, 1.0
        if opp_aggro < 0.15 and self._stats["opp_aggression"][1] >= 8:
            mult = round(1.0 + 0.3 * ramp, 2)
            self._mode_reason = f"passive(agg={opp_aggro:.2f})"
            return MODE_AGRO, mult
        if pf_raise > 0.50:
            mult = round(1.0 + 0.2 * ramp, 2)
            self._mode_reason = f"loose_pf({pf_raise:.2f})"
            return MODE_AGRO, mult
        if pf_raise < 0.12:
            self._mode_reason = f"limper({pf_raise:.2f})"
            return MODE_VALUE, 1.0

        self._mode_reason = "default"
        mode, mult = MODE_AGRO, 1.0

        # ── Transition smoothing: hands 50-100 stay slightly boosted ─────────
        # Avoids cliff drop from early-phase 1.38x back to 1.0x
        if (EARLY_PHASE_HANDS < self.hand_number <= EARLY_PHASE_HANDS + 50
                and mode == MODE_AGRO):
            t_frac = 1.0 - (self.hand_number - EARLY_PHASE_HANDS) / 50.0
            bonus  = round((EARLY_AGRO_MULT - 1.0) * t_frac, 2)
            mult   = round(mult + bonus, 2)

        return mode, mult

    # ── Bleed-out lock ────────────────────────────────────────────────────────

    def _bleedout_check(self, valid):
        """If chip lead > max remaining forced losses, lock into fold/check."""
        hands_remaining = max(0, TOTAL_HANDS - self.hand_number)
        sb_left   = (hands_remaining + 1) // 2
        bb_left   = hands_remaining // 2
        max_bleed = sb_left * 1 + bb_left * 2
        if self._running_pnl > max_bleed:
            if valid[FOLD]:  return (FOLD,  0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
        return None

    # ════════════════════════════════════════════════════════════════════════
    # DISCARD — Mode-aware KeepScore
    # ════════════════════════════════════════════════════════════════════════

    def _pressure_potential(self, keep2, community, opp_discards):
        score   = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]
        if k_suits[0] == k_suits[1]: score += 0.40
        eg = _effective_gap(keep2[0], keep2[1])
        if eg <= 1:   score += 0.30
        elif eg <= 2: score += 0.20
        elif eg <= 3: score += 0.10
        if community:
            b_suits = [_suit(c) for c in community]
            for s in set(k_suits):
                if sum(1 for bs in b_suits if bs == s) >= 2:
                    score += 0.15
                    break
        max_rank = max(k_ranks)
        if max_rank >= RANK_A:   score += 0.15
        elif max_rank >= RANK_9: score += 0.08
        return min(score, 1.0)

    def _draw_density(self, keep2, community):
        score   = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]
        if not community:
            if k_suits[0] == k_suits[1]: score += 0.30
            eg = _effective_gap(keep2[0], keep2[1])
            if eg <= 1:   score += 0.25
            elif eg <= 2: score += 0.15
            return min(score, 1.0)
        b_suits = [_suit(c) for c in community]
        b_ranks = [_rank(c) for c in community]
        for s in set(k_suits):
            total = (sum(1 for cs in b_suits if cs == s) +
                     sum(1 for cs in k_suits if cs == s))
            if total >= 4:   score += 0.40
            elif total == 3: score += 0.25
            elif total == 2: score += 0.08
        conn = _max_connectivity(k_ranks + b_ranks)
        if conn >= 5:   score += 0.35
        elif conn >= 4: score += 0.20
        elif conn >= 3: score += 0.08
        return min(score, 1.0)

    def _made_hand_value(self, keep2, community):
        score   = 0.0
        k_ranks = [_rank(c) for c in keep2]
        if k_ranks[0] == k_ranks[1]:
            r = k_ranks[0]
            if r >= RANK_9:   score += 0.80
            elif r >= RANK_8: score += 0.55
            elif r >= 4:      score += 0.35
            else:             score += 0.20
            return min(score, 1.0)
        if not community:
            return 0.0
        b_ranks     = [_rank(c) for c in community]
        rank_counts = Counter(k_ranks + b_ranks)
        max_count   = rank_counts.most_common(1)[0][1]
        pairs       = [r for r, v in rank_counts.items() if v >= 2]
        if max_count >= 3:    score += 0.70
        elif len(pairs) >= 2: score += 0.50
        elif len(pairs) == 1:
            pr = pairs[0]
            if pr in k_ranks:
                if pr >= RANK_A:   score += 0.35
                elif pr >= RANK_8: score += 0.20
                else:              score += 0.10
        return min(score, 1.0)

    def _blocker_value(self, keep2, community, opp_discards):
        score   = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]
        if RANK_A in k_ranks: score += 0.15
        if RANK_9 in k_ranks: score += 0.08
        if not community:
            return min(score, 1.0)
        b_suit_c = Counter(_suit(c) for c in community)
        dom_suit, dom_cnt = b_suit_c.most_common(1)[0]
        if dom_cnt >= 2:
            score += 0.20 * sum(1 for s in k_suits if s == dom_suit)
        return min(score, 1.0)

    def _story_flexibility(self, keep2, community):
        score   = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]
        if k_suits[0] == k_suits[1]: score += 0.35
        eg = _effective_gap(keep2[0], keep2[1])
        if eg <= 1:   score += 0.30
        elif eg <= 2: score += 0.20
        max_r = max(k_ranks)
        if max_r >= RANK_A:   score += 0.20
        elif max_r >= RANK_9: score += 0.10
        if k_ranks[0] == k_ranks[1] and k_ranks[0] >= RANK_8:
            score += 0.30
        return min(score, 1.0)

    def _keep_score(self, keep2, community, opp_discards, toss_dead, d_sims, mode=MODE_AGRO):
        eq  = self._mc_equity(keep2, community, toss_dead, num_sims=d_sims)
        pp  = self._pressure_potential(keep2, community, opp_discards)
        dd  = self._draw_density(keep2, community)
        mhv = self._made_hand_value(keep2, community)
        bv  = self._blocker_value(keep2, community, opp_discards)
        sf  = self._story_flexibility(keep2, community)
        if mode in (MODE_VALUE, MODE_TRAP):
            return 3.5*eq + 0.6*pp + 1.7*dd + 2.0*mhv + 0.8*bv + 0.2*sf, eq
        # AGRO/GTO: profile-tunable weights, equity-first by default
        return _KW_EQ*eq + _KW_PP*pp + _KW_DD*dd + _KW_MHV*mhv + _KW_BV*bv + _KW_SF*sf, eq

    def _choose_keep(self, my_cards, community, opp_discards, d_sims, mode=MODE_AGRO):
        # Postflop: deterministic engine ranks all 10 pairs, then MC overlay on top-3
        if community and len(community) >= 3:
            blind_pos = 1 if not opp_discards else 0
            # Get deterministic engine's first choice as baseline
            engine_ij = _choose_keep_postflop(my_cards, community, opp_discards, blind_pos)
            # MC overlay: score all 10 keep-pairs via keep-score, take top-3 by heuristic,
            # then re-rank using 65% heuristic + 35% MC equity
            if len(my_cards) == 5:
                heuristic_scores = []
                for i, j in combinations(range(5), 2):
                    keep      = [my_cards[i], my_cards[j]]
                    toss      = [my_cards[k] for k in range(5) if k not in (i, j)]
                    toss_dead = set(toss) | (set(opp_discards) if opp_discards else set())
                    sc, eq    = self._keep_score(keep, community, opp_discards,
                                                 toss_dead, d_sims, mode)
                    heuristic_scores.append((i, j, sc, eq))
                # Normalise heuristic scores to [0,1] range
                sc_vals = [sc for _, _, sc, _ in heuristic_scores]
                sc_min, sc_max = min(sc_vals), max(sc_vals)
                sc_range = (sc_max - sc_min) if sc_max > sc_min else 1.0
                # Top-3 by heuristic
                top3 = sorted(heuristic_scores, key=lambda x: x[2], reverse=True)[:3]
                best_blend = -1.0
                best_blend_ij = engine_ij  # fallback to engine
                for i, j, sc, _ in top3:
                    keep      = [my_cards[i], my_cards[j]]
                    toss      = [my_cards[k] for k in range(5) if k not in (i, j)]
                    toss_dead = set(toss) | (set(opp_discards) if opp_discards else set())
                    mc_eq     = self._mc_equity(keep, community, toss_dead,
                                                num_sims=max(d_sims, 40))
                    norm_sc   = (sc - sc_min) / sc_range
                    blend     = 0.65 * norm_sc + 0.35 * mc_eq
                    if blend > best_blend:
                        best_blend    = blend
                        best_blend_ij = (i, j)
                return best_blend_ij
            return engine_ij
        # Preflop fallback: MC keep-score over all 10 combos
        best_score    = -999.0
        best_ij       = (0, 1)
        best_eq_ij    = (0, 1)
        best_raw_eq   = -1.0
        chosen_raw_eq = 0.0
        candidates    = []
        for i, j in combinations(range(len(my_cards)), 2):
            keep      = [my_cards[i], my_cards[j]]
            toss      = [my_cards[k] for k in range(len(my_cards)) if k not in (i, j)]
            toss_dead = set(toss) | (set(opp_discards) if opp_discards else set())
            sc, eq    = self._keep_score(keep, community, opp_discards, toss_dead, d_sims, mode)
            candidates.append((i, j, sc, eq))
            if eq > best_raw_eq:
                best_raw_eq = eq
                best_eq_ij  = (i, j)
            if sc > best_score:
                best_score    = sc
                best_ij       = (i, j)
                chosen_raw_eq = eq
        ties = [(i, j) for i, j, sc, _ in candidates if best_score - sc < 0.03]
        if len(ties) > 1:
            best_ij = self._rng.choice(ties)
            chosen_raw_eq = next(eq for i, j, _, eq in candidates if (i, j) == best_ij)
        if best_raw_eq - chosen_raw_eq > 0.10:
            best_ij = best_eq_ij
        return best_ij

    # ════════════════════════════════════════════════════════════════════════
    # GTO_DEFAULT — Libratus policy table + range-vs-range equity
    # ════════════════════════════════════════════════════════════════════════

    def _gto_equity(self, my_cards, community, opp_discards, my_discards, p_sims):
        dead   = set()
        if my_discards:  dead |= set(my_discards)
        if opp_discards: dead |= set(opp_discards)
        if len(my_cards) > 2 and not community:
            return self._preflop_equity(my_cards, p_sims)
        cards2 = my_cards[:2] if len(my_cards) > 2 else my_cards
        if len(cards2) == 2 and len(community) >= 3:
            opp_b  = _bucket_opp_discard(opp_discards) if len(opp_discards) >= 3 else "unknown"
            equity = self._range_equity(cards2, community, dead, opp_b, p_sims)
        else:
            equity = self._mc_equity(cards2, community, dead, num_sims=max(p_sims // 3, 30))
        if _board_paired_and_we_weak(cards2, community):
            equity -= 0.08
        if len(opp_discards) >= 3:
            opp_b = _bucket_opp_discard(opp_discards)
            adj   = {"low_junk": -0.04, "suited_cluster": -0.03,
                     "connected_cluster": -0.02, "high_junk": 0.02, "discarded_pair": 0.03}
            equity += adj.get(opp_b, 0.0)
            equity -= _alpha_discard_threat(opp_discards, community)
        return _clamp(equity, 0.0, 0.98)

    def _gto_action(self, street, my_cards, community, opp_discards, my_discards,
                    valid, min_raise, max_raise, my_bet, opp_bet, pot_size,
                    blind_pos, p_sims):
        to_call  = max(0, opp_bet - my_bet)
        position = "sb" if blind_pos == 0 else "bb"
        equity   = self._gto_equity(my_cards, community, opp_discards, my_discards, p_sims)
        self._last_equity = equity
        strength = _bucket_strength(equity)
        board_b  = _bucket_flop_simple(community) if street >= 1 else "any"
        tc_b     = _bucket_to_call(to_call, pot_size)
        key    = str((street, position, strength, board_b, tc_b))
        policy = POLICY.get(key)
        if not policy:
            key2   = str((street, position, strength, "medium", "none"))
            policy = POLICY.get(key2, {
                "fold": 0.20, "check_call": 0.40, "small_bet": 0.25,
                "medium_bet": 0.10, "large_bet": 0.05, "jam": 0.0
            })
        policy = dict(policy)
        fold_r = self._safe_rate("fold_to_raise")
        agg_r  = self._safe_rate("opp_aggression")
        if fold_r > 0.50:
            policy["small_bet"]  = policy.get("small_bet", 0) + 0.12
            policy["fold"]       = max(0.0, policy.get("fold", 0) - 0.12)
        elif fold_r < 0.20:
            policy["small_bet"]  = max(0.0, policy.get("small_bet", 0) - 0.10)
            policy["check_call"] = policy.get("check_call", 0) + 0.10
        if agg_r > 0.55 and strength in ("marginal", "weak"):
            policy["check_call"] = max(0.0, policy.get("check_call", 0) - 0.12)
            policy["fold"]       = policy.get("fold", 0) + 0.12
        action = self._sample_policy(policy)
        return self._policy_to_tuple(action, valid, min_raise, max_raise,
                                     pot_size, to_call, equity)

    def _sample_policy(self, policy):
        r     = self._rng.random()
        cumul = 0.0
        for act, prob in policy.items():
            cumul += prob
            if r < cumul:
                return act
        return "check_call"

    def _policy_to_tuple(self, action, valid, min_raise, max_raise,
                         pot_size, to_call, equity):
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        if action == "fold":
            if valid[FOLD]:  return (FOLD,  0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)
        if action == "check_call":
            if to_call > 0 and valid[CALL]: return (CALL,  0, 0, 0)
            if valid[CHECK]:                return (CHECK, 0, 0, 0)
            if valid[CALL]:                 return (CALL,  0, 0, 0)
            return (CHECK, 0, 0, 0)
        if action == "jam":
            if valid[RAISE] and max_raise > 0: return (RAISE, max_raise, 0, 0)
            if valid[CALL]:                     return (CALL,  0, 0, 0)
            if valid[CHECK]:                    return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)
        pot_ref = max(pot_size, 1)
        if action == "small_bet":    frac = self._rng.uniform(0.30, 0.45)
        elif action == "medium_bet": frac = self._rng.uniform(0.55, 0.75)
        elif action == "large_bet":  frac = self._rng.uniform(0.85, 1.10)
        else:                        frac = 0.50
        amount = _clamp(int(pot_ref * frac), min_raise, max_raise)
        if valid[RAISE] and max_raise >= min_raise: return (RAISE, amount, 0, 0)
        if valid[CALL]:                              return (CALL,  0, 0, 0)
        if valid[CHECK]:                             return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    # ════════════════════════════════════════════════════════════════════════
    # AGRO_EXPLOIT — Cap / Story / Backup scoring + semi-bluff band
    # ════════════════════════════════════════════════════════════════════════

    def _cap_score(self, opp_discards, community, my_discards):
        if not opp_discards or len(opp_discards) < 3:
            return 0.20
        score      = 0.0
        opp_suit_c = Counter(_suit(c) for c in opp_discards)
        opp_ranks  = sorted(_rank(c) for c in opp_discards)
        my_suit_d  = Counter(_suit(c) for c in my_discards) if my_discards else Counter()
        b_suit_c   = Counter(_suit(c) for c in community)   if community   else Counter()
        for suit, opp_cnt in opp_suit_c.items():
            if opp_cnt < 2: continue
            base = 0.6 if opp_cnt == 3 else 0.3
            board_in = b_suit_c.get(suit, 0)
            if board_in >= 3:   base += 0.35
            elif board_in >= 2: base += 0.15
            my_d = my_suit_d.get(suit, 0)
            if my_d >= 2:   base *= 0.20
            elif my_d == 1: base *= 0.70
            score += base
        opp_conn = sum(1 for i in range(len(opp_ranks) - 1)
                       if opp_ranks[i + 1] - opp_ranks[i] <= 1)
        if opp_conn >= 2 and community and len(community) >= 3:
            b_conn = _max_connectivity([_rank(c) for c in community])
            if b_conn >= 3:   score += 0.20
            elif b_conn >= 2: score += 0.10
        elif opp_conn >= 2:
            score += 0.08
        if sum(1 for c in opp_discards if _rank(c) >= RANK_8) >= 2:
            score += 0.10
        return min(score, 1.0)

    def _story_score(self, my_cards, community, my_discards, street):
        if not community or len(community) < 3:
            return 0.40
        b_suits  = [_suit(c) for c in community]
        b_ranks  = [_rank(c) for c in community]
        b_suit_c = Counter(b_suits)
        dom_suit, dom_cnt = b_suit_c.most_common(1)[0]
        flush_base    = 0.45 if dom_cnt >= 3 else (0.20 if dom_cnt >= 2 else 0.0)
        b_conn        = _max_connectivity(b_ranks)
        straight_base = 0.35 if b_conn >= 3 else (0.15 if b_conn >= 2 else 0.0)
        score = max(flush_base, straight_base)
        if my_cards and len(my_cards) == 2:
            k_suits = [_suit(c) for c in my_cards]
            k_ranks = [_rank(c) for c in my_cards]
            if flush_base > 0:
                n_dom = sum(1 for s in k_suits if s == dom_suit)
                if n_dom >= 2:   score += 0.20
                elif n_dom >= 1: score += 0.10
            if straight_base > 0:
                all_conn = _max_connectivity(k_ranks + b_ranks)
                if all_conn >= 5:   score += 0.20
                elif all_conn >= 4: score += 0.10
        if my_discards and flush_base > 0:
            d_dom = Counter(_suit(c) for c in my_discards).get(dom_suit, 0)
            if d_dom >= 2:   score -= 0.30
            elif d_dom >= 1: score -= 0.10
        if street >= 2 and self._betting_history.get("bet_flop"):  score += 0.10
        if street == 3 and self._betting_history.get("bet_turn"):  score += 0.10
        if self._hand_override is not None:
            score = max(0.0, score - 0.30)
        return _clamp(score, 0.0, 1.0)

    def _backup_score(self, my_cards, community, dead, equity, opp_discards):
        score = equity * 0.6
        if not my_cards or len(my_cards) != 2:
            return min(score, 1.0)
        k_suits = [_suit(c) for c in my_cards]
        k_ranks = [_rank(c) for c in my_cards]
        if k_ranks[0] == k_ranks[1]:
            if k_ranks[0] >= RANK_9:   score += 0.15
            elif k_ranks[0] >= RANK_8: score += 0.08
            else:                      score += 0.04
        if community:
            b_suits = [_suit(c) for c in community]
            b_ranks = [_rank(c) for c in community]
            for s in set(k_suits):
                total = (sum(1 for cs in b_suits if cs == s) +
                         sum(1 for cs in k_suits if cs == s))
                if total >= 4:   score += 0.15
                elif total == 3: score += 0.08
            conn = _max_connectivity(k_ranks + b_ranks)
            if conn >= 5:   score += 0.15
            elif conn >= 4: score += 0.08
            b_suit_c = Counter(b_suits)
            if b_suit_c:
                dom_s, dom_c = b_suit_c.most_common(1)[0]
                if dom_c >= 2:
                    score += 0.05 * sum(1 for s in k_suits if s == dom_s)
        return min(score, 1.0)

    def _hard_triggers(self, my_cards, community, opp_discards, my_discards):
        bonus = 0.0
        brake = 0.0
        if not opp_discards or len(opp_discards) < 3:
            return bonus, brake
        opp_suit_c = Counter(_suit(c) for c in opp_discards)
        my_suit_d  = Counter(_suit(c) for c in my_discards) if my_discards else Counter()
        b_suit_c   = Counter(_suit(c) for c in community)   if community   else Counter()
        for suit, opp_cnt in opp_suit_c.items():
            board_cnt = b_suit_c.get(suit, 0)
            my_d_cnt  = my_suit_d.get(suit, 0)
            if opp_cnt == 3 and board_cnt >= 2 and my_d_cnt <= 1:
                bonus += 0.35
            elif opp_cnt == 2:
                if board_cnt >= 3:
                    bonus += 0.20
                elif board_cnt >= 2:
                    our = sum(1 for c in my_cards if _suit(c) == suit) if my_cards else 0
                    if our >= 1 or my_d_cnt == 0:
                        bonus += 0.12
        opp_ranks = sorted(_rank(c) for c in opp_discards)
        opp_conn  = sum(1 for i in range(len(opp_ranks) - 1)
                        if opp_ranks[i + 1] - opp_ranks[i] <= 1)
        if opp_conn >= 2 and community and len(community) >= 3:
            if _max_connectivity([_rank(c) for c in community]) >= 2:
                bonus += 0.15
        if community and len(community) >= 3 and my_cards and len(my_cards) == 2:
            k_suits_e = [_suit(c) for c in my_cards]
            k_ranks_e = [_rank(c) for c in my_cards]
            b_suits_e = [_suit(c) for c in community]
            best_ft   = max(
                sum(1 for cs in b_suits_e if cs == s) + sum(1 for cs in k_suits_e if cs == s)
                for s in set(k_suits_e)
            )
            if (best_ft >= 3 or
                    _max_connectivity(k_ranks_e + [_rank(c) for c in community]) >= 4):
                if self._safe_rate("fold_to_bet") > 0.45:
                    bonus += 0.12
        if community and my_discards:
            b_sc2 = Counter(_suit(c) for c in community)
            if b_sc2:
                dom_s2, dom_c2 = b_sc2.most_common(1)[0]
                if dom_c2 >= 2 and my_suit_d.get(dom_s2, 0) >= 2:
                    brake += 0.30
        if self._safe_rate("fold_to_bet") < 0.20: brake += 0.20
        if self._safe_rate("check_raise")  > 0.15: brake += 0.15
        brake = min(brake, 0.45)
        return bonus, brake

    def _aggro_to_action(self, aggro, equity, street, valid, min_raise,
                         max_raise, pot_size, to_call, bluff_enable):
        pot_ref  = max(pot_size, 1)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        eff      = aggro if bluff_enable else aggro * (0.5 + 0.5 * equity)
        if street == 3:
            if eff > 0.70:
                if valid[RAISE] and max_raise >= min_raise:
                    if eff > 0.85:
                        return (RAISE, max_raise, 0, 0)
                    amt = _clamp(int(pot_ref * self._rng.uniform(0.80, 1.00)), min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            elif eff < 0.30:
                if to_call > 0:
                    if equity > pot_odds + 0.15 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if equity > pot_odds and to_call <= pot_ref * 0.25 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[FOLD]: return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            else:
                if to_call > 0:
                    if equity > pot_odds and to_call <= pot_ref * 0.35 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if equity > pot_odds + 0.15 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[FOLD]: return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
        else:
            if eff > 0.70:
                if valid[RAISE] and max_raise >= min_raise:
                    frac = self._rng.uniform(0.70, 1.10) if street == 2 \
                           else self._rng.uniform(0.60, 0.85)
                    return (RAISE, _clamp(int(pot_ref * frac), min_raise, max_raise), 0, 0)
                if to_call > 0 and valid[CALL]: return (CALL,  0, 0, 0)
                if valid[CHECK]:                return (CHECK, 0, 0, 0)
            elif eff > 0.55:
                if valid[RAISE] and max_raise >= min_raise:
                    amt = _clamp(int(pot_ref * self._rng.uniform(0.45, 0.70)), min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
            elif eff > 0.40:
                if self._rng.random() < 0.55 and valid[RAISE] and max_raise >= min_raise:
                    amt = _clamp(int(pot_ref * self._rng.uniform(0.25, 0.45)), min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
                if to_call > 0 and equity > pot_odds and valid[CALL]: return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            elif eff > 0.25:
                if to_call > 0:
                    if equity > pot_odds + 0.15 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if equity > pot_odds and valid[CALL]: return (CALL, 0, 0, 0)
                    if valid[FOLD]: return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            else:
                if to_call > 0 and equity > pot_odds + 0.15 and valid[CALL]:
                    return (CALL, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                if valid[FOLD]:  return (FOLD,  0, 0, 0)
        if valid[CHECK]:  return (CHECK, 0, 0, 0)
        if to_call > 0 and equity > pot_odds and valid[CALL]: return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _agro_preflop(self, my_cards, valid, min_raise, max_raise,
                      my_bet, opp_bet, pot_size, blind_pos, p_sims,
                      in_early_phase=False):
        is_prem = _has_any_premium(my_cards)
        if len(my_cards) > 2:
            equity = self._preflop_equity(my_cards, p_sims)
        else:
            equity = self._mc_equity(my_cards, [], set(), num_sims=max(p_sims // 3, 30))
        to_call  = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        noise    = self._rng.uniform(0.90, 1.10)
        pot_ref  = max(pot_size, 1)

        # Early phase: wider ranges, bigger opens
        eq_thresh_sb = 0.40 if in_early_phase else 0.45
        eq_thresh_bb = 0.35 if in_early_phase else 0.40
        open_mult    = 2.8  if in_early_phase else 2.5
        reraise_mult = 3.2  if in_early_phase else 3.0
        sb_raise_p   = 0.75 if in_early_phase else 0.65
        bb_3bet_p    = 0.25 if in_early_phase else 0.15

        if blind_pos == 0:  # SB
            if is_prem:
                if self._rng.random() < 0.10:
                    if valid[CALL]:  return (CALL,  0, 0, 0)
                    if valid[CHECK]: return (CHECK, 0, 0, 0)
                if valid[RAISE] and max_raise >= min_raise:
                    return (RAISE, _clamp(int(pot_ref * open_mult * noise), min_raise, max_raise), 0, 0)
            elif equity > eq_thresh_sb:
                if self._rng.random() < sb_raise_p and valid[RAISE] and max_raise >= min_raise:
                    return (RAISE, _clamp(int(pot_ref * 2.0 * noise), min_raise, max_raise), 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            else:
                p_bluff = 0.35 if in_early_phase else 0.30
                if self._rng.random() < p_bluff and valid[RAISE] and max_raise >= min_raise:
                    return (RAISE, _clamp(int(pot_ref * 1.5 * noise), min_raise, max_raise), 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                if valid[FOLD]:  return (FOLD,  0, 0, 0)
        else:  # BB
            if is_prem:
                if to_call > 0:
                    if self._rng.random() < (0.80 if in_early_phase else 0.70) and valid[RAISE] and max_raise >= min_raise:
                        return (RAISE, _clamp(int(to_call * reraise_mult * noise), min_raise, max_raise), 0, 0)
                    if valid[CALL]: return (CALL, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            elif equity > eq_thresh_bb:
                if to_call > 0:
                    if self._rng.random() < bb_3bet_p and valid[RAISE] and max_raise >= min_raise:
                        return (RAISE, _clamp(int(to_call * 2.5 * noise), min_raise, max_raise), 0, 0)
                    if valid[CALL]: return (CALL, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            else:
                if to_call > 0:
                    if equity > pot_odds and valid[CALL]: return (CALL, 0, 0, 0)
                    if valid[FOLD]: return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
        if valid[CHECK]: return (CHECK, 0, 0, 0)
        if valid[CALL]:  return (CALL,  0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _agro_postflop(self, street, my_cards, community, opp_discards, my_discards,
                       valid, min_raise, max_raise, my_bet, opp_bet, pot_size,
                       opp_last, p_sims, in_early_phase=False):
        to_call = max(0, opp_bet - my_bet)
        dead    = set()
        if my_discards:  dead |= set(my_discards)
        if opp_discards: dead |= set(opp_discards)
        equity  = self._mc_equity(my_cards, community, dead, num_sims=p_sims) \
                  if len(my_cards) == 2 and len(community) >= 3 else 0.45

        if _board_paired_and_we_weak(my_cards, community):
            equity -= 0.08

        # Apply Alpha-style discard threat
        equity -= _alpha_discard_threat(opp_discards, community)
        equity  = _clamp(equity, 0.0, 0.98)

        cap    = self._cap_score(opp_discards, community, my_discards)
        story  = self._story_score(my_cards, community, my_discards, street)
        backup = self._backup_score(my_cards, community, dead, equity, opp_discards)
        aggro  = 0.35 * cap + 0.25 * story + 0.40 * backup

        bonus, brake = self._hard_triggers(my_cards, community, opp_discards, my_discards)
        if opp_last in ("check", "call") and story >= 0.40 and cap >= 0.30:
            bonus += 0.10
        if backup < 0.20 and (self._betting_history.get("bet_flop") or
                               self._betting_history.get("bet_turn")):
            brake += 0.15
        if street == 3:
            bf = self._betting_history.get("bet_flop", False)
            bt = self._betting_history.get("bet_turn", False)
            if not bf and not bt: brake += 0.20
            elif not bt:          brake += 0.10
        brake = min(brake, 0.45)
        aggro = _clamp(aggro + bonus - brake, 0.0, 1.0)
        aggro = min(1.0, aggro * self._agro_mult)
        if street == 3 and opp_last == "check":
            if (self._betting_history.get("bet_flop") or
                    self._betting_history.get("bet_turn")):
                aggro = min(1.0, aggro + 0.10)
        aggro = _clamp(aggro + self._rng.uniform(-0.05, 0.05), 0.0, 1.0)
        if equity > 0.55:
            aggro = max(aggro, 0.30)

        # ── Semi-bluff band (Alpha edge): equity 0.38-0.52 on flop/turn ──────
        # Freq scaled by fold_rate (not gated): higher folding = more semi-bluffs.
        # Board texture reduces frequency on monotone/paired boards.
        if street in (1, 2):
            fold_rate = self._safe_rate("fold_to_bet")
            # Scale by fold rate — not a hard gate, just a multiplier
            fold_scale    = _clamp(fold_rate / 0.40, 0.5, 1.5)
            # Reduce on textured boards where draws are more likely to be behind
            board_suits_c = Counter(_suit(c) for c in community)
            board_ranks_c = Counter(_rank(c) for c in community)
            is_monotone   = board_suits_c.most_common(1)[0][1] >= 3
            is_paired_brd = board_ranks_c.most_common(1)[0][1] >= 2
            texture_mult  = 0.5 if (is_monotone or is_paired_brd) else 1.0
            early_mult    = 1.2 if in_early_phase else 1.0
            semi_freq     = SEMI_BLUFF_FREQ * fold_scale * texture_mult * early_mult
            semi_freq     = min(0.50, semi_freq)
            if (SEMI_BLUFF_LO <= equity <= SEMI_BLUFF_HI and
                    self._rng.random() < semi_freq):
                if valid[RAISE] and max_raise >= min_raise:
                    frac = self._rng.uniform(0.25, 0.45)
                    amt  = _clamp(int(pot_size * frac), min_raise, max_raise)
                    self._last_equity = equity
                    self._last_aggro  = aggro
                    return (RAISE, amt, 0, 0)

        self._last_equity = equity
        self._last_aggro  = aggro
        return self._aggro_to_action(aggro, equity, street, valid, min_raise,
                                     max_raise, pot_size, to_call, True)

    # ════════════════════════════════════════════════════════════════════════
    # VALUE_GRIND — Equity ladder, no bluffing
    # ════════════════════════════════════════════════════════════════════════

    def _value_preflop(self, my_cards, valid, min_raise, max_raise,
                       my_bet, opp_bet, pot_size, blind_pos):
        to_call      = max(0, opp_bet - my_bet)
        pot_odds     = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        is_prem      = _has_any_premium(my_cards)
        is_prem_pair = _has_premium_pair(my_cards)
        noise        = self._rng.uniform(0.90, 1.10)
        if is_prem:
            if is_prem_pair and to_call >= 15 and valid[RAISE] and max_raise >= min_raise:
                return (RAISE, max_raise, 0, 0)
            if is_prem_pair and self._rng.random() < 0.20:
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            open_size = _clamp(int(10 * noise), min_raise, max_raise)
            if valid[RAISE] and max_raise >= min_raise: return (RAISE, open_size, 0, 0)
            if valid[CALL]:  return (CALL,  0, 0, 0)
            return (CHECK, 0, 0, 0)
        else:
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

    def _value_postflop(self, street, my_cards, community, opp_discards, my_discards,
                        valid, min_raise, max_raise, my_bet, opp_bet, pot_size,
                        blind_pos, p_sims):
        to_call  = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        dead     = set()
        if my_discards:  dead |= set(my_discards)
        if opp_discards: dead |= set(opp_discards)
        equity = self._mc_equity(my_cards, community, dead, num_sims=p_sims) \
                 if len(my_cards) == 2 and len(community) >= 3 else 0.45
        if _board_paired_and_we_weak(my_cards, community):
            equity -= 0.08
        equity -= _alpha_discard_threat(opp_discards, community)
        pos_adj  = 0.04 if blind_pos == 1 else -0.04
        avg_bf   = self._avg_bet_frac()
        size_adj = -0.04 if avg_bf > 0.80 else (0.04 if avg_bf < 0.30 else 0.0)
        if len(opp_discards) >= 3:
            opp_b = _bucket_opp_discard(opp_discards)
            adj   = {"suited_cluster": -0.03, "connected_cluster": -0.02,
                     "high_junk": 0.02, "discarded_pair": 0.03}
            equity = _clamp(equity + adj.get(opp_b, 0.0), 0.0, 0.98)
        pot_ref        = max(pot_size, 1)
        noise          = self._rng.uniform(0.92, 1.08)
        self._last_equity = equity
        bet_thresh_hi  = 0.82 - pos_adj
        bet_thresh_mid = 0.68 - pos_adj
        bet_thresh_lo  = 0.52 - pos_adj
        call_limit     = 0.35 - size_adj
        if street == 3:
            if equity > bet_thresh_hi:
                if valid[RAISE] and max_raise >= min_raise: return (RAISE, max_raise, 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                return (CHECK, 0, 0, 0)
            elif equity > bet_thresh_mid:
                amt = _clamp(int(pot_ref * self._rng.uniform(0.70, 0.90) * noise), min_raise, max_raise)
                if valid[RAISE] and max_raise >= min_raise: return (RAISE, amt, 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                return (CHECK, 0, 0, 0)
            elif equity > pot_odds and to_call <= pot_ref * call_limit:
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            else:
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
        else:
            if equity > bet_thresh_hi:
                amt = _clamp(int(pot_ref * self._rng.uniform(0.65, 0.90) * noise), min_raise, max_raise)
                if valid[RAISE] and max_raise >= min_raise: return (RAISE, amt, 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                return (CHECK, 0, 0, 0)
            elif equity > bet_thresh_mid:
                amt = _clamp(int(pot_ref * self._rng.uniform(0.55, 0.75) * noise), min_raise, max_raise)
                if valid[RAISE] and max_raise >= min_raise: return (RAISE, amt, 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                return (CHECK, 0, 0, 0)
            elif equity > bet_thresh_lo:
                amt = _clamp(int(pot_ref * self._rng.uniform(0.30, 0.48) * noise), min_raise, max_raise)
                if valid[RAISE] and max_raise >= min_raise: return (RAISE, amt, 0, 0)
                if to_call > 0 and equity >= pot_odds and valid[CALL]: return (CALL, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            elif equity >= pot_odds and to_call <= pot_ref * (0.30 - size_adj):
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            else:
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)

    # ════════════════════════════════════════════════════════════════════════
    # TRAP_MODE — Check-call heavy, river value-bet
    # ════════════════════════════════════════════════════════════════════════

    def _trap_preflop(self, my_cards, valid, min_raise, max_raise,
                      my_bet, opp_bet, pot_size, blind_pos):
        to_call  = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        is_prem  = _has_any_premium(my_cards)
        if is_prem:
            if blind_pos == 0 and self._rng.random() < 0.30 and valid[RAISE] and max_raise >= min_raise:
                amt = _clamp(int(pot_size * 2.5), min_raise, max_raise)
                return (RAISE, amt, 0, 0)
            if valid[CALL]:  return (CALL,  0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
        if valid[CHECK]: return (CHECK, 0, 0, 0)
        if to_call > 0 and pot_odds < 0.35 and valid[CALL]: return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _trap_postflop(self, street, my_cards, community, opp_discards, my_discards,
                       valid, min_raise, max_raise, my_bet, opp_bet, pot_size, p_sims):
        to_call  = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        dead     = set()
        if my_discards:  dead |= set(my_discards)
        if opp_discards: dead |= set(opp_discards)
        equity  = self._mc_equity(my_cards, community, dead, num_sims=p_sims) \
                  if len(my_cards) == 2 and len(community) >= 3 else 0.45
        self._last_equity = equity
        pot_ref  = max(pot_size, 1)
        avg_bf   = self._avg_bet_frac()
        call_adj = -0.05 if avg_bf < 0.30 else (0.05 if avg_bf > 0.80 else 0.0)
        if to_call > pot_ref * 0.50 and equity < 0.40:
            if valid[FOLD]:  return (FOLD,  0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
        if street == 3:
            if equity > 0.65 and valid[RAISE] and max_raise >= min_raise:
                amt = _clamp(int(pot_ref * self._rng.uniform(0.60, 0.80)), min_raise, max_raise)
                return (RAISE, amt, 0, 0)
            if equity > pot_odds and to_call <= pot_ref * (0.45 - call_adj):
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)
        else:
            if equity > 0.58 and valid[RAISE] and max_raise >= min_raise:
                amt = _clamp(int(pot_ref * 0.75), min_raise, max_raise)
                return (RAISE, amt, 0, 0)
            if equity > pot_odds and to_call <= pot_ref * 0.40:
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

    # ════════════════════════════════════════════════════════════════════════
    # RIVER — Dedicated river logic (all modes)
    # ════════════════════════════════════════════════════════════════════════

    def _river_action(self, mode, my_cards, community, opp_discards, my_discards,
                      valid, min_raise, max_raise, my_bet, opp_bet, pot_size,
                      blind_pos, opp_last, p_sims):
        """Unified river decision for all modes.
        River is terminal: dead draws fold, calls capped, value bet by hand strength."""
        to_call  = max(0, opp_bet - my_bet)
        pot_ref  = max(pot_size, 1)
        pot_odds = to_call / (pot_ref + to_call) if (pot_ref + to_call) > 0 else 0
        dead     = set()
        if my_discards:  dead |= set(my_discards)
        if opp_discards: dead |= set(opp_discards)

        equity = self._mc_equity(my_cards, community, dead, num_sims=p_sims) \
                 if len(my_cards) == 2 and len(community) >= 3 else 0.45
        if _board_paired_and_we_weak(my_cards, community):
            equity -= 0.08
        equity -= _alpha_discard_threat(opp_discards, community)
        equity  = _clamp(equity, 0.0, 0.98)
        self._last_equity = equity

        hand_cat   = _hand_rank_category(my_cards, community) if len(my_cards) == 2 else "nothing"
        fold_rate  = self._safe_rate("fold_to_bet")
        call_down  = self._safe_rate("call_down")
        is_prem_pair  = _has_premium_pair(my_cards)

        # ── Dead draws: no outs remain on river ──────────────────────────
        if hand_cat in ("flush_draw", "oesd", "gutshot", "nothing"):
            if to_call > 0:
                if valid[FOLD]: return (FOLD, 0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        # ── River call cap (Alpha feature) ────────────────────────────────
        # Never call a large river bet without a strong hand
        if to_call > pot_ref * RIVER_CALL_POT_RATIO_MAX:
            if not is_prem_pair and hand_cat not in ("flush", "full_house"):
                if valid[FOLD]:  return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)

        # ── Value bets by hand category ───────────────────────────────────
        if hand_cat in ("flush", "full_house"):
            if equity > 0.72:
                if valid[RAISE] and max_raise >= min_raise:
                    if equity > 0.85:
                        amt = max_raise
                    else:
                        amt = _clamp(int(pot_ref * self._rng.uniform(0.80, 1.00)),
                                     min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
            if to_call > 0 and equity > pot_odds:
                if valid[CALL]: return (CALL, 0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        if hand_cat == "straight":
            if equity > 0.65:
                if valid[RAISE] and max_raise >= min_raise:
                    amt = _clamp(int(pot_ref * self._rng.uniform(0.70, 0.90)),
                                 min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
            if to_call > 0 and equity > pot_odds and to_call <= pot_ref * RIVER_CALL_POT_RATIO_MAX:
                if valid[CALL]: return (CALL, 0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        if hand_cat == "trips":
            if equity > 0.68:
                if valid[RAISE] and max_raise >= min_raise:
                    amt = _clamp(int(pot_ref * self._rng.uniform(0.60, 0.80)),
                                 min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
            if to_call > 0 and equity > pot_odds and to_call <= pot_ref * RIVER_CALL_POT_RATIO_MAX:
                if valid[CALL]: return (CALL, 0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        if hand_cat == "two_pair":
            if equity > 0.60 and opp_last in ("check", "", None):
                if valid[RAISE] and max_raise >= min_raise:
                    amt = _clamp(int(pot_ref * self._rng.uniform(0.40, 0.60)),
                                 min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
            if to_call > 0 and equity > pot_odds and to_call <= pot_ref * RIVER_CALL_POT_RATIO_MAX:
                if valid[CALL]: return (CALL, 0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        if hand_cat == "one_pair":
            # Thin value: AGRO only, dry board, vs over-folder, opp checked
            if (mode == MODE_AGRO and equity > 0.55 and fold_rate > 0.50 and
                    opp_last in ("check", "", None) and not _board_is_wet(community)):
                if valid[RAISE] and max_raise >= min_raise:
                    amt = _clamp(int(pot_ref * self._rng.uniform(0.25, 0.40)),
                                 min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
            # River bluff: only AGRO, low equity, opp checked, folds enough, not a station
            if (mode == MODE_AGRO and equity < 0.35 and
                    opp_last in ("check", "", None) and call_down < 0.50):
                bluff_freq = 0.15 * _clamp(fold_rate / 0.40, 0.5, 1.5)
                if self._rng.random() < bluff_freq:
                    if valid[RAISE] and max_raise >= min_raise:
                        amt = _clamp(int(pot_ref * 0.50), min_raise, max_raise)
                        return (RAISE, amt, 0, 0)
            # Call only when getting good odds, hand can showdown, and bet not too large
            if to_call > 0 and equity > pot_odds and to_call <= pot_ref * RIVER_CALL_POT_RATIO_MAX:
                # Additional discipline: don't call with marginal one-pair vs non-bluffer
                if fold_rate >= 0.30 or call_down < 0.55:
                    if valid[CALL]: return (CALL, 0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        # Fallback
        if valid[CHECK]: return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    # ════════════════════════════════════════════════════════════════════════
    # Main act / observe
    # ════════════════════════════════════════════════════════════════════════

    def act(self, observation, reward, terminated, truncated, info):
        my_cards     = [c for c in observation["my_cards"]           if c != -1]
        community    = [c for c in observation["community_cards"]     if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards  = [c for c in observation["my_discarded_cards"]  if c != -1]
        valid        = observation["valid_actions"]
        street       = observation["street"]
        min_raise    = observation["min_raise"]
        max_raise    = observation["max_raise"]
        my_bet       = observation["my_bet"]
        opp_bet      = observation["opp_bet"]
        pot_size     = observation.get("pot_size",        my_bet + opp_bet)
        blind_pos    = observation.get("blind_position",  0)
        time_left    = observation.get("time_left",       600.0)

        opp_last = _normalize_action(observation.get("opp_last_action", ""))

        prev_was_bet = self._last_was_bet
        prev_street  = self._last_street

        # ── New hand detection ─────────────────────────────────────────────
        is_new_hand = (street == 0 and my_bet <= 2 and opp_bet <= 2)
        if is_new_hand:
            self.hand_number += 1
            self._hand_mode, self._agro_mult = self._select_mode()
            self._hand_override    = None
            self._betting_history  = {"bet_flop": False, "bet_turn": False}
            self._last_was_bet     = False
            self._last_street      = 0
            self._opp_folded       = False
            self._we_folded        = False
            mode_change = (
                f" [WAS {self._prev_mode}]" if self._hand_mode != self._prev_mode else ""
            )
            self.logger.info(
                f"Hand {self.hand_number}: mode={self._hand_mode}{mode_change}"
                f"[{self._mode_reason}], mult={self._agro_mult:.1f}, "
                f"pnl={self._running_pnl:+d}, "
                f"aggro={self._fmt_stat('opp_aggression')}, "
                f"fb={self._fmt_stat('fold_to_bet')}, "
                f"cd={self._fmt_stat('call_down')}"
            )
            self._prev_mode = self._hand_mode

            if self.hand_number % 50 == 0:
                parts = []
                for k, (num, den) in self._stats.items():
                    if k == "opp_avg_bet_frac":
                        val = f"{num/den:.2f}" if den > 0 else "prior"
                        parts.append(f"bet_frac={val}({den})")
                    else:
                        parts.append(f"{k}={num}/{den}")
                self.logger.info(f"[STATS H{self.hand_number}] " + " | ".join(parts))

        if opp_last and not is_new_hand:
            self._process_opponent_action(observation, opp_last, prev_was_bet, prev_street)
        elif is_new_hand and opp_last:
            self._process_opponent_action(observation, opp_last, prev_was_bet, prev_street)

        self._prev_opp_bet = opp_bet
        self._prev_my_bet  = my_bet

        d_sims, p_sims = self._sim_budget(time_left)

        # ── Bleed-out lock (skip during discard — must complete it) ───────
        # DISCARD is mandatory when valid[DISCARD] is True; we cannot fold instead.
        # The guard below is intentional: only apply lock on betting actions.
        if not valid[DISCARD]:
            lock = self._bleedout_check(valid)
            if lock is not None:
                self._last_was_bet = False
                self._last_street  = street
                return lock

        mode           = self._hand_override or self._hand_mode
        in_early_phase = (self.hand_number <= EARLY_PHASE_HANDS)

        # ── Discard ───────────────────────────────────────────────────────
        if valid[DISCARD]:
            i, j = self._choose_keep(my_cards, community, opp_discards, d_sims, mode)
            self._last_was_bet = False
            self._last_street  = street
            return (DISCARD, 0, i, j)

        # ── Preflop ───────────────────────────────────────────────────────
        if street == 0:
            if mode == MODE_VALUE:
                result = self._value_preflop(my_cards, valid, min_raise, max_raise,
                                             my_bet, opp_bet, pot_size, blind_pos)
            elif mode == MODE_TRAP:
                result = self._trap_preflop(my_cards, valid, min_raise, max_raise,
                                            my_bet, opp_bet, pot_size, blind_pos)
            elif mode == MODE_GTO:
                result = self._gto_action(street, my_cards, community, opp_discards,
                                          my_discards, valid, min_raise, max_raise,
                                          my_bet, opp_bet, pot_size, blind_pos, p_sims)
            else:  # AGRO
                result = self._agro_preflop(my_cards, valid, min_raise, max_raise,
                                            my_bet, opp_bet, pot_size, blind_pos, p_sims,
                                            in_early_phase=in_early_phase)
            if result[0] == FOLD:
                self._we_folded = True
            self._last_was_bet = (result[0] == RAISE)
            self._last_street  = street
            return result

        # ── Post-flop ─────────────────────────────────────────────────────
        # River gets dedicated logic that handles all modes internally
        if street == 3 and len(my_cards) == 2 and len(community) >= 3:
            result = self._river_action(mode, my_cards, community, opp_discards,
                                        my_discards, valid, min_raise, max_raise,
                                        my_bet, opp_bet, pot_size, blind_pos,
                                        opp_last, p_sims)
        elif mode == MODE_VALUE:
            result = self._value_postflop(street, my_cards, community, opp_discards,
                                          my_discards, valid, min_raise, max_raise,
                                          my_bet, opp_bet, pot_size, blind_pos, p_sims)
        elif mode == MODE_TRAP:
            result = self._trap_postflop(street, my_cards, community, opp_discards,
                                         my_discards, valid, min_raise, max_raise,
                                         my_bet, opp_bet, pot_size, p_sims)
        elif mode == MODE_GTO:
            result = self._gto_action(street, my_cards, community, opp_discards,
                                      my_discards, valid, min_raise, max_raise,
                                      my_bet, opp_bet, pot_size, blind_pos, p_sims)
        else:  # AGRO
            result = self._agro_postflop(street, my_cards, community, opp_discards,
                                         my_discards, valid, min_raise, max_raise,
                                         my_bet, opp_bet, pot_size, opp_last, p_sims,
                                         in_early_phase=in_early_phase)

        # ── Softened one-pair cap (flop/turn only — river handled above) ──
        # METAV5 hard-blocked all raises with one pair. Here we allow raises
        # when equity is strong, opponent over-folds, and board is dry.
        # Also fold dead draws facing bets on turn.
        if street != 3 and len(my_cards) == 2 and len(community) >= 3:
            hand_cat    = _hand_rank_category(my_cards, community)
            to_call_now = max(0, opp_bet - my_bet)
            pot_ref_now = max(pot_size, 1)

            if hand_cat == "one_pair":
                fold_to_bet = self._safe_rate("fold_to_bet")
                opp_aggro_r = self._safe_rate("opp_aggression")
                # Allow raise with strong one-pair when conditions favor it
                can_raise = (
                    self._last_equity >= ONE_PAIR_RAISE_EQ and
                    fold_to_bet >= ONE_PAIR_RAISE_FOLD and
                    opp_aggro_r < ONE_PAIR_RAISE_AGGRO and
                    not _board_is_wet(community)
                )
                if result[0] == RAISE and not can_raise:
                    if to_call_now > 0 and valid[CALL]:
                        result = (CALL, 0, 0, 0)
                    elif valid[CHECK]:
                        result = (CHECK, 0, 0, 0)
                # Raised fold threshold vs METAV5's hard 0.40
                if result[0] == CALL and to_call_now > pot_ref_now * ONE_PAIR_FOLD_PRESS:
                    if valid[FOLD]:
                        result = (FOLD, 0, 0, 0)

            elif hand_cat in ("nothing", "flush_draw", "oesd", "gutshot"):
                # Dead draws on turn (street 2) facing a bet: fold
                if to_call_now > 0:
                    if valid[FOLD]:
                        result = (FOLD, 0, 0, 0)
                elif valid[CHECK]:
                    result = (CHECK, 0, 0, 0)

        if result[0] == RAISE:
            if street == 1:   self._betting_history["bet_flop"] = True
            elif street == 2: self._betting_history["bet_turn"] = True

        if result[0] == FOLD:
            self._we_folded = True

        if street >= 1 and result[0] in (RAISE, FOLD):
            act_name = "RAISE" if result[0] == RAISE else "FOLD"
            agro_tag = f" agg={self._last_aggro:.2f}" if mode == MODE_AGRO else ""
            self.logger.info(
                f"  [{act_name} H{self.hand_number} S{street} {mode}] "
                f"eq={self._last_equity:.2f}{agro_tag}"
                + (f" amt={result[1]}" if result[0] == RAISE else "")
            )

        self._last_was_bet = (result[0] == RAISE)
        self._last_street  = street
        return result

    def observe(self, observation, reward, terminated, truncated, info):
        raw = observation.get("opp_last_action", "")
        opp_action = _normalize_action(raw)

        if opp_action == "fold":
            self._opp_folded = True

        if terminated:
            self._running_pnl += int(reward)
            # Terminal fold: opponent folded to our bet — update fold stats here
            if self._opp_folded and self._last_was_bet:
                self._stats["fold_to_bet"][0]  += 1
                self._stats["fold_to_bet"][1]  += 1
                self._stats["fold_to_raise"][0] += 1
                self._stats["fold_to_raise"][1] += 1
