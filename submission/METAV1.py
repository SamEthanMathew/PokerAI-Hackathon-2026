# MetaBot: Adaptive Decision-Tree Poker Agent
#
# Routes between four strategy modes based on live opponent profiling:
#   GTO_DEFAULT  — Libratus policy tables  (unknown villain, first ~8 hands)
#   AGRO_EXPLOIT — Cap/Story/Backup scoring (Overfolders and default)
#   VALUE_GRIND  — Equity ladder, no bluffs (Calling Stations)
#   TRAP_MODE    — Check-call heavy        (Maniacs and Trappers)
#
# Decision tree (per hand, priority order):
#   total_obs < 8            -> GTO_DEFAULT
#   opp_aggression > 0.55    -> TRAP_MODE    (Maniac)
#   fold_flop > 0.50         -> AGRO x 1.4  (Overfolder)
#   call_down > 0.60         -> VALUE_GRIND  (Station)
#   check_raise > 0.15       -> TRAP_MODE    (Trapper)
#   showdown_win > 0.65      -> VALUE_GRIND  (HonestStrong)
#   default                  -> AGRO x 1.0

import random
from collections import Counter
from itertools import combinations

from agents.agent import Agent
from gym_env import PokerEnv

try:
    from submission.libratus_tables import POLICY, KEEP_EQUITY, POSTERIOR, MATCHUPS
except ImportError:
    from libratus_tables import POLICY, KEEP_EQUITY, POSTERIOR, MATCHUPS

# ── Constants ─────────────────────────────────────────────────────────────────

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

# ── Card helpers ──────────────────────────────────────────────────────────────

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
    """Longest run of consecutive ranks in a list."""
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

# ── Libratus bucket helpers ────────────────────────────────────────────────────

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

# ── PlayerAgent ───────────────────────────────────────────────────────────────

class PlayerAgent(Agent):

    _STAT_PRIORS = {
        "fold_flop_bet":    0.35,
        "fold_turn_bet":    0.35,
        "fold_river_bet":   0.35,
        "fold_to_raise":    0.35,
        "check_raise":      0.05,
        "call_down":        0.40,
        "showdown_win":     0.50,
        "suit_attack_fold": 0.40,
        "opp_aggression":   0.25,   # raised from 0.20 — triggers TRAP sooner
        "opp_avg_bet_frac": 0.50,   # prior: medium bet sizing
        "opp_af_flop":      1.0,    # AF = (bets+raises)/calls; neutral start
        "opp_af_turn":      1.0,
        "opp_af_river":     1.0,
        "opp_3bet_freq":    0.08,   # 3-bets are rare
        "opp_preflop_raise": 0.30,  # how often villain raises preflop
        "opp_river_call_freq": 0.45, # how often villain calls a river bet
    }

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self._env      = PokerEnv()
        self.evaluator = self._env.evaluator
        self.hand_number = 0
        self._rng        = random.Random(42)

        self._stats = {
            "fold_flop_bet":    [0, 0],
            "fold_turn_bet":    [0, 0],
            "fold_river_bet":   [0, 0],
            "fold_to_raise":    [0, 0],
            "check_raise":      [0, 0],
            "call_down":        [0, 0],
            "showdown_win":     [0, 0],
            "suit_attack_fold": [0, 0],
            "opp_aggression":   [0, 0],
            "opp_avg_bet_frac": [0.0, 0],  # [sum_of_fracs, count]
            "opp_af_flop":      [0, 0],    # [bets+raises, calls] on flop
            "opp_af_turn":      [0, 0],    # [bets+raises, calls] on turn
            "opp_af_river":     [0, 0],    # [bets+raises, calls] on river
            "opp_3bet_freq":    [0, 0],    # [re-raises into our bet, times faced our bet]
            "opp_preflop_raise": [0, 0],   # [preflop raises, total preflop actions]
            "opp_river_call_freq": [0, 0], # [river calls vs our bet, times we bet river]
        }

        # Per-hand state
        self._hand_mode       = MODE_GTO
        self._hand_override   = None
        self._agro_mult       = 1.0
        self._betting_history = {"bet_flop": False, "bet_turn": False}
        self._last_was_bet    = False
        self._last_street     = 0
        self._last_suit_attack = False
        self._opp_folded       = False
        self._we_folded        = False

    def __name__(self):
        return "MetaBot"

    # ── RNG ───────────────────────────────────────────────────────────────────

    def _seed_rng(self, my_cards, street):
        seed = hash((self.hand_number, street, tuple(sorted(my_cards))))
        self._rng = random.Random(seed)

    # ── Time-aware sim budget ─────────────────────────────────────────────────

    def _sim_budget(self, time_left):
        """Returns (discard_sims_per_combo, postflop_sims)."""
        if time_left > 300:   return 200, 400
        elif time_left > 150: return 120, 250
        elif time_left > 60:  return 60,  150
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
        """Average of top-3 2-card keep equities from a 5-card preflop hand."""
        sims = max(p_sims // 10, 15)
        scores = []
        for i, j in combinations(range(len(my5)), 2):
            keep = [my5[i], my5[j]]
            toss = set(my5[k] for k in range(len(my5)) if k not in (i, j))
            scores.append(self._mc_equity(keep, [], toss, num_sims=sims))
        scores.sort(reverse=True)
        top = scores[:3]
        return sum(top) / len(top) if top else 0.45

    # ── Stat helpers ──────────────────────────────────────────────────────────

    def _safe_rate(self, key):
        folds, total = self._stats.get(key, [0, 0])
        if total < 5:
            return self._STAT_PRIORS.get(key, 0.35)
        return folds / total

    def _avg_bet_frac(self):
        """Average opponent bet size as fraction of pot when they raise."""
        total_sum, count = self._stats["opp_avg_bet_frac"]
        if count < 5:
            return self._STAT_PRIORS["opp_avg_bet_frac"]
        return total_sum / count

    def _opp_af(self, street=None):
        """
        Aggression Factor = (bets+raises) / calls per street or overall.
        >5 = maniac (trap them), <1.2 = passive (bluff freely).
        Falls back to prior 1.5 until 3+ calls observed.
        """
        key_map = {1: "opp_af_flop", 2: "opp_af_turn", 3: "opp_af_river"}
        if street in key_map:
            bets, calls = self._stats[key_map[street]]
        else:
            bets  = sum(self._stats[k][0] for k in key_map.values())
            calls = sum(self._stats[k][1] for k in key_map.values())
        if calls < 3:
            return 1.5   # prior: slightly aggressive is normal
        return bets / calls if calls > 0 else 1.5

    def _total_obs(self):
        return self.hand_number

    # ── Villain classifier ────────────────────────────────────────────────────

    def _select_mode(self):
        """Returns (mode, agro_mult). Called once per hand at hand start."""
        if self._total_obs() < 5:
            return MODE_GTO, 1.0

        # Ramp factor: smoothly transition from 0→1 over hands 5–12
        # Prevents abrupt style shifts that a human/adaptive opponent can exploit
        ramp = min(1.0, (self.hand_number - 5) / 7.0)

        fold_flop  = self._safe_rate("fold_flop_bet")
        fold_turn  = self._safe_rate("fold_turn_bet")
        fold_river = self._safe_rate("fold_river_bet")
        cr_rate    = self._safe_rate("check_raise")
        call_down  = self._safe_rate("call_down")
        opp_aggro  = self._safe_rate("opp_aggression")
        showdown_w = self._safe_rate("showdown_win")

        overall_af = self._opp_af()
        opp_3b     = self._safe_rate("opp_3bet_freq")
        pf_raise   = self._safe_rate("opp_preflop_raise")

        # Maniac: high raise-frequency OR high overall AF → TRAP and charge them
        if opp_aggro > 0.55 or overall_af > 5.0:
            return MODE_TRAP, 1.0
        # 3-bet heavy: strong ranges, stop bluffing into them
        if opp_3b > 0.20:
            return MODE_TRAP, 1.0
        # Overfolder: exploit by raising frequently
        if fold_flop > 0.50 or fold_turn > 0.55 or fold_river > 0.60:
            return MODE_AGRO, 1.0 + 0.4 * ramp   # ramps 1.0 → 1.4 over first 7 exploit hands
        # Calling station: pure equity, no bluffs
        if fold_flop < 0.25 or call_down > 0.60:
            return MODE_VALUE, 1.0
        # Check-raiser: slow down, trap them back
        if cr_rate > 0.15:
            return MODE_TRAP, 1.0
        # HonestStrong: hands up at showdown tell us they play honestly → value exploit
        if showdown_w > 0.65:
            return MODE_VALUE, 1.0
        # Passive overall AF: villain rarely raises or bets → bluff freely
        if overall_af < 1.2:
            return MODE_AGRO, 1.0 + 0.3 * ramp
        # Loose preflop raiser: plays wide, susceptible to postflop pressure
        if pf_raise > 0.55:
            return MODE_AGRO, 1.0 + 0.2 * ramp
        # Very passive preflop (limper): lean VALUE since they likely have decent hands
        if pf_raise < 0.10:
            return MODE_VALUE, 1.0
        return MODE_AGRO, 1.0

    # =========================================================================
    # DISCARD — Always AgroMonkey KeepScore
    # =========================================================================

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
        # VALUE/TRAP: weight equity and made hands heavily; bluff-structure components are useless
        if mode in (MODE_VALUE, MODE_TRAP):
            return 3.5*eq + 0.6*pp + 1.7*dd + 2.0*mhv + 0.8*bv + 0.2*sf
        return 2.8*eq + 2.2*pp + 1.7*dd + 1.0*mhv + 0.8*bv + 0.6*sf

    def _choose_keep(self, my_cards, community, opp_discards, d_sims, mode=MODE_AGRO):
        best_score = -999.0
        best_ij    = (0, 1)
        candidates = []
        for i, j in combinations(range(len(my_cards)), 2):
            keep      = [my_cards[i], my_cards[j]]
            toss      = [my_cards[k] for k in range(len(my_cards)) if k not in (i, j)]
            toss_dead = set(toss) | (set(opp_discards) if opp_discards else set())
            sc        = self._keep_score(keep, community, opp_discards, toss_dead, d_sims, mode)
            candidates.append((i, j, sc))
            if sc > best_score:
                best_score = sc
                best_ij    = (i, j)
        ties = [(i, j) for i, j, sc in candidates if best_score - sc < 0.03]
        if len(ties) > 1:
            best_ij = self._rng.choice(ties)
        return best_ij

    # =========================================================================
    # GTO_DEFAULT — Libratus policy table + range-vs-range equity
    # =========================================================================

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
        # Board-paired penalty
        if len(community) >= 3 and len(cards2) == 2:
            b_ranks = [_rank(c) for c in community]
            rc      = Counter(b_ranks)
            bp_rank = next((r for r, cnt in rc.items() if cnt >= 2), None)
            if bp_rank is not None:
                my_ranks = [_rank(c) for c in cards2]
                if bp_rank not in my_ranks and not (
                    my_ranks[0] == my_ranks[1] and my_ranks[0] in (RANK_A, RANK_9)
                ):
                    equity -= 0.08
        # Opponent discard bucket adjustment
        if len(opp_discards) >= 3:
            opp_b = _bucket_opp_discard(opp_discards)
            adj   = {"low_junk": -0.04, "suited_cluster": -0.03,
                     "connected_cluster": -0.02, "high_junk": 0.02, "discarded_pair": 0.03}
            equity += adj.get(opp_b, 0.0)
        return _clamp(equity, 0.0, 0.98)

    def _gto_action(self, street, my_cards, community, opp_discards, my_discards,
                    valid, min_raise, max_raise, my_bet, opp_bet, pot_size,
                    blind_pos, p_sims):
        to_call  = max(0, opp_bet - my_bet)
        position = "sb" if blind_pos == 0 else "bb"
        equity   = self._gto_equity(my_cards, community, opp_discards, my_discards, p_sims)
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

        # Opponent-model nudges
        fold_r = self._safe_rate("fold_to_raise")
        agg_r  = self._safe_rate("opp_aggression")
        if fold_r > 0.50:
            policy["small_bet"]  = policy.get("small_bet", 0) + 0.05
            policy["fold"]       = max(0.0, policy.get("fold", 0) - 0.05)
        elif fold_r < 0.20:
            policy["small_bet"]  = max(0.0, policy.get("small_bet", 0) - 0.04)
            policy["check_call"] = policy.get("check_call", 0) + 0.04
        if agg_r > 0.55 and strength in ("marginal", "weak"):
            policy["check_call"] = max(0.0, policy.get("check_call", 0) - 0.05)
            policy["fold"]       = policy.get("fold", 0) + 0.05

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

    # =========================================================================
    # AGRO_EXPLOIT — Cap / Story / Backup scoring
    # =========================================================================

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
        if street == 3 and self._betting_history.get("bet_turn"):   score += 0.10
        # Story is broken if villain raised us this hand — penalize bluff narrative
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

        # Trigger E: strong draw + villain folds often
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
                if self._safe_rate("fold_flop_bet") > 0.45:
                    bonus += 0.12

        # Scale suit-attack bonus by historical success
        sa_rate = self._safe_rate("suit_attack_fold")
        if bonus > 0.0:
            if sa_rate > 0.60:   bonus = min(1.0, bonus * 1.20)
            elif sa_rate < 0.25: bonus *= 0.70

        # Brake A: we discarded the dominant board suit
        if community and my_discards:
            b_sc2 = Counter(_suit(c) for c in community)
            if b_sc2:
                dom_s2, dom_c2 = b_sc2.most_common(1)[0]
                if dom_c2 >= 2 and my_suit_d.get(dom_s2, 0) >= 2:
                    brake += 0.30
        if self._safe_rate("fold_flop_bet") < 0.20: brake += 0.20  # Brake B
        if self._safe_rate("check_raise")   > 0.15: brake += 0.15  # Brake C
        return bonus, brake

    def _aggro_to_action(self, aggro, equity, street, valid, min_raise,
                         max_raise, pot_size, to_call, bluff_enable):
        pot_ref  = max(pot_size, 1)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        eff      = aggro if bluff_enable else aggro * (0.5 + 0.5 * equity)

        if street == 3:  # River: polarized
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
                    if equity > pot_odds and to_call <= pot_ref * 0.25 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[FOLD]: return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            else:
                if to_call > 0:
                    if equity > pot_odds and to_call <= pot_ref * 0.35 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[FOLD]: return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
        else:  # Flop / Turn
            if eff > 0.70:
                if valid[RAISE] and max_raise >= min_raise:
                    frac = self._rng.uniform(0.70, 1.10) if street == 2 \
                           else self._rng.uniform(0.60, 0.85)
                    return (RAISE, _clamp(int(pot_ref * frac), min_raise, max_raise), 0, 0)
                # Can't raise — next best is call, not check
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
                    if equity > pot_odds and valid[CALL]: return (CALL, 0, 0, 0)
                    if valid[FOLD]: return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            else:
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                if valid[FOLD]:  return (FOLD,  0, 0, 0)

        if valid[CHECK]:  return (CHECK, 0, 0, 0)
        if to_call > 0 and equity > pot_odds and valid[CALL]: return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _agro_preflop(self, my_cards, valid, min_raise, max_raise,
                      my_bet, opp_bet, pot_size, blind_pos, p_sims):
        if len(my_cards) > 2:
            equity  = self._preflop_equity(my_cards, p_sims)
            is_prem = equity > 0.65 or any(
                _rank(my_cards[a]) == _rank(my_cards[b]) and _rank(my_cards[a]) >= RANK_8
                for a in range(len(my_cards)) for b in range(a + 1, len(my_cards))
            )
        else:
            equity  = self._mc_equity(my_cards, [], set(), num_sims=max(p_sims // 3, 30))
            r0, r1  = _rank(my_cards[0]), _rank(my_cards[1])
            is_prem = equity > 0.65 or (r0 == r1 and r0 >= RANK_8)
        to_call  = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        noise    = self._rng.uniform(0.90, 1.10)

        pot_ref_pf = max(pot_size, 1)
        if blind_pos == 0:  # SB
            if is_prem:
                if self._rng.random() < 0.15:
                    if valid[CALL]:  return (CALL,  0, 0, 0)
                    if valid[CHECK]: return (CHECK, 0, 0, 0)
                if valid[RAISE] and max_raise >= min_raise:
                    return (RAISE, _clamp(int(pot_ref_pf * 2.5 * noise), min_raise, max_raise), 0, 0)
            elif equity > 0.45:
                if self._rng.random() < 0.65 and valid[RAISE] and max_raise >= min_raise:
                    return (RAISE, _clamp(int(pot_ref_pf * 2.0 * noise), min_raise, max_raise), 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            else:
                if self._rng.random() < 0.30 and valid[RAISE] and max_raise >= min_raise:
                    return (RAISE, _clamp(int(pot_ref_pf * 1.5 * noise), min_raise, max_raise), 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                if valid[FOLD]:  return (FOLD,  0, 0, 0)
        else:  # BB
            if is_prem:
                if to_call > 0:
                    if self._rng.random() < 0.70 and valid[RAISE] and max_raise >= min_raise:
                        return (RAISE, _clamp(int(to_call * 3 * noise), min_raise, max_raise), 0, 0)
                    if valid[CALL]: return (CALL, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            elif equity > 0.40:
                if to_call > 0:
                    if self._rng.random() < 0.15 and valid[RAISE] and max_raise >= min_raise:
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
                       opp_last, p_sims):
        to_call = max(0, opp_bet - my_bet)
        dead    = set()
        if my_discards:  dead |= set(my_discards)
        if opp_discards: dead |= set(opp_discards)
        equity  = self._mc_equity(my_cards, community, dead, num_sims=p_sims) \
                  if len(my_cards) == 2 and len(community) >= 3 else 0.45

        cap    = self._cap_score(opp_discards, community, my_discards)
        story  = self._story_score(my_cards, community, my_discards, street)
        backup = self._backup_score(my_cards, community, dead, equity, opp_discards)
        aggro  = 0.45 * cap + 0.35 * story + 0.20 * backup

        bonus, brake = self._hard_triggers(my_cards, community, opp_discards, my_discards)
        if opp_last in ("check", "call") and story >= 0.40 and cap >= 0.30:
            bonus += 0.10
        self._last_suit_attack = (bonus >= 0.12)

        # Brake D: very low backup on multi-street bluff
        if backup < 0.20 and (self._betting_history.get("bet_flop") or
                               self._betting_history.get("bet_turn")):
            brake += 0.15
        # Brake E: incoherent line on river
        if street == 3:
            bf = self._betting_history.get("bet_flop", False)
            bt = self._betting_history.get("bet_turn", False)
            if not bf and not bt: brake += 0.20
            elif not bt:          brake += 0.10

        # AF-based street adjustments: use observed villain tendencies per street
        flop_af  = self._opp_af(street=1)
        river_af = self._opp_af(street=3)
        riv_call = self._safe_rate("opp_river_call_freq")

        # Flop: if villain rarely fights back (AF < 1.0), fire c-bets more aggressively
        if street == 1 and flop_af < 1.0:
            bonus += 0.08

        # River: adjust based on villain's river call rate
        if street == 3:
            if riv_call > 0.60:
                brake += 0.20   # villain calls rivers → suppress bluffs, value only
            elif riv_call < 0.25:
                aggro = min(1.0, aggro + 0.15)   # villain folds rivers → fire more

        aggro = _clamp(aggro + bonus - brake, 0.0, 1.0)
        aggro = min(1.0, aggro * self._agro_mult)

        # River bluff boost — villain checked into us with a coherent story
        if street == 3 and opp_last == "check":
            if (self._betting_history.get("bet_flop") or
                    self._betting_history.get("bet_turn")):
                aggro = min(1.0, aggro + 0.10)

        aggro = _clamp(aggro + self._rng.uniform(-0.05, 0.05), 0.0, 1.0)

        return self._aggro_to_action(aggro, equity, street, valid, min_raise,
                                     max_raise, pot_size, to_call, True)

    # =========================================================================
    # VALUE_GRIND — Equity ladder, no bluffing
    # =========================================================================

    def _value_preflop(self, my_cards, valid, min_raise, max_raise,
                       my_bet, opp_bet, pot_size, blind_pos):
        to_call  = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        PREM_PAIRS  = {frozenset([8, 8]), frozenset([7, 7]), frozenset([6, 6])}
        PREM_ANY    = {frozenset([8, 7]), frozenset([8, 6])}
        PREM_SUITED = {frozenset([7, 6]), frozenset([6, 5]), frozenset([5, 4]), frozenset([5, 7])}
        is_prem_pair = False
        is_prem      = False
        for a in range(len(my_cards)):
            for b in range(a + 1, len(my_cards)):
                ra, rb = _rank(my_cards[a]), _rank(my_cards[b])
                sa, sb = _suit(my_cards[a]), _suit(my_cards[b])
                ranks  = frozenset([ra, rb])
                if ra == rb and frozenset([ra, ra]) in PREM_PAIRS:
                    is_prem_pair = True
                    is_prem      = True
                elif ranks in PREM_ANY or (sa == sb and ranks in PREM_SUITED):
                    is_prem = True
        noise = self._rng.uniform(0.90, 1.10)
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
            if to_call > 0 and pot_odds < 0.28 and valid[CALL]: return (CALL, 0, 0, 0)
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

        # Board-paired penalty (standardized to -0.08, same as GTO mode)
        if len(community) >= 3:
            b_ranks = [_rank(c) for c in community]
            rc      = Counter(b_ranks)
            bp_rank = next((r for r, cnt in rc.items() if cnt >= 2), None)
            if bp_rank is not None:
                my_ranks = [_rank(c) for c in my_cards]
                if bp_rank not in my_ranks:
                    equity -= 0.08

        # Position factor: IP (blind_pos==1, dealer) bets wider; OOP tightens slightly
        pos_adj = 0.04 if blind_pos == 1 else -0.04

        # Opponent bet sizing: fold more vs large bettors, call wider vs min-raisers
        avg_bf = self._avg_bet_frac()
        size_adj = -0.04 if avg_bf > 0.80 else (0.04 if avg_bf < 0.30 else 0.0)

        # Opponent discard adjustment
        if len(opp_discards) >= 3:
            opp_b = _bucket_opp_discard(opp_discards)
            adj   = {"suited_cluster": -0.03, "connected_cluster": -0.02,
                     "high_junk": 0.02, "discarded_pair": 0.03}
            equity = _clamp(equity + adj.get(opp_b, 0.0), 0.0, 0.98)

        pot_ref = max(pot_size, 1)
        noise   = self._rng.uniform(0.92, 1.08)

        # Adjusted thresholds incorporating position and opponent bet sizing
        bet_thresh_hi  = 0.82 - pos_adj          # IP bets at lower equity (0.78), OOP higher (0.86)
        bet_thresh_mid = 0.68 - pos_adj          # Same shift for medium bets
        bet_thresh_lo  = 0.52 - pos_adj          # And for thin bets
        call_limit     = 0.35 - size_adj         # Call wider vs min-raisers, tighter vs pot-bettors

        if street == 3:  # River
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
        else:  # Flop / Turn
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

    # =========================================================================
    # TRAP_MODE — Check-call heavy, river value-bet
    # =========================================================================

    def _trap_preflop(self, my_cards, valid, min_raise, max_raise,
                      my_bet, opp_bet, pot_size, blind_pos):
        to_call  = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        # Scan all card pairs — don't miss premiums in positions 2-4
        is_prem = any(
            _rank(my_cards[a]) == _rank(my_cards[b]) and _rank(my_cards[a]) >= RANK_8
            for a in range(len(my_cards)) for b in range(a + 1, len(my_cards))
        )
        if is_prem:
            # SB with premiums: occasionally raise to build pot (30%), otherwise slow-play
            if blind_pos == 0 and self._rng.random() < 0.30 and valid[RAISE] and max_raise >= min_raise:
                amt = _clamp(int(pot_size * 2.5), min_raise, max_raise)
                return (RAISE, amt, 0, 0)
            # BB or SB slow-play: call/check to keep villain in and trap postflop
            if valid[CALL]:  return (CALL,  0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
        # Medium hands: call reasonable prices to reach the flop where TRAP mode operates
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
        pot_ref = max(pot_size, 1)

        # Adjust call-down threshold based on opponent bet sizing:
        # Min-raisers (avg_bf < 0.30) have weaker ranges — call wider;
        # Pot-bettors (avg_bf > 0.80) have stronger ranges — fold more.
        avg_bf = self._avg_bet_frac()
        call_adj = -0.05 if avg_bf < 0.30 else (0.05 if avg_bf > 0.80 else 0.0)

        # Fold quickly when weak and facing pressure
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
            # Semi-bluff raise with strong equity to charge villain's draws
            if equity > 0.58 and valid[RAISE] and max_raise >= min_raise:
                amt = _clamp(int(pot_ref * 0.75), min_raise, max_raise)
                return (RAISE, amt, 0, 0)
            if equity > pot_odds and to_call <= pot_ref * 0.40:
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

    # =========================================================================
    # Main act / observe
    # =========================================================================

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
        opp_last     = observation.get("opp_last_action", "")
        time_left    = observation.get("time_left",       400.0)

        # ── New hand detection ────────────────────────────────────────────────
        if street == 0 and my_bet <= 2 and opp_bet <= 2:
            self.hand_number += 1
            self._hand_mode, self._agro_mult = self._select_mode()
            self._hand_override    = None
            self._betting_history  = {"bet_flop": False, "bet_turn": False}
            self._last_was_bet     = False
            self._last_suit_attack = False
            self._opp_folded       = False
            self._we_folded        = False
            self.logger.info(
                f"Hand {self.hand_number}: mode={self._hand_mode}, "
                f"mult={self._agro_mult:.1f}, obs={self._total_obs()}"
            )

        self._seed_rng(my_cards, street)
        d_sims, p_sims = self._sim_budget(time_left)

        # ── Resolve effective mode ────────────────────────────────────────────
        mode = self._hand_override or self._hand_mode

        # ── Discard (mode-aware KeepScore) ────────────────────────────────────
        if valid[DISCARD]:
            i, j = self._choose_keep(my_cards, community, opp_discards, d_sims, mode)
            self._last_was_bet = False
            self._last_street  = street
            return (DISCARD, 0, i, j)

        # ── Preflop ──────────────────────────────────────────────────────────
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
                                            my_bet, opp_bet, pot_size, blind_pos, p_sims)
            if result[0] == FOLD:
                self._we_folded = True
            self._last_was_bet = (result[0] == RAISE)
            self._last_street  = street
            return result

        # ── Post-flop ─────────────────────────────────────────────────────────
        if mode == MODE_VALUE:
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
                                         my_bet, opp_bet, pot_size, opp_last, p_sims)

        # ── Track betting history ─────────────────────────────────────────────
        if result[0] == RAISE:
            if street == 1:   self._betting_history["bet_flop"] = True
            elif street == 2: self._betting_history["bet_turn"] = True

        if result[0] == FOLD:
            self._we_folded = True
        self._last_was_bet = (result[0] == RAISE)
        self._last_street  = street
        return result

    def observe(self, observation, reward, terminated, truncated, info):
        opp_action = observation.get("opp_last_action", "")

        if opp_action == "fold":
            self._opp_folded = True

        # Showdown tracking: only count genuine showdowns (no folds this hand)
        if terminated and not self._we_folded and not self._opp_folded:
            self._stats["showdown_win"][1] += 1
            if reward < 0:
                self._stats["showdown_win"][0] += 1

        if not opp_action:
            return

        street_map = {1: "fold_flop_bet", 2: "fold_turn_bet", 3: "fold_river_bet"}
        street_key = street_map.get(self._last_street)

        if self._last_was_bet:
            if opp_action == "fold":
                if street_key:
                    self._stats[street_key][0] += 1
                    self._stats[street_key][1] += 1
                self._stats["fold_to_raise"][0] += 1
                self._stats["fold_to_raise"][1] += 1
                if self._last_suit_attack:
                    self._stats["suit_attack_fold"][0] += 1
                    self._stats["suit_attack_fold"][1] += 1
            elif opp_action in ("call", "check", "raise"):
                if street_key:
                    self._stats[street_key][1] += 1
                self._stats["fold_to_raise"][1] += 1
                if self._last_suit_attack:
                    self._stats["suit_attack_fold"][1] += 1
                if opp_action == "call" and self._last_street >= 1:
                    self._stats["call_down"][0] += 1
                    self._stats["call_down"][1] += 1
                elif self._last_street >= 1:
                    self._stats["call_down"][1] += 1

        # Check-raise detection + mid-hand override
        if not self._last_was_bet and opp_action == "raise":
            self._stats["check_raise"][0] += 1
            self._stats["check_raise"][1] += 1
            if self._hand_override is None:
                self._hand_override = MODE_VALUE  # stop bluffing this hand
        elif not self._last_was_bet and opp_action in ("check", "call", "fold"):
            self._stats["check_raise"][1] += 1
        elif self._last_was_bet and opp_action == "raise":
            # Villain re-raised our bet — stronger signal than check-raise, stop bluffing
            if self._hand_override is None:
                self._hand_override = MODE_VALUE

        # Overall opponent aggression — exclude folds so passive folders don't
        # dilute the rate and mask genuine aggression signals
        if opp_action in ("raise", "call", "check"):
            self._stats["opp_aggression"][1] += 1
            if opp_action == "raise":
                self._stats["opp_aggression"][0] += 1
                # Track opponent bet size as fraction of pot when they raise
                opp_bet_obs = observation.get("opp_bet", 0)
                my_bet_obs  = observation.get("my_bet", 0)
                pot_obs     = observation.get("pot_size", opp_bet_obs + my_bet_obs)
                raise_size  = max(0, opp_bet_obs - my_bet_obs)
                if pot_obs > 0 and raise_size > 0:
                    frac = raise_size / pot_obs
                    self._stats["opp_avg_bet_frac"][0] += frac
                    self._stats["opp_avg_bet_frac"][1] += 1

        # ── New stats: AF per street, 3-bet freq, preflop raise, river call freq ──

        # 3-bet frequency: villain re-raised our bet
        if self._last_was_bet and opp_action in ("raise", "call", "fold", "check"):
            self._stats["opp_3bet_freq"][1] += 1
            if opp_action == "raise":
                self._stats["opp_3bet_freq"][0] += 1

        # Preflop raise tracking (any preflop action by opponent)
        if self._last_street == 0 and opp_action in ("raise", "call", "check", "fold"):
            self._stats["opp_preflop_raise"][1] += 1
            if opp_action == "raise":
                self._stats["opp_preflop_raise"][0] += 1

        # Street-specific Aggression Factor: [0]=bets+raises, [1]=calls
        af_street_map = {1: "opp_af_flop", 2: "opp_af_turn", 3: "opp_af_river"}
        if self._last_street in af_street_map:
            af_key = af_street_map[self._last_street]
            if opp_action == "raise":
                self._stats[af_key][0] += 1   # villain bet/raised → aggression count
            elif opp_action == "call":
                self._stats[af_key][1] += 1   # villain called → passive count

        # River call frequency: how often villain calls our river bet
        if self._last_street == 3 and self._last_was_bet:
            self._stats["opp_river_call_freq"][1] += 1
            if opp_action == "call":
                self._stats["opp_river_call_freq"][0] += 1
