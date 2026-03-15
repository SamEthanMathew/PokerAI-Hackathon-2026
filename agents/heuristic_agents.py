# agents/heuristic_agents.py
"""
12 heuristic agents for the 27-card keep-2 poker variant.

Each class returns actions in 4-tuple format:
(action_type, raise_amount, keep1, keep2)
"""

from __future__ import annotations

import os
import math
import random
from typing import List, Tuple, Optional

import numpy as np

from agents.agent import Agent
from gym_env import PokerEnv

from agents.heuristics_core import (
    action_types,
    KEEP_INDEX_PAIRS,
    count_suits,
    count_ranks,
    rank_of,
    suit_of,
    is_connector,
    is_gap1,
    has_trips,
    board_texture,
    eval_class,
    eval_score,
    is_straight_plus,
    is_two_pair_plus,
    is_pair_plus,
    pot_odds,
    pot_size,
    pot_frac_raise,
    clamp_raise,
    make_legal_action,
    user_priority_tier,
    prefer_flush_leverage,
    straight_coverage,
    exact_strength_keep,
    exact_equity_keep,
    exact_equity_discard,
    load_rank7_table,
    load_board_hand_table,
    OpponentStats,
    default_rng,
)

# -----------------------------
# Shared base class (thin)
# -----------------------------
class HeuristicBaseAgent(Agent):
    """
    Base heuristic agent.
    Subclasses should override:
      - choose_keep_indices(...)
      - choose_betting_action(...)
    """

    def __init__(self, stream: bool = False, player_id: str = None):
        super().__init__(stream=stream, player_id=player_id)
        self._reset_hand_state()

    def _reset_hand_state(self):
        self.force_fold = False
        self.seen_hand = False
        self.opp_stats = getattr(self, "opp_stats", OpponentStats())
        self.we_raised_this_hand = False
        self.last_seen_street = None
        self.last_seen_opp_last_action = None
        self.opp_checks_on_street = 0

    def _maybe_new_hand(self, obs: dict):
        my_cards = [c for c in obs["my_cards"] if c != -1]
        if obs["street"] == 0 and len(my_cards) == 5 and not self.seen_hand:
            self.seen_hand = True
            self.we_raised_this_hand = False
            self.last_seen_street = obs["street"]
            self.opp_checks_on_street = 0
            self.last_seen_opp_last_action = obs.get("opp_last_action", "None")

    def _update_per_street_counters(self, obs: dict):
        street = obs["street"]
        opp_last = obs.get("opp_last_action", "None")
        if self.last_seen_street is None or street != self.last_seen_street:
            self.last_seen_street = street
            self.opp_checks_on_street = 0
            self.last_seen_opp_last_action = opp_last
            return
        if opp_last != self.last_seen_opp_last_action:
            if opp_last == "CHECK":
                self.opp_checks_on_street += 1
            self.last_seen_opp_last_action = opp_last

    def act(self, observation, reward, terminated, truncated, info):
        obs = observation
        self._maybe_new_hand(obs)
        self._update_per_street_counters(obs)

        va = obs["valid_actions"]
        my_cards = [c for c in obs["my_cards"] if c != -1]
        board = [c for c in obs["community_cards"] if c != -1]
        opp_disc = list(obs.get("opp_discarded_cards", [-1, -1, -1]))
        time_left = float(obs.get("time_left", 1e9))

        if va[action_types.DISCARD.value]:
            i, j = self.choose_keep_indices(my_cards, board, obs, opp_disc, time_left)
            return make_legal_action(obs, (action_types.DISCARD.value, 0, i, j))

        if self.force_fold and va[action_types.FOLD.value]:
            return make_legal_action(obs, (action_types.FOLD.value, 0, 0, 0))

        action = self.choose_betting_action(my_cards, board, obs, opp_disc, time_left)
        if action[0] == action_types.RAISE.value:
            self.we_raised_this_hand = True
        return make_legal_action(obs, action)

    def observe(self, observation, reward, terminated, truncated, info):
        if not terminated:
            return
        opp_last = observation.get("opp_last_action", "None")
        self.opp_stats.update_on_terminal(opp_last_action=opp_last, we_raised_this_hand=self.we_raised_this_hand)
        self.seen_hand = False
        self.force_fold = False

    def choose_keep_indices(self, my5: List[int], boardN: List[int], obs: dict, opp_disc: List[int], time_left: float) -> Tuple[int, int]:
        return (0, 1)

    def choose_betting_action(self, hole: List[int], boardN: List[int], obs: dict, opp_disc: List[int], time_left: float) -> Tuple[int, int, int, int]:
        if obs["valid_actions"][action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, 0, 0)
        if obs["valid_actions"][action_types.CALL.value]:
            return (action_types.CALL.value, 0, 0, 0)
        return (action_types.FOLD.value, 0, 0, 0)

# -----------------------------
# Shared discard primitives used by multiple agents
# -----------------------------
def choose_keep_user_priority(
    my5: List[int],
    flop3: List[int],
    bump_suited_tier: bool = False,
    pair_penalty_threshold_rank: Optional[int] = None,
) -> Tuple[int, int]:
    bt = board_texture(flop3, street=1)
    two_tone = bt.is_two_tone

    best = None
    for i, j in KEEP_INDEX_PAIRS:
        hole2 = [my5[i], my5[j]]
        tier, neg_suited, neg_conn, neg_top, neg_sum = user_priority_tier(hole2)

        if bump_suited_tier and two_tone and (suit_of(hole2[0]) == suit_of(hole2[1])):
            tier = max(1, tier - 1)

        if pair_penalty_threshold_rank is not None:
            r1, r2 = rank_of(hole2[0]), rank_of(hole2[1])
            if r1 == r2 and r1 < pair_penalty_threshold_rank:
                tier = min(4, tier + 1)

        flush_bonus = -prefer_flush_leverage(hole2, flop3)

        key = (tier, flush_bonus, neg_suited, neg_conn, neg_top, neg_sum)
        if best is None or key < best[0]:
            best = (key, (i, j))
    return best[1]

def strong_start_from_my5(my5: List[int]) -> bool:
    best_pair_rank = -1
    for i, j in KEEP_INDEX_PAIRS:
        c1, c2 = my5[i], my5[j]
        r1, r2 = rank_of(c1), rank_of(c2)
        s1, s2 = suit_of(c1), suit_of(c2)
        top = max(r1, r2)

        if r1 == r2:
            best_pair_rank = max(best_pair_rank, r1)
        if s1 == s2 and is_connector(r1, r2) and top >= 6:
            return True
        if s1 == s2 and is_gap1(r1, r2) and top in (7, 8):
            return True

    if best_pair_rank >= 6:
        return True
    return False

# -----------------------------
# Betting primitives (shared)
# -----------------------------
def equity_proxy_from_class(rank_class: int) -> float:
    if rank_class <= 3:
        return 0.95
    if rank_class == 4:
        return 0.85
    if rank_class == 5:
        return 0.80
    if rank_class == 6:
        return 0.70
    if rank_class == 7:
        return 0.62
    if rank_class == 8:
        return 0.52
    return 0.42

def default_bet_tree(
    hole2: List[int],
    boardN: List[int],
    obs: dict,
    value_thresh: float,
    raise_frac_value: float,
    call_margin: float = 0.0,
) -> Tuple[int, int, int, int]:
    va = obs["valid_actions"]
    E = equity_proxy_from_class(eval_class(hole2, boardN))
    O = pot_odds(obs)

    if max(0, obs["opp_bet"] - obs["my_bet"]) > 0:
        if va[action_types.CALL.value] and E >= O + call_margin:
            return (action_types.CALL.value, 0, 0, 0)
        return (action_types.FOLD.value, 0, 0, 0)

    if va[action_types.RAISE.value] and E >= value_thresh:
        ra = pot_frac_raise(obs, raise_frac_value)
        return (action_types.RAISE.value, ra, 0, 0)

    if va[action_types.CHECK.value]:
        return (action_types.CHECK.value, 0, 0, 0)
    if va[action_types.CALL.value]:
        return (action_types.CALL.value, 0, 0, 0)
    return (action_types.FOLD.value, 0, 0, 0)

# =====================================================================
# Heuristic 1: PriorityConservative
# =====================================================================
class PriorityConservativeAgent(HeuristicBaseAgent):
    """
    Strict user-priority discard ordering (suited connectors > suited gap1
    > high pairs > highs). Conservative betting: value-bet only two-pair+
    post-discard, otherwise pot-odds call or fold.
    """

    def __init__(
        self,
        stream: bool = False,
        player_id: str = None,
        min_raise_factor: float = 1.0,
        value_bet_class_cutoff: int = 7,
        call_margin: float = 0.02,
        emergency_time_left: float = 3.0,
    ):
        super().__init__(stream=stream, player_id=player_id)
        self.min_raise_factor = min_raise_factor
        self.value_bet_class_cutoff = value_bet_class_cutoff
        self.call_margin = call_margin
        self.emergency_time_left = emergency_time_left

    def __name__(self):
        return "PriorityConservativeAgent"

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if obs["street"] == 1 and has_trips(my5):
            self.force_fold = True
        return choose_keep_user_priority(my5, flop3, bump_suited_tier=False)

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        va = obs["valid_actions"]
        street = obs["street"]

        if street == 0:
            if strong_start_from_my5(hole + [-1, -1, -1]):
                if va[action_types.RAISE.value]:
                    ra = clamp_raise(obs, int(obs["min_raise"] * self.min_raise_factor))
                    return (action_types.RAISE.value, ra, 0, 0)
                if va[action_types.CALL.value]:
                    return (action_types.CALL.value, 0, 0, 0)
            if va[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        hole2 = hole[:2]
        cls = eval_class(hole2, boardN)

        if max(0, obs["opp_bet"] - obs["my_bet"]) > 0:
            if is_pair_plus(cls) and va[action_types.CALL.value] and equity_proxy_from_class(cls) >= pot_odds(obs) + self.call_margin:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        if cls <= self.value_bet_class_cutoff and va[action_types.RAISE.value]:
            ra = clamp_raise(obs, int(obs["min_raise"] * self.min_raise_factor))
            return (action_types.RAISE.value, ra, 0, 0)

        if va[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, 0, 0)
        return (action_types.FOLD.value, 0, 0, 0)


# =====================================================================
# Heuristic 2: PriorityAggressive
# =====================================================================
class PriorityAggressiveAgent(HeuristicBaseAgent):
    """
    Same discard baseline as PriorityConservative but values suited hands more
    on two-tone flops. Bets aggressively when checked to with pair+.
    """

    def __init__(
        self,
        stream: bool = False,
        player_id: str = None,
        cbet_size_frac: float = 0.50,
        thin_value_threshold_class: int = 8,
        equity_buffer: float = 0.05,
        emergency_time_left: float = 3.0,
    ):
        super().__init__(stream=stream, player_id=player_id)
        self.cbet_size_frac = cbet_size_frac
        self.thin_value_threshold_class = thin_value_threshold_class
        self.equity_buffer = equity_buffer
        self.emergency_time_left = emergency_time_left

    def __name__(self):
        return "PriorityAggressiveAgent"

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if obs["street"] == 1 and has_trips(my5):
            self.force_fold = True
        return choose_keep_user_priority(my5, flop3, bump_suited_tier=True)

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        va = obs["valid_actions"]
        street = obs["street"]

        if street == 0:
            my5 = [c for c in obs["my_cards"] if c != -1]
            if strong_start_from_my5(my5):
                if va[action_types.RAISE.value]:
                    return (action_types.RAISE.value, clamp_raise(obs, obs["min_raise"]), 0, 0)
                if va[action_types.CALL.value]:
                    return (action_types.CALL.value, 0, 0, 0)
            if obs["blind_position"] == 1 and va[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        hole2 = hole[:2]
        cls = eval_class(hole2, boardN)
        E = equity_proxy_from_class(cls)
        O = pot_odds(obs)

        facing = max(0, obs["opp_bet"] - obs["my_bet"]) > 0
        if facing:
            if va[action_types.CALL.value] and E >= O + self.equity_buffer:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        if va[action_types.RAISE.value] and cls <= self.thin_value_threshold_class and obs.get("opp_last_action", "None") == "CHECK":
            ra = pot_frac_raise(obs, self.cbet_size_frac)
            return (action_types.RAISE.value, ra, 0, 0)

        if street == 3 and va[action_types.RAISE.value] and cls <= self.thin_value_threshold_class and self.opp_checks_on_street >= 1:
            ra = pot_frac_raise(obs, 0.25)
            return (action_types.RAISE.value, ra, 0, 0)

        if va[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, 0, 0)
        return (action_types.FOLD.value, 0, 0, 0)


# =====================================================================
# Heuristic 3: BoardMadeFirst
# =====================================================================
class BoardMadeFirstAgent(HeuristicBaseAgent):
    """
    Discard prioritizes the best immediate made hand on flop (hole2+flop3).
    Betting values made hands strongly; reduces aggression on scary boards
    unless holding blockers.
    """

    def __init__(
        self,
        stream: bool = False,
        player_id: str = None,
        raise_frac_straight_plus: float = 0.75,
        raise_frac_trips_2pair: float = 0.50,
        blocker_downgrade: bool = True,
        emergency_time_left: float = 3.0,
    ):
        super().__init__(stream=stream, player_id=player_id)
        self.raise_frac_straight_plus = raise_frac_straight_plus
        self.raise_frac_trips_2pair = raise_frac_trips_2pair
        self.blocker_downgrade = blocker_downgrade
        self.emergency_time_left = emergency_time_left

    def __name__(self):
        return "BoardMadeFirstAgent"

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if obs["street"] == 1 and has_trips(my5):
            self.force_fold = True

        best_made = None
        for i, j in KEEP_INDEX_PAIRS:
            hole2 = [my5[i], my5[j]]
            cls = eval_class(hole2, flop3)
            sc = eval_score(hole2, flop3)
            cand = (cls, sc, (i, j))
            if best_made is None or cand < best_made:
                best_made = cand

        if best_made[0] <= 7:
            return best_made[2]
        return choose_keep_user_priority(my5, flop3, bump_suited_tier=False)

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        va = obs["valid_actions"]
        street = obs["street"]

        if street == 0:
            my5 = [c for c in obs["my_cards"] if c != -1]
            if strong_start_from_my5(my5) and va[action_types.CALL.value]:
                return (action_types.CALL.value, 0, 0, 0)
            if va[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        hole2 = hole[:2]
        cls = eval_class(hole2, boardN)

        tex = board_texture(boardN, street=street)
        board_scary = tex.four_to_straight_flag or (tex.max_suit >= 3 and street >= 2)

        holds_flush_blocker = False
        if tex.max_suit >= 3:
            target_suit = int(np.argmax(tex.suit_counts))
            holds_flush_blocker = any(suit_of(c) == target_suit for c in hole2)

        facing = max(0, obs["opp_bet"] - obs["my_bet"]) > 0
        if facing:
            E = equity_proxy_from_class(cls)
            if va[action_types.CALL.value] and E >= pot_odds(obs):
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        if not va[action_types.RAISE.value]:
            return (action_types.CHECK.value, 0, 0, 0)

        if self.blocker_downgrade and board_scary and not holds_flush_blocker and cls > 5:
            return (action_types.CHECK.value, 0, 0, 0)

        if is_straight_plus(cls):
            return (action_types.RAISE.value, pot_frac_raise(obs, self.raise_frac_straight_plus), 0, 0)
        if cls in (6, 7):
            return (action_types.RAISE.value, pot_frac_raise(obs, self.raise_frac_trips_2pair), 0, 0)
        if cls == 8 and obs.get("opp_last_action", "None") == "CHECK":
            return (action_types.RAISE.value, pot_frac_raise(obs, 0.25), 0, 0)

        return (action_types.CHECK.value, 0, 0, 0)


# =====================================================================
# Heuristic 4: OutsMaximizer
# =====================================================================
class OutsMaximizerAgent(HeuristicBaseAgent):
    """
    Discard chooses keep pair that maximizes probability of finishing straight+
    by river (enumerates all legal turn/river runouts; ignores opponent).
    """

    def __init__(
        self,
        stream: bool = False,
        player_id: str = None,
        semi_bluff_freq: float = 0.10,
        foldrate_trigger: float = 0.55,
        emergency_time_left: float = 3.0,
        rng_seed: int = 0,
    ):
        super().__init__(stream=stream, player_id=player_id)
        self.semi_bluff_freq = semi_bluff_freq
        self.foldrate_trigger = foldrate_trigger
        self.emergency_time_left = emergency_time_left
        self.rng = default_rng(rng_seed)

    def __name__(self):
        return "OutsMaximizerAgent"

    def _p_finish_classes(self, my5, flop3, keep_idx, opp_disc):
        i, j = keep_idx
        hole2 = [my5[i], my5[j]]
        dead = my5 + flop3 + [c for c in opp_disc if c != -1]
        dm = 0
        for c in dead:
            dm |= (1 << int(c))
        from agents.heuristics_core import enumerate_runouts
        runouts = enumerate_runouts(dm)

        cnt = 0
        cnt_straight_plus = 0
        cnt_pair_plus = 0
        cnt_full_house_plus = 0

        for t, r in runouts.tolist():
            board5 = flop3 + [int(t), int(r)]
            cls = eval_class(hole2, board5)
            cnt += 1
            if cls <= 3:
                cnt_full_house_plus += 1
            if cls <= 5:
                cnt_straight_plus += 1
            if cls <= 8:
                cnt_pair_plus += 1

        if cnt == 0:
            return (0.0, 0.0, 0.0)
        return (
            cnt_straight_plus / cnt,
            cnt_pair_plus / cnt,
            cnt_full_house_plus / cnt,
        )

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if obs["street"] == 1 and has_trips(my5):
            self.force_fold = True

        best = None
        for i, j in KEEP_INDEX_PAIRS:
            p_str, p_pair, p_fh = self._p_finish_classes(my5, flop3, (i, j), opp_disc)
            key = (p_str, p_fh, (rank_of(my5[i]) + rank_of(my5[j])))
            if best is None or key > best[0]:
                best = (key, (i, j))
        return best[1]

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        va = obs["valid_actions"]
        street = obs["street"]

        if street == 0:
            if va[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, 0, 0)
            if va[action_types.CALL.value]:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        hole2 = hole[:2]
        cls = eval_class(hole2, boardN)
        E = equity_proxy_from_class(cls)
        O = pot_odds(obs)

        facing = max(0, obs["opp_bet"] - obs["my_bet"]) > 0
        if facing:
            if va[action_types.CALL.value] and E >= O:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        if va[action_types.RAISE.value] and self.opp_stats.fold_to_raise >= self.foldrate_trigger:
            if cls >= 8 and self.rng.random() < self.semi_bluff_freq:
                return (action_types.RAISE.value, pot_frac_raise(obs, 0.25), 0, 0)

        if va[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, 0, 0)
        return (action_types.FOLD.value, 0, 0, 0)


# =====================================================================
# Heuristic 5: ExactStrengthEnumerate
# =====================================================================
class ExactStrengthEnumerateAgent(HeuristicBaseAgent):
    """
    Discard enumerates all legal turn/river runouts for each keep pair and
    chooses the keep with best expected evaluator score (mean), tie-breaking
    by lower variance.
    """

    def __init__(
        self,
        stream: bool = False,
        player_id: str = None,
        board_hand_rank_path: str = "board_hand_rank_uint16.npy",
        rank_weight: float = 1.0,
        var_weight: float = 0.10,
        raise_thresh: float = 0.60,
        call_margin: float = 0.02,
        emergency_time_left: float = 3.0,
    ):
        super().__init__(stream=stream, player_id=player_id)
        self.board_hand_table = load_board_hand_table(board_hand_rank_path)
        self.rank_weight = rank_weight
        self.var_weight = var_weight
        self.raise_thresh = raise_thresh
        self.call_margin = call_margin
        self.emergency_time_left = emergency_time_left

    def __name__(self):
        return "ExactStrengthEnumerateAgent"

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if obs["street"] == 1 and has_trips(my5):
            self.force_fold = True

        best = None
        for i, j in KEEP_INDEX_PAIRS:
            mean_s, var_s = exact_strength_keep(my5, flop3, (i, j), opp_disc, self.board_hand_table)
            key = (mean_s * self.rank_weight + var_s * self.var_weight, mean_s, var_s)
            if best is None or key < best[0]:
                best = (key, (i, j))
        return best[1]

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        va = obs["valid_actions"]
        street = obs["street"]

        if street == 0:
            if va[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, 0, 0)
            if va[action_types.CALL.value]:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        hole2 = hole[:2]
        cls = eval_class(hole2, boardN)
        E = equity_proxy_from_class(cls)
        O = pot_odds(obs)

        facing = max(0, obs["opp_bet"] - obs["my_bet"]) > 0
        if facing:
            if va[action_types.CALL.value] and E >= O + self.call_margin:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        if va[action_types.RAISE.value] and E >= self.raise_thresh:
            return (action_types.RAISE.value, pot_frac_raise(obs, 0.50), 0, 0)

        if va[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, 0, 0)
        return (action_types.FOLD.value, 0, 0, 0)


# =====================================================================
# Heuristic 6: ExactEquityEnumerate
# =====================================================================
class ExactEquityEnumerateAgent(HeuristicBaseAgent):
    """
    Discard chooses keep pair by exact equity: enumerate all turn/river runouts
    and all opponent hole2 holdings from remaining cards.
    """

    def __init__(
        self,
        stream: bool = False,
        player_id: str = None,
        board_hand_rank_path: str = "board_hand_rank_uint16.npy",
        equity_raise_threshold: float = 0.75,
        tie_weight: float = 0.5,
        top_k_refine: int = 10,
        exact_time_left: float = 10.0,
        emergency_time_left: float = 3.0,
    ):
        super().__init__(stream=stream, player_id=player_id)
        self.board_hand_table = load_board_hand_table(board_hand_rank_path)
        self.equity_raise_threshold = equity_raise_threshold
        self.tie_weight = tie_weight
        self.top_k_refine = int(top_k_refine)
        self.exact_time_left = exact_time_left
        self.emergency_time_left = emergency_time_left

    def __name__(self):
        return "ExactEquityEnumerateAgent"

    def _pre_score(self, my5, flop3, keep_idx, opp_disc):
        i, j = keep_idx
        hole2 = [my5[i], my5[j]]
        tier = user_priority_tier(hole2)[0]
        cov = straight_coverage(hole2)
        fl = prefer_flush_leverage(hole2, flop3)
        return (-tier) + 0.20 * cov + 0.50 * fl

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if obs["street"] == 1 and has_trips(my5):
            self.force_fold = True

        if time_left < self.emergency_time_left:
            return choose_keep_user_priority(my5, flop3, bump_suited_tier=True)

        K = None if self.top_k_refine >= 10 else self.top_k_refine
        (i, j), eq = exact_equity_discard(
            my5=my5,
            flop3=flop3,
            opp_discards=opp_disc,
            board_hand_table=self.board_hand_table,
            top_k=K,
            pre_score_fn=self._pre_score if K is not None else None,
            tie_weight=self.tie_weight,
        )
        return (i, j)

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        va = obs["valid_actions"]
        street = obs["street"]

        if street == 0:
            if va[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, 0, 0)
            if va[action_types.CALL.value]:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        hole2 = hole[:2]
        cls = eval_class(hole2, boardN)
        E = equity_proxy_from_class(cls)
        O = pot_odds(obs)

        facing = max(0, obs["opp_bet"] - obs["my_bet"]) > 0
        if facing:
            if va[action_types.CALL.value] and E >= O:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        if va[action_types.RAISE.value] and E >= self.equity_raise_threshold:
            return (action_types.RAISE.value, pot_frac_raise(obs, 0.75), 0, 0)

        if va[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, 0, 0)
        if va[action_types.CALL.value]:
            return (action_types.CALL.value, 0, 0, 0)
        return (action_types.FOLD.value, 0, 0, 0)


# =====================================================================
# Heuristic 7: OppDiscardAware
# =====================================================================
class OppDiscardAwareAgent(HeuristicBaseAgent):
    """
    Uses opponent discarded cards (when visible) to adjust discard selection
    and betting thresholds.
    """

    def __init__(
        self,
        stream: bool = False,
        player_id: str = None,
        board_hand_rank_path: str = "board_hand_rank_uint16.npy",
        blocker_bonus: float = 0.15,
        worst_case_mode: bool = True,
        emergency_time_left: float = 3.0,
    ):
        super().__init__(stream=stream, player_id=player_id)
        self.board_hand_table = load_board_hand_table(board_hand_rank_path)
        self.blocker_bonus = blocker_bonus
        self.worst_case_mode = worst_case_mode
        self.emergency_time_left = emergency_time_left

    def __name__(self):
        return "OppDiscardAwareAgent"

    def _infer_target_suits(self, opp_disc: List[int]) -> List[int]:
        sc = count_suits(opp_disc)
        targets = [s for s in range(3) if sc[s] == 0]
        return targets

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if obs["street"] == 1 and has_trips(my5):
            self.force_fold = True

        opp_visible = all(c != -1 for c in opp_disc)
        if not opp_visible or time_left < self.emergency_time_left:
            best = None
            for i, j in KEEP_INDEX_PAIRS:
                mean_s, var_s = exact_strength_keep(my5, flop3, (i, j), opp_disc, self.board_hand_table)
                key = (mean_s, var_s)
                if best is None or key < best[0]:
                    best = (key, (i, j))
            return best[1]

        targets = self._infer_target_suits(opp_disc)

        best = None
        for i, j in KEEP_INDEX_PAIRS:
            hole2 = [my5[i], my5[j]]
            mean_s, var_s = exact_strength_keep(my5, flop3, (i, j), opp_disc, self.board_hand_table)

            block = 0.0
            for s in targets:
                if any(suit_of(c) == s for c in hole2):
                    block += self.blocker_bonus

            score = -(mean_s / 10000.0) + block - 0.01 * (var_s / 1e7)
            if best is None or score > best[0]:
                best = (score, (i, j))
        return best[1]

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        if obs["street"] == 0:
            va = obs["valid_actions"]
            if va[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, 0, 0)
            if va[action_types.CALL.value]:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        targets = self._infer_target_suits(opp_disc) if all(c != -1 for c in opp_disc) else []
        flush_threat = len(targets) == 0

        base_value_thresh = 0.65 if not flush_threat else 0.70
        return default_bet_tree(
            hole2=hole[:2],
            boardN=boardN,
            obs=obs,
            value_thresh=base_value_thresh,
            raise_frac_value=0.50,
            call_margin=0.02,
        )


# =====================================================================
# Heuristic 8: BlockerValue
# =====================================================================
class BlockerValueAgent(HeuristicBaseAgent):
    """
    Tries to keep cards that block opponent's likely kept suit/straight lines
    inferred from their discards. Adapts raise sizing to fold/station tendencies.
    """

    def __init__(
        self,
        stream: bool = False,
        player_id: str = None,
        blocker_weight: float = 0.30,
        station_size: float = 0.35,
        foldy_size: float = 0.25,
        emergency_time_left: float = 3.0,
    ):
        super().__init__(stream=stream, player_id=player_id)
        self.blocker_weight = blocker_weight
        self.station_size = station_size
        self.foldy_size = foldy_size
        self.emergency_time_left = emergency_time_left

    def __name__(self):
        return "BlockerValueAgent"

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if obs["street"] == 1 and has_trips(my5):
            self.force_fold = True

        opp_visible = all(c != -1 for c in opp_disc)
        if not opp_visible or time_left < self.emergency_time_left:
            return choose_keep_user_priority(my5, flop3, bump_suited_tier=True)

        sc = count_suits(opp_disc)
        target_suit = int(np.argmin(sc))

        best = None
        for i, j in KEEP_INDEX_PAIRS:
            hole2 = [my5[i], my5[j]]
            base_key = user_priority_tier(hole2)
            block = 1 if any(suit_of(c) == target_suit for c in hole2) else 0
            cov = straight_coverage(hole2)

            score = (-base_key[0]) + self.blocker_weight * block + 0.10 * cov + 0.50 * prefer_flush_leverage(hole2, flop3)
            if best is None or score > best[0]:
                best = (score, (i, j))
        return best[1]

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        va = obs["valid_actions"]
        street = obs["street"]

        if street == 0:
            my5 = [c for c in obs["my_cards"] if c != -1]
            if strong_start_from_my5(my5) and va[action_types.RAISE.value]:
                return (action_types.RAISE.value, clamp_raise(obs, obs["min_raise"]), 0, 0)
            if va[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, 0, 0)
            if va[action_types.CALL.value]:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        hole2 = hole[:2]
        cls = eval_class(hole2, boardN)
        E = equity_proxy_from_class(cls)
        O = pot_odds(obs)

        facing = max(0, obs["opp_bet"] - obs["my_bet"]) > 0
        if facing:
            if va[action_types.CALL.value] and E >= O:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        if not va[action_types.RAISE.value]:
            return (action_types.CHECK.value, 0, 0, 0)

        frac = self.foldy_size if self.opp_stats.fold_to_raise >= 0.55 else self.station_size

        if cls <= 8:
            return (action_types.RAISE.value, pot_frac_raise(obs, frac), 0, 0)

        return (action_types.CHECK.value, 0, 0, 0)


# =====================================================================
# Heuristic 9: MinRiskPressure
# =====================================================================
class MinRiskPressureAgent(HeuristicBaseAgent):
    """
    Discard uses user-priority but penalizes non-top pairs. Betting uses
    mostly min-raise pressure and avoids large bluffs.
    """

    def __init__(
        self,
        stream: bool = False,
        player_id: str = None,
        pair_penalty_rank: int = 7,
        minraise_bluff_freq: float = 0.10,
        station_factor: float = 1.5,
        info_leak_penalty: float = 0.20,
        emergency_time_left: float = 3.0,
        rng_seed: int = 1,
    ):
        super().__init__(stream=stream, player_id=player_id)
        self.pair_penalty_rank = pair_penalty_rank
        self.minraise_bluff_freq = minraise_bluff_freq
        self.station_factor = station_factor
        self.info_leak_penalty = info_leak_penalty
        self.emergency_time_left = emergency_time_left
        self.rng = default_rng(rng_seed)

    def __name__(self):
        return "MinRiskPressureAgent"

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if obs["street"] == 1 and has_trips(my5):
            self.force_fold = True
        return choose_keep_user_priority(
            my5, flop3,
            bump_suited_tier=True,
            pair_penalty_threshold_rank=self.pair_penalty_rank
        )

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        va = obs["valid_actions"]
        street = obs["street"]

        if street == 0:
            my5 = [c for c in obs["my_cards"] if c != -1]
            if strong_start_from_my5(my5) and va[action_types.RAISE.value]:
                return (action_types.RAISE.value, clamp_raise(obs, obs["min_raise"]), 0, 0)
            if va[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, 0, 0)
            if va[action_types.CALL.value]:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        hole2 = hole[:2]
        cls = eval_class(hole2, boardN)
        E = equity_proxy_from_class(cls)

        facing = max(0, obs["opp_bet"] - obs["my_bet"]) > 0
        if facing:
            if va[action_types.CALL.value] and E >= pot_odds(obs) + 0.05:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        if va[action_types.RAISE.value]:
            if cls <= 8:
                return (action_types.RAISE.value, clamp_raise(obs, obs["min_raise"]), 0, 0)
            if self.opp_stats.fold_to_raise >= 0.60 and self.rng.random() < self.minraise_bluff_freq:
                return (action_types.RAISE.value, clamp_raise(obs, obs["min_raise"]), 0, 0)

        if va[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, 0, 0)
        return (action_types.FOLD.value, 0, 0, 0)


# =====================================================================
# Heuristic 10: InfoHiderMixed
# =====================================================================
class InfoHiderMixedAgent(HeuristicBaseAgent):
    """
    Discard computes top-K "good" keeps then randomizes among them,
    penalizing choices where the 3 discarded cards reveal an obvious narrative.
    """

    def __init__(
        self,
        stream: bool = False,
        player_id: str = None,
        top_k: int = 3,
        temperature: float = 1.0,
        bluff_freq: float = 0.08,
        entropy_penalty: float = 0.30,
        emergency_time_left: float = 3.0,
        rng_seed: int = 2,
    ):
        super().__init__(stream=stream, player_id=player_id)
        self.top_k = int(top_k)
        self.temperature = temperature
        self.bluff_freq = bluff_freq
        self.entropy_penalty = entropy_penalty
        self.emergency_time_left = emergency_time_left
        self.rng = default_rng(rng_seed)

    def __name__(self):
        return "InfoHiderMixedAgent"

    def _discard_entropy_penalty(self, my5: List[int], keep_idx: Tuple[int, int]) -> float:
        i, j = keep_idx
        disc = [my5[k] for k in range(5) if k not in (i, j)]
        sc = count_suits(disc)
        rc = count_ranks(disc)
        suit_clump = float(sc.max() == 3)
        ranks_present = {r for r in range(9) if rc[r] > 0}
        rank_clump = float(len(ranks_present) == 3 and max(ranks_present) - min(ranks_present) <= 2)
        return self.entropy_penalty * (suit_clump + 0.5 * rank_clump)

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if obs["street"] == 1 and has_trips(my5):
            self.force_fold = True

        scored = []
        for i, j in KEEP_INDEX_PAIRS:
            hole2 = [my5[i], my5[j]]
            base = -user_priority_tier(hole2)[0] + 0.10 * straight_coverage(hole2) + 0.50 * prefer_flush_leverage(hole2, flop3)
            pen = self._discard_entropy_penalty(my5, (i, j))
            scored.append((base - pen, (i, j)))

        scored.sort(reverse=True)
        top = scored[: max(1, self.top_k)]

        vals = np.array([s for s, _ in top], dtype=np.float64)
        vals = (vals - vals.max()) / max(1e-9, self.temperature)
        w = np.exp(vals)
        w = w / w.sum()
        idx = int(np.searchsorted(np.cumsum(w), self.rng.random()))
        idx = max(0, min(len(top) - 1, idx))
        return top[idx][1]

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        va = obs["valid_actions"]
        street = obs["street"]

        if street == 0:
            if va[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, 0, 0)
            if va[action_types.CALL.value]:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        hole2 = hole[:2]
        cls = eval_class(hole2, boardN)
        E = equity_proxy_from_class(cls)
        O = pot_odds(obs)

        facing = max(0, obs["opp_bet"] - obs["my_bet"]) > 0
        if facing:
            if va[action_types.CALL.value] and E >= O:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        if va[action_types.RAISE.value] and self.opp_checks_on_street >= 2 and self.rng.random() < self.bluff_freq:
            return (action_types.RAISE.value, pot_frac_raise(obs, 0.25), 0, 0)

        if va[action_types.RAISE.value] and cls <= 8:
            return (action_types.RAISE.value, pot_frac_raise(obs, 0.35), 0, 0)

        if va[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, 0, 0)
        return (action_types.FOLD.value, 0, 0, 0)


# =====================================================================
# Heuristic 11: TripsAutoFoldStrict
# =====================================================================
class TripsAutoFoldStrictAgent(HeuristicBaseAgent):
    """
    If the original 5 cards contain trips, set force_fold=True and fold at
    the first betting opportunity after discard. Discard tries to minimize
    revealed strength (avoid keeping a pair).
    """

    def __init__(self, stream: bool = False, player_id: str = None, emergency_time_left: float = 3.0):
        super().__init__(stream=stream, player_id=player_id)
        self.emergency_time_left = emergency_time_left

    def __name__(self):
        return "TripsAutoFoldStrictAgent"

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if has_trips(my5):
            self.force_fold = True
            best = None
            for i, j in KEEP_INDEX_PAIRS:
                r1, r2 = rank_of(my5[i]), rank_of(my5[j])
                is_pair = int(r1 == r2)
                tier = user_priority_tier([my5[i], my5[j]])[0]
                key = (is_pair, tier)
                if best is None or key < best[0]:
                    best = (key, (i, j))
            return best[1]
        return choose_keep_user_priority(my5, flop3, bump_suited_tier=True)

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        return (action_types.FOLD.value, 0, 0, 0)


# =====================================================================
# Heuristic 12: AdaptiveOpponentModel
# =====================================================================
class AdaptiveOpponentModelAgent(HeuristicBaseAgent):
    """
    Tracks opponent tendencies (fold-to-raise, call-down) and adapts both
    discard and betting: vs foldy opponents prefer keeps with immediate
    strength; vs stations prefer higher final made-hand realization.
    """

    def __init__(
        self,
        stream: bool = False,
        player_id: str = None,
        board_hand_rank_path: str = "board_hand_rank_uint16.npy",
        adapt_lr: float = 0.05,
        min_hands_before_adapt: int = 30,
        trip_rule_enabled: bool = True,
        emergency_time_left: float = 3.0,
    ):
        super().__init__(stream=stream, player_id=player_id)
        self.board_hand_table = load_board_hand_table(board_hand_rank_path)
        self.adapt_lr = adapt_lr
        self.min_hands_before_adapt = int(min_hands_before_adapt)
        self.trip_rule_enabled = trip_rule_enabled
        self.emergency_time_left = emergency_time_left
        self.opp_stats = OpponentStats()

    def __name__(self):
        return "AdaptiveOpponentModelAgent"

    def _opponent_is_foldy(self) -> bool:
        if self.opp_stats.hands_seen < self.min_hands_before_adapt:
            return False
        return self.opp_stats.fold_to_raise >= 0.55

    def choose_keep_indices(self, my5, flop3, obs, opp_disc, time_left):
        if self.trip_rule_enabled and obs["street"] == 1 and has_trips(my5):
            self.force_fold = True

        foldy = self._opponent_is_foldy()

        best = None
        for i, j in KEEP_INDEX_PAIRS:
            hole2 = [my5[i], my5[j]]
            immediate_cls = eval_class(hole2, flop3)
            immediate_score = equity_proxy_from_class(immediate_cls)

            mean_s, var_s = exact_strength_keep(my5, flop3, (i, j), opp_disc, self.board_hand_table)
            future_score = -mean_s / 10000.0

            w_now = 0.70 if foldy else 0.30
            w_future = 1.0 - w_now
            score = w_now * immediate_score + w_future * future_score - 0.02 * (var_s / 1e7)

            if best is None or score > best[0]:
                best = (score, (i, j))
        return best[1]

    def choose_betting_action(self, hole, boardN, obs, opp_disc, time_left):
        va = obs["valid_actions"]
        street = obs["street"]

        if street == 0:
            my5 = [c for c in obs["my_cards"] if c != -1]
            if strong_start_from_my5(my5) and va[action_types.RAISE.value]:
                return (action_types.RAISE.value, clamp_raise(obs, obs["min_raise"]), 0, 0)
            if va[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, 0, 0)
            if va[action_types.CALL.value]:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        hole2 = hole[:2]
        cls = eval_class(hole2, boardN)
        E = equity_proxy_from_class(cls)
        O = pot_odds(obs)

        foldy = self._opponent_is_foldy()
        facing = max(0, obs["opp_bet"] - obs["my_bet"]) > 0

        if facing:
            margin = 0.02 if foldy else 0.00
            if va[action_types.CALL.value] and E >= O + margin:
                return (action_types.CALL.value, 0, 0, 0)
            return (action_types.FOLD.value, 0, 0, 0)

        if not va[action_types.RAISE.value]:
            return (action_types.CHECK.value, 0, 0, 0)

        frac = 0.25 if foldy else 0.55

        if cls <= 8:
            return (action_types.RAISE.value, pot_frac_raise(obs, frac), 0, 0)

        return (action_types.CHECK.value, 0, 0, 0)
