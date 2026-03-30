"""
Multi-component reward function with decaying shaping.

Primary:  chips won/lost (always weight 1.0)
Shaping:  decays to zero by 80% through training to let RL take over.

Shaping components:
  +0.30 fold equity bonus (won pot without showdown)
  -0.30 bad fold penalty  (folded with good pot odds)
  +0.15 discard quality   (chose optimal keep pair)
  +0.10 value extraction  (won big pot at showdown)
"""

from typing import Optional


class RewardShaper:
    """
    Stateful reward shaper.  Pass training_progress (0.0 → 1.0) to shape().

    training_progress is computed as cycle / total_cycles by the training loop.
    """

    def __init__(
        self,
        shape_decay_end: float = 0.80,
        fold_equity_bonus: float = 0.30,
        bad_fold_penalty: float = 0.30,
        discard_quality_bonus: float = 0.15,
        value_extraction_bonus: float = 0.10,
        big_pot_threshold: int = 20,
    ):
        self.shape_decay_end = shape_decay_end
        self.fold_equity_bonus = fold_equity_bonus
        self.bad_fold_penalty = bad_fold_penalty
        self.discard_quality_bonus = discard_quality_bonus
        self.value_extraction_bonus = value_extraction_bonus
        self.big_pot_threshold = big_pot_threshold

    def _shaping_weight(self, training_progress: float) -> float:
        """Full shaping early, linearly decays to 0 by shape_decay_end."""
        return max(0.0, 1.0 - training_progress / self.shape_decay_end)

    def shape(
        self,
        obs: dict,
        action: tuple,
        hand_result: Optional[dict],
        training_progress: float,
        tables: Optional[dict] = None,
    ) -> float:
        """
        Compute the full reward for one step/hand.

        obs:              Current observation (before action is resolved)
        action:           (action_type, raise_amount, keep1, keep2)
        hand_result:      dict with keys: chips_won_lost, won (bool), showdown (bool)
                          OR None (for intermediate steps within a hand)
        training_progress: float 0.0 → 1.0
        tables:           Precomputed tables dict (for discard quality lookup)

        Returns total reward float.
        """
        sw = self._shaping_weight(training_progress)

        # ── Primary: chips ────────────────────────────────────────────────
        reward = 0.0
        if hand_result is not None:
            reward = float(hand_result.get("chips_won_lost", 0.0))

        if sw <= 0.0 or hand_result is None:
            return reward

        action_type = action[0] if isinstance(action, (tuple, list)) else action

        # ── Fold equity bonus ─────────────────────────────────────────────
        if (hand_result.get("won", False) and not hand_result.get("showdown", True)):
            reward += self.fold_equity_bonus * sw

        # ── Bad fold penalty ──────────────────────────────────────────────
        if action_type == 0:  # FOLD
            to_call = max(0, obs["opp_bet"] - obs["my_bet"])
            pot = obs["pot_size"]
            if to_call > 0 and pot > 0:
                pot_odds = to_call / (pot + to_call)
                if pot_odds < 0.20:
                    # We needed < 20% equity — probably had it, bad fold
                    reward -= self.bad_fold_penalty * sw

        # ── Discard quality bonus ─────────────────────────────────────────
        if action_type == 4:  # DISCARD
            optimal = _get_optimal_discard(obs, tables)
            if optimal is not None:
                keep1, keep2 = action[2], action[3]
                opt_i, opt_j = optimal
                if tuple(sorted([keep1, keep2])) == tuple(sorted([opt_i, opt_j])):
                    reward += self.discard_quality_bonus * sw

        # ── Value extraction bonus ────────────────────────────────────────
        if hand_result.get("won", False) and hand_result.get("showdown", False):
            chips = hand_result.get("chips_won_lost", 0)
            if chips > self.big_pot_threshold:
                reward += self.value_extraction_bonus * sw * (chips / 100.0)

        return reward


def _get_optimal_discard(obs: dict, tables: Optional[dict]):
    """
    Return (keep_i, keep_j) for optimal discard from tables, or None.
    """
    if tables is None:
        return None
    from precompute import lookup_optimal_discard
    my_cards = [c for c in obs.get("my_cards", []) if c >= 0]
    community = [c for c in obs.get("community_cards", []) if c >= 0]
    if len(my_cards) < 5 or len(community) < 3:
        return None
    return lookup_optimal_discard(tables, my_cards[:5], community[:3])


def compute_reward(
    obs: dict,
    action: tuple,
    hand_result: Optional[dict],
    training_progress: float,
    tables: Optional[dict] = None,
    shaper: Optional[RewardShaper] = None,
) -> float:
    """
    Convenience wrapper. Creates a default RewardShaper if none provided.
    """
    if shaper is None:
        shaper = RewardShaper()
    return shaper.shape(obs, action, hand_result, training_progress, tables)
