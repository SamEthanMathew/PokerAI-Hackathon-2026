"""
Genesis Agent – main player implementation.
Street 0: full preflop strategy via street0_score and street0_bet_sizing + opponent_recon.
Street >= 1: minimal pass-through (discard when required, else check/fold) so the engine runs.
"""

from agents.agent import Agent
from gym_env import PokerEnv

from submission.functions.street0_score import final_street0_score
from submission.functions.street0_bet_sizing import get_street0_action_from_recon
from submission.functions.opponent_recon import (
    OpponentRecon,
    start_new_hand,
    start_new_street,
    record_our_bet,
    update_opponent_actions,
    update_vpip_pfr,
    update_aggression_flags,
    update_fold_on_terminate,
    to_opponent_profile,
)

# Default sample counts for street0 score (keep in genesis; optional to move to config later)
DEFAULT_N_FLOP_SAMPLES = 150
DEFAULT_N_TR_SAMPLES = 50


class GenesisAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.recon = OpponentRecon()

    def __name__(self):
        return "GenesisAgent"

    def observe(self, observation, reward, terminated, truncated, info):
        """Update recon when hand terminates (fold-to-bet stats)."""
        if terminated:
            street = observation.get("street", -1)
            opp_last_action = observation.get("opp_last_action")
            update_fold_on_terminate(
                self.recon, street, opp_last_action, terminated
            )

    def act(self, observation, reward, terminated, truncated, info):
        """
        Street 0: full preflop strategy. Street >= 1: minimal valid actions.
        Returns: (action_type, raise_amount, keep_card_1, keep_card_2).
        """
        street = observation.get("street", 0)
        valid_actions = observation.get("valid_actions", [1, 1, 1, 1, 0])
        at = self.action_types

        # Ensure valid_actions is indexable for all 5 action types
        if len(valid_actions) < 5:
            valid_actions = list(valid_actions) + [0] * (5 - len(valid_actions))
        else:
            valid_actions = list(valid_actions)[:5]

        # --- Street >= 1: minimal pass-through ---
        if street > 0:
            if valid_actions[at.DISCARD.value]:
                return at.DISCARD.value, 0, 0, 1
            if valid_actions[at.CHECK.value]:
                return at.CHECK.value, 0, 0, 1
            return at.FOLD.value, 0, 0, 1

        # --- Street 0: full preflop strategy ---
        hand_number = info.get("hand_number", 0)

        if hand_number != self.recon.last_hand_number:
            start_new_hand(self.recon, hand_number)
        if street != self.recon.last_street:
            start_new_street(self.recon, street)

        update_opponent_actions(self.recon, observation.get("opp_last_action"))
        update_vpip_pfr(self.recon, observation, street)
        update_aggression_flags(self.recon, observation, street)

        if terminated:
            update_fold_on_terminate(
                self.recon,
                street,
                observation.get("opp_last_action"),
                terminated,
            )

        hand5 = [c for c in observation.get("my_cards", []) if c != -1]
        if len(hand5) != 5:
            # Fallback if cards not ready
            if valid_actions[at.CHECK.value]:
                return at.CHECK.value, 0, 0, 1
            if valid_actions[at.CALL.value]:
                return at.CALL.value, 0, 0, 1
            return at.FOLD.value, 0, 0, 1

        opponent_profile = to_opponent_profile(self.recon)
        score, _ = final_street0_score(
            hand5,
            opponent_profile=opponent_profile,
            n_flop_samples=DEFAULT_N_FLOP_SAMPLES,
            n_tr_samples=DEFAULT_N_TR_SAMPLES,
        )

        action_type, raise_amount = get_street0_action_from_recon(
            score, observation, self.recon
        )

        if action_type == at.RAISE.value:
            record_our_bet(self.recon, street)

        amount = raise_amount if action_type == at.RAISE.value else 0
        return action_type, amount, 0, 1
