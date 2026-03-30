"""
Genesis Agent – main player implementation.

Street 0: Preflop via street0_score and street0_bet_sizing + opponent_recon.
Street 1A: Discard via street1a_discard.
Street 1B: Flop betting (multi-round) via street1b_betting.
Street 2: Turn via street2_turn.
Street 3: River via street3_river.
Street > 3: Pass-through (CHECK/CALL/FOLD as fallback).
"""

import os
import sys

# Ensure "other bots" is on path so "support" is findable when this module is loaded
_other_bots_root = os.path.abspath(os.path.dirname(__file__))
if _other_bots_root not in sys.path:
    sys.path.insert(0, _other_bots_root)

from agents.agent import Agent
from gym_env import PokerEnv

from support.street0_score import final_street0_score, classify_board_texture
from support.street0_bet_sizing import get_street0_action_from_recon
from support.opponent_recon import (
    OpponentRecon,
    start_new_hand,
    start_new_street,
    record_our_bet,
    update_opponent_actions,
    update_vpip_pfr,
    update_aggression_flags,
    update_fold_on_terminate,
    to_opponent_profile,
    get_vpip,
    get_pfr,
    get_aggression_factor,
    get_fold_to_non_river_bet,
    get_fold_to_river_bet,
    get_non_river_bet_percentage,
    get_opponent_type,
    get_river_bet_when_checked_to,
    # Street 1 recon APIs
    record_our_flop_discard_class,
    record_our_flop_bet,
    record_flop_texture,
    classify_opponent_flop_discard,
    update_opponent_flop_response,
    update_opponent_flop_aggression,
    _is_betting_action,
    # Street 2 recon APIs
    record_our_turn_bet,
    record_turn_texture,
    set_flop_opp_action,
    update_opponent_turn_response,
    # Street 3 recon APIs
    record_our_river_bet,
    record_river_texture,
    update_opponent_river_response,
    update_opponent_river_aggression,
)
from support.street1a_discard import (
    Street1AContext,
    choose_keep_with_controlled_mixing,
    classify_flop_texture,
)
from support.street1b_betting import (
    Street1BContext,
    get_street1b_action,
)
from support.street2_turn import (
    Street2Context,
    get_street2_action,
    classify_turn_texture,
)
from support.street3_river import (
    Street3Context,
    get_street3_action,
)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_N_FLOP_SAMPLES = 150
DEFAULT_N_TR_SAMPLES = 50

STREET0_LOG_PREFIX = "STREET0"
STREET1_LOG_PREFIX = "STREET1"
STREET2_LOG_PREFIX = "STREET2"
STREET3_LOG_PREFIX = "STREET3"
OPP_RECON_LOG_PREFIX = "OPP_RECON"
HAND_RESULT_LOG_PREFIX = "HAND_RESULT"


# ── GenesisAgent ──────────────────────────────────────────────────────────────


class GenesisAgent(Agent):
    def __init__(self, stream: bool = True, entry_point: str = "genesis"):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.recon = OpponentRecon()
        self._entry_point = entry_point

        # Per-hand state for Street 1
        self._street1_discard_class = ""
        self._street1_flop_texture = ""
        self._street1_size_bucket = ""
        self._street1_opp_discard_classified = False
        self._street1b_round_count = 0
        self._last_hand_number = -1
        # Flop line summary (set during Street 1B)
        self._flop_we_checked = False
        self._flop_we_bet_bucket = ""
        self._flop_line_summary = ""
        # Street 2 per-hand state
        self._street2_size_bucket = ""
        self._street2_round_count = 0
        # Turn opponent action (set at start of each _act_street2)
        self._turn_opp_action = ""
        # Street 3 (river) per-hand state
        self._river_we_checked = False
        self._river_size_bucket = ""

        # Running chip total across the match (updated on each hand end in observe)
        self._running_chips = 0.0

    def __name__(self):
        return "GenesisAgent"

    # ── observe() ─────────────────────────────────────────────────────────────

    def observe(self, observation, reward, terminated, truncated, info):
        """Called when we receive an observation (non-acting player)."""
        street = observation.get("street", -1)
        opp_last_action = observation.get("opp_last_action", "") or ""

        # Street 3 recon: when we see opponent's river action (we had bet or we had checked)
        if street == 3 and _is_betting_action(opp_last_action):
            if self.recon.our_river_size_bucket:
                update_opponent_river_response(self.recon, opp_last_action, self._street1_discard_class)
            elif self._river_we_checked:
                update_opponent_river_aggression(self.recon, True, opp_last_action)

        if terminated:
            update_fold_on_terminate(self.recon, street, opp_last_action, terminated)
            hand_number = info.get("hand_number", -1)
            entry = self._entry_point

            if street == 0:
                self.logger.info(
                    f"{STREET0_LOG_PREFIX} | hand_end | entry={entry} | hand={hand_number}"
                    f" | street=0 | reward={reward} | opp_last={opp_last_action!r}"
                )
            elif street >= 1 and self._street1_discard_class:
                self.logger.info(
                    f"{STREET1_LOG_PREFIX} | hand_end | entry={entry} | hand={hand_number}"
                    f" | street={street} | reward={reward}"
                    f" | our_discard_class={self._street1_discard_class}"
                    f" | opp_discard_class={self.recon.opp_flop_discard_class}"
                    f" | texture={self._street1_flop_texture}"
                )
            if street >= 2:
                turn_tex = getattr(self.recon, "turn_texture_this_hand", "") or ""
                self.logger.info(
                    f"{STREET2_LOG_PREFIX} | hand_end | entry={entry} | hand={hand_number}"
                    f" | street={street} | reward={reward}"
                    f" | our_discard_class={self._street1_discard_class}"
                    f" | opp_discard_class={self.recon.opp_flop_discard_class}"
                    f" | turn_texture={turn_tex} | flop_line={self._flop_line_summary or ''}"
                )
            if street >= 3:
                river_tex = getattr(self.recon, "river_texture_this_hand", "") or ""
                self.logger.info(
                    f"{STREET3_LOG_PREFIX} | hand_end | entry={entry} | hand={hand_number}"
                    f" | street={street} | reward={reward}"
                    f" | river_texture={river_tex} | flop_line={self._flop_line_summary or ''}"
                    f" | turn_we_bet={getattr(self, '_street2_size_bucket', '') != ''}"
                )
            # Opponent recon snapshot after each hand (how opponent data evolves over the match)
            self._log_opponent_recon_snapshot(hand_number, entry)
            # End type: they_fold / we_fold / showdown / showdown_tie
            street_ended = street
            if street_ended <= 3:
                end_type = "they_fold" if (opp_last_action or "").strip().upper() == "FOLD" else "we_fold"
            else:
                end_type = "showdown_tie" if reward == 0 else "showdown"
            self._running_chips += reward
            self._log_hand_result(hand_number, entry, reward, observation, info, end_type, street_ended, truncated)
            return

        # Non-terminal observe on street 1: update flop response/aggression recon
        if street == 1 and _is_betting_action(opp_last_action):
            update_opponent_flop_response(self.recon, opp_last_action)
            update_opponent_flop_aggression(self.recon, opp_last_action)
        # Non-terminal observe on street 2: update turn response recon
        if street == 2 and _is_betting_action(opp_last_action):
            update_opponent_turn_response(self.recon, opp_last_action, self._street1_discard_class)

    # ── act() and street dispatch ─────────────────────────────────────────────

    def act(self, observation, reward, terminated, truncated, info):
        """Main decision entry; returns (action_type, raise_amount, keep_card_1, keep_card_2)."""
        street = observation.get("street", 0)
        valid_actions = observation.get("valid_actions", [1, 1, 1, 1, 0])
        at = self.action_types

        if len(valid_actions) < 5:
            valid_actions = list(valid_actions) + [0] * (5 - len(valid_actions))
        else:
            valid_actions = list(valid_actions)[:5]

        hand_number = info.get("hand_number", 0)

        # --- New hand / new street detection ---
        if hand_number != self._last_hand_number:
            self._last_hand_number = hand_number
            start_new_hand(self.recon, hand_number)
            self._street1_discard_class = ""
            self._street1_flop_texture = ""
            self._street1_size_bucket = ""
            self._street1_opp_discard_classified = False
            self._street1b_round_count = 0
            self._flop_we_checked = False
            self._flop_we_bet_bucket = ""
            self._flop_line_summary = ""
            self._street2_size_bucket = ""
            self._street2_round_count = 0
            self._turn_opp_action = ""
            self._river_we_checked = False
            self._river_size_bucket = ""

        if street != self.recon.last_street:
            start_new_street(self.recon, street)

        # --- Standard recon updates (every act() call) ---
        opp_last_action = observation.get("opp_last_action", "") or ""
        update_opponent_actions(self.recon, opp_last_action)
        update_vpip_pfr(self.recon, observation, street)
        update_aggression_flags(self.recon, observation, street)

        # Street 1B-specific recon: catch opponent flop responses
        if (street == 1 and valid_actions[at.DISCARD.value] == 0
                and _is_betting_action(opp_last_action)):
            update_opponent_flop_response(self.recon, opp_last_action)
            update_opponent_flop_aggression(self.recon, opp_last_action)

        # Street 2 recon: catch opponent turn responses
        if street == 2 and _is_betting_action(opp_last_action):
            update_opponent_turn_response(self.recon, opp_last_action, self._street1_discard_class)

        # Street 3 recon: catch opponent river response (before terminated so we record then log hand_end)
        if street == 3 and _is_betting_action(opp_last_action):
            if self.recon.our_river_size_bucket:
                update_opponent_river_response(self.recon, opp_last_action, self._street1_discard_class)
            elif self._river_we_checked:
                update_opponent_river_aggression(self.recon, True, opp_last_action)

        if terminated:
            update_fold_on_terminate(self.recon, street, opp_last_action, terminated)

        # --- Street 2: Turn engine ---
        if street == 2:
            if valid_actions[at.DISCARD.value]:
                return at.DISCARD.value, 0, 0, 1
            return self._act_street2(observation, info, valid_actions)

        # --- Street 3: River engine ---
        if street == 3:
            if valid_actions[at.DISCARD.value]:
                return at.DISCARD.value, 0, 0, 1
            return self._act_street3(observation, info, valid_actions)

        # --- Street > 3: pass-through (future) ---
        if street > 3:
            if valid_actions[at.DISCARD.value]:
                return at.DISCARD.value, 0, 0, 1
            if valid_actions[at.CHECK.value]:
                return at.CHECK.value, 0, 0, 1
            if valid_actions[at.CALL.value]:
                return at.CALL.value, 0, 0, 1
            return at.FOLD.value, 0, 0, 1

        # --- Street 1A: Discard ---
        if street == 1 and valid_actions[at.DISCARD.value] == 1:
            return self._act_street1a(observation, info, valid_actions)

        # --- Street 1B: Flop Betting ---
        if street == 1 and valid_actions[at.DISCARD.value] == 0:
            return self._act_street1b(observation, info, valid_actions)

        # --- Street 0: Full preflop strategy ---
        return self._act_street0(observation, info, valid_actions, terminated)

    # ── Street 0: Preflop ─────────────────────────────────────────────────────

    def _act_street0(self, observation, info, valid_actions, terminated):
        at = self.action_types
        street = observation.get("street", 0)

        hand5 = [c for c in observation.get("my_cards", []) if c != -1]
        if len(hand5) != 5:
            if valid_actions[at.CHECK.value]:
                return at.CHECK.value, 0, 0, 1
            if valid_actions[at.CALL.value]:
                return at.CALL.value, 0, 0, 1
            return at.FOLD.value, 0, 0, 1

        opponent_profile = to_opponent_profile(self.recon)
        # Street 0 scoring can use multiple workers; set POKER_N_WORKERS=2 (default) or 4 for next phase.
        score, breakdown = final_street0_score(
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

        self._log_street0(
            hand_number=info.get("hand_number", 0),
            hand5=hand5,
            score=score,
            breakdown=breakdown,
            action_type=action_type,
            raise_amount=raise_amount,
            observation=observation,
        )

        amount = raise_amount if action_type == at.RAISE.value else 0
        return action_type, amount, 0, 1

    # ── Street 1A: Discard ────────────────────────────────────────────────────

    def _act_street1a(self, observation, info, valid_actions):
        at = self.action_types
        hand_number = info.get("hand_number", 0)

        hand5 = [c for c in observation.get("my_cards", []) if c != -1]
        flop3 = [c for c in observation.get("community_cards", []) if c != -1]
        position = observation.get("blind_position", 1)  # 0=SB, 1=BB

        # SB sees opponent discards; BB does not
        if position == 0:
            opp_discard3 = [c for c in observation.get("opp_discarded_cards", []) if c != -1]
            if not opp_discard3:
                opp_discard3 = None
        else:
            opp_discard3 = None

        # Compute flop texture
        texture = classify_flop_texture(flop3) if len(flop3) >= 3 else "unknown"

        # Build context
        opponent_profile = to_opponent_profile(self.recon)
        ctx = Street1AContext(
            hand5=hand5,
            flop3=flop3,
            position=position,
            opp_discard3=opp_discard3,
            pot_size=observation.get("my_bet", 0) + observation.get("opp_bet", 0),
            opponent_profile=opponent_profile,
        )

        ki0, ki1, chosen_keep, chosen_discard, discard_class, records = \
            choose_keep_with_controlled_mixing(hand5, flop3, ctx)

        # Store per-hand state
        self._street1_discard_class = discard_class
        self._street1_flop_texture = texture

        # Record for recon
        record_our_flop_discard_class(self.recon, discard_class)
        record_flop_texture(self.recon, texture)

        # Log
        self._log_street1a(
            hand_number=hand_number,
            position=position,
            hand5=hand5,
            flop3=flop3,
            opp_discard3=opp_discard3,
            chosen_keep=chosen_keep,
            chosen_discard=chosen_discard,
            discard_class=discard_class,
            records=records,
        )

        return at.DISCARD.value, 0, ki0, ki1

    # ── Street 1B: Flop ───────────────────────────────────────────────────────

    def _act_street1b(self, observation, info, valid_actions):
        at = self.action_types
        hand_number = info.get("hand_number", 0)

        our_keep2 = [c for c in observation.get("my_cards", []) if c != -1]
        our_discard3 = [c for c in observation.get("my_discarded_cards", []) if c != -1]
        opp_discard3 = [c for c in observation.get("opp_discarded_cards", []) if c != -1]
        community = [c for c in observation.get("community_cards", []) if c != -1]
        flop3 = community[:3] if len(community) >= 3 else community

        # Classify opponent discard once per hand
        if not self._street1_opp_discard_classified and opp_discard3:
            classify_opponent_flop_discard(
                self.recon, opp_discard3, flop3, self._street1_flop_texture
            )
            self._street1_opp_discard_classified = True

        self._street1b_round_count += 1

        opp_last_action = observation.get("opp_last_action", "") or ""
        set_flop_opp_action(self.recon, opp_last_action)
        my_bet = observation.get("my_bet", 0)
        opp_bet = observation.get("opp_bet", 0)
        position = observation.get("blind_position", 1)

        # Attach recon reference so P and X components can query it
        opponent_profile = to_opponent_profile(self.recon)
        opponent_profile._recon_ref = self.recon

        ctx = Street1BContext(
            our_keep2=our_keep2,
            our_discard3=our_discard3,
            opp_discard3=opp_discard3,
            flop3=flop3,
            pot_size=my_bet + opp_bet,
            amount_to_call=max(0, opp_bet - my_bet),
            valid_actions=valid_actions,
            min_raise=int(observation.get("min_raise", 2)),
            max_raise=int(observation.get("max_raise", 0)),
            opponent_profile=opponent_profile,
            our_discard_class=self._street1_discard_class,
            opp_discard_class=self.recon.opp_flop_discard_class,
            flop_texture=self._street1_flop_texture,
            opp_last_action=opp_last_action,
            position=position,
        )

        action_type, raise_amount, size_bucket, bd = get_street1b_action(ctx)

        if action_type == at.RAISE.value:
            record_our_bet(self.recon, 1)
            if not self._street1_size_bucket:
                record_our_flop_bet(self.recon, size_bucket)
                self._street1_size_bucket = size_bucket
            self._flop_we_checked = False
            self._flop_we_bet_bucket = size_bucket
        else:
            self._flop_we_checked = True
            self._flop_we_bet_bucket = ""
        self._flop_line_summary = self._build_flop_line_summary()

        # Log
        self._log_street1b(
            hand_number=hand_number,
            bd=bd,
            action_type=action_type,
            raise_amount=raise_amount,
            size_bucket=size_bucket,
            observation=observation,
        )

        amount = raise_amount if action_type == at.RAISE.value else 0
        return action_type, amount, 0, 1

    def _build_flop_line_summary(self):
        """Build flop line string for Street 2, e.g. we_checked_opp_checked, we_bet_small_opp_called."""
        we = "we_checked" if self._flop_we_checked else f"we_bet_{self._flop_we_bet_bucket}" if self._flop_we_bet_bucket else "we_checked"
        opp = (getattr(self.recon, "_flop_opp_action", "") or "unknown").replace(" ", "_")
        return f"{we}_opp_{opp}"

    # ── Street 2: Turn ─────────────────────────────────────────────────────────

    def _act_street2(self, observation, info, valid_actions):
        at = self.action_types
        hand_number = info.get("hand_number", 0)
        opp_last_action = observation.get("opp_last_action", "") or ""
        al = opp_last_action.lower()
        if "raise" in al:
            self._turn_opp_action = "raise"
        elif "call" in al:
            self._turn_opp_action = "call"
        elif "bet" in al:
            self._turn_opp_action = "bet"
        elif "check" in al:
            self._turn_opp_action = "check"
        else:
            self._turn_opp_action = ""

        our_keep2 = [c for c in observation.get("my_cards", []) if c != -1]
        our_discard3 = [c for c in observation.get("my_discarded_cards", []) if c != -1]
        opp_discard3 = [c for c in observation.get("opp_discarded_cards", []) if c != -1]
        community = [c for c in observation.get("community_cards", []) if c != -1]
        board4 = community[:4] if len(community) >= 4 else community
        flop3 = community[:3] if len(community) >= 3 else []
        turn_card = community[3] if len(community) >= 4 else -1

        my_bet = observation.get("my_bet", 0)
        opp_bet = observation.get("opp_bet", 0)
        pot_size = my_bet + opp_bet
        amount_to_call = max(0, opp_bet - my_bet)
        position = observation.get("blind_position", 1)
        opp_last_action = observation.get("opp_last_action", "") or ""

        turn_texture = classify_turn_texture(board4) if board4 else "blank_turn"
        record_turn_texture(self.recon, turn_texture)

        self._street2_round_count += 1
        opponent_profile = to_opponent_profile(self.recon)
        opponent_profile._recon_ref = self.recon

        ctx = Street2Context(
            our_keep2=our_keep2,
            our_discard3=our_discard3,
            opp_discard3=opp_discard3,
            flop3=flop3,
            turn_card=turn_card,
            board4=board4,
            pot_size=pot_size,
            amount_to_call=amount_to_call,
            valid_actions=valid_actions,
            min_raise=int(observation.get("min_raise", 2)),
            max_raise=int(observation.get("max_raise", 0)),
            opponent_profile=opponent_profile,
            recon=self.recon,
            our_discard_class=self._street1_discard_class,
            opp_discard_class=self.recon.opp_flop_discard_class,
            flop_texture=self._street1_flop_texture,
            turn_texture=turn_texture,
            flop_line=self._flop_line_summary or "",
            opp_last_action=opp_last_action,
            position=position,
        )

        action_type, raise_amount, size_bucket, breakdown = get_street2_action(ctx)

        if action_type == at.RAISE.value:
            record_our_bet(self.recon, 2)
            if not self._street2_size_bucket:
                record_our_turn_bet(self.recon, size_bucket)
                self._street2_size_bucket = size_bucket

        self._log_street2(
            hand_number=hand_number,
            breakdown=breakdown,
            action_type=action_type,
            raise_amount=raise_amount,
            size_bucket=size_bucket,
            observation=observation,
        )

        amount = raise_amount if action_type == at.RAISE.value else 0
        return action_type, amount, 0, 1

    # ── Street 3: River ────────────────────────────────────────────────────────

    def _act_street3(self, observation, info, valid_actions):
        at = self.action_types
        hand_number = info.get("hand_number", 0)

        our_keep2 = [c for c in observation.get("my_cards", []) if c != -1]
        community = [c for c in observation.get("community_cards", []) if c != -1]
        board5 = community[:5] if len(community) >= 5 else community

        my_bet = observation.get("my_bet", 0)
        opp_bet = observation.get("opp_bet", 0)
        pot_size = my_bet + opp_bet
        amount_to_call = max(0, opp_bet - my_bet)
        position = observation.get("blind_position", 1)
        opp_last_action = observation.get("opp_last_action", "") or ""

        river_texture = classify_board_texture(board5) if len(board5) >= 5 else ""
        if not self.recon.river_texture_this_hand and river_texture:
            record_river_texture(self.recon, river_texture)

        turn_we_bet = bool(self._street2_size_bucket)

        ctx = Street3Context(
            our_keep2=our_keep2,
            board5=board5,
            pot_size=pot_size,
            amount_to_call=amount_to_call,
            valid_actions=valid_actions,
            min_raise=int(observation.get("min_raise", 2)),
            max_raise=int(observation.get("max_raise", 0)),
            position=position,
            recon=self.recon,
            flop_line=self._flop_line_summary or "",
            turn_we_bet=turn_we_bet,
            turn_opp_action=self._turn_opp_action or "",
            opp_last_action=opp_last_action,
            river_texture=river_texture,
            our_discard_class=self._street1_discard_class,
            opp_discard_class=self.recon.opp_flop_discard_class,
        )

        action_type, raise_amount, size_bucket, breakdown = get_street3_action(ctx)

        if action_type == at.RAISE.value:
            record_our_bet(self.recon, 3)
            if not self._river_size_bucket:
                record_our_river_bet(self.recon, size_bucket)
                self._river_size_bucket = size_bucket
            self._river_we_checked = False
        else:
            self._river_we_checked = True

        self._log_street3(
            hand_number=hand_number,
            breakdown=breakdown,
            action_type=action_type,
            raise_amount=raise_amount,
            size_bucket=size_bucket,
            observation=observation,
        )

        amount = raise_amount if action_type == at.RAISE.value else 0
        return action_type, amount, 0, 1

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_opponent_recon_snapshot(self, hand_number, entry):
        """Log opponent recon data after each hand so we can see how it evolves over the match."""
        r = self.recon
        parts = [
            OPP_RECON_LOG_PREFIX,
            f"entry={entry}",
            f"hand={hand_number}",
            f"total_hands={r.total_hands}",
            f"vpip={get_vpip(r):.3f}",
            f"vpip_n={r.opp_vpip_count}",
            f"pfr={get_pfr(r):.3f}",
            f"pfr_n={r.opp_pfr_count}",
            f"af={get_aggression_factor(r):.3f}",
            f"raise_n={r.opp_raise_count}",
            f"call_n={r.opp_call_count}",
            f"fold_non_river={get_fold_to_non_river_bet(r):.3f}",
            f"non_river_faced={r.opp_non_river_bets_faced}",
            f"non_river_folds={r.opp_non_river_folds}",
            f"fold_river={get_fold_to_river_bet(r):.3f}",
            f"river_faced={r.opp_river_bets_faced}",
            f"river_folds={r.opp_river_folds}",
            f"non_river_bet_pct={get_non_river_bet_percentage(r):.3f}",
            f"streets_seen={r.opp_non_river_streets_seen}",
            f"non_river_bet_n={r.opp_non_river_bet_count}",
            f"opp_type={get_opponent_type(r)}",
            f"river_bet_when_checked={get_river_bet_when_checked_to(r):.3f}",
            f"river_we_checked_opp_bet={r.river_we_checked_opp_bet_count}",
            f"river_we_checked_opp_check={r.river_we_checked_opp_check_count}",
        ]
        self.logger.info(" | ".join(parts))

    def _log_hand_result(self, hand_number, entry, reward, observation, info, end_type, street_ended, truncated):
        """Log one pipe-separated HAND_RESULT line per hand for loss diagnostics and tuning."""
        my_bet = int(observation.get("my_bet", 0) or 0)
        opp_bet = int(observation.get("opp_bet", 0) or 0)
        pot = my_bet + opp_bet
        blind_pos = observation.get("blind_position", -1)
        position = "SB" if blind_pos == 0 else "BB" if blind_pos == 1 else str(blind_pos)
        opp_last_action = (observation.get("opp_last_action") or "").strip()
        invalid_action = bool(info.get("invalid_action", False))
        is_showdown = street_ended > 3
        won = reward > 0
        lost = reward < 0
        tie = reward == 0

        def card_str(c):
            if c is None or c == -1:
                return None
            return PokerEnv.int_card_to_str(int(c))

        my_cards_raw = observation.get("my_cards", []) or []
        my_cards_str = " ".join(card_str(c) for c in my_cards_raw if card_str(c))
        community = observation.get("community_cards", []) or []
        board_str = " ".join(card_str(c) for c in community if card_str(c))

        flop_tex = getattr(self, "_street1_flop_texture", "") or ""
        turn_tex = getattr(self.recon, "turn_texture_this_hand", None) or ""
        river_tex = getattr(self.recon, "river_texture_this_hand", None) or ""
        flop_line = getattr(self, "_flop_line_summary", "") or ""
        flop_bucket = getattr(self, "_flop_we_bet_bucket", "") or ""
        flop_rounds = getattr(self, "_street1b_round_count", 0)
        turn_we_bet = bool(getattr(self, "_street2_size_bucket", "") or "")
        turn_bucket = getattr(self, "_street2_size_bucket", "") or ""
        turn_rounds = getattr(self, "_street2_round_count", 0)
        river_we_bet = bool(getattr(self, "_river_size_bucket", "") or "")
        river_bucket = getattr(self, "_river_size_bucket", "") or ""
        our_discard = getattr(self, "_street1_discard_class", "") or ""
        opp_discard = getattr(self.recon, "opp_flop_discard_class", "") or ""
        time_used = observation.get("time_used", 0)
        time_left = observation.get("time_left", 0)

        parts = [
            HAND_RESULT_LOG_PREFIX,
            f"entry={entry}",
            f"hand={hand_number}",
            f"reward={reward}",
            f"won={won}",
            f"lost={lost}",
            f"tie={tie}",
            f"running_chips={self._running_chips:.1f}",
            f"street_ended={street_ended}",
            f"is_showdown={is_showdown}",
            f"end_type={end_type}",
            f"invalid_action={invalid_action}",
            f"position={position}",
            f"pot={pot}",
            f"my_bet={my_bet}",
            f"opp_bet={opp_bet}",
            f"our_discard_class={our_discard}",
            f"flop_line={flop_line}",
            f"flop_bet_bucket={flop_bucket}",
            f"flop_rounds={flop_rounds}",
            f"turn_we_bet={turn_we_bet}",
            f"turn_bet_bucket={turn_bucket}",
            f"turn_rounds={turn_rounds}",
            f"river_we_bet={river_we_bet}",
            f"river_bet_bucket={river_bucket}",
            f"flop_texture={flop_tex}",
            f"turn_texture={turn_tex}",
            f"river_texture={river_tex}",
            f"opp_last_action={opp_last_action!r}",
            f"opp_type={get_opponent_type(self.recon)}",
            f"opp_discard_class={opp_discard}",
            f"my_cards={my_cards_str}",
            f"board={board_str}",
            f"time_used={time_used}",
            f"time_left={time_left}",
            f"truncated={truncated}",
        ]
        self.logger.info(" | ".join(parts))

    def _log_street0(self, hand_number, hand5, score, breakdown,
                     action_type, raise_amount, observation):
        at = self.action_types
        action_name = at(action_type).name if action_type in range(5) else "UNKNOWN"
        my_bet = observation.get("my_bet", 0)
        opp_bet = observation.get("opp_bet", 0)
        amount_to_call = opp_bet - my_bet
        pot_size = my_bet + opp_bet
        blind_position = observation.get("blind_position", -1)
        opp_last_action = observation.get("opp_last_action", "") or ""

        s_base_val = (
            f"{breakdown.s_base:.4f}"
            if hasattr(breakdown, "s_base") and breakdown.s_base is not None
            else "n/a"
        )
        entry = self._entry_point
        parts = [
            STREET0_LOG_PREFIX,
            f"entry={entry}",
            f"hand={hand_number}",
            "street=0",
            f"hand5={hand5}",
            f"score={score:.4f}",
            f"s_base={s_base_val}",
            f"action={action_name}",
            f"raise_amt={raise_amount}",
            f"my_bet={my_bet}",
            f"opp_bet={opp_bet}",
            f"to_call={amount_to_call}",
            f"pot={pot_size}",
            f"blind_pos={blind_position}",
            f"vpip={get_vpip(self.recon):.3f}",
            f"pfr={get_pfr(self.recon):.3f}",
            f"af={get_aggression_factor(self.recon):.3f}",
            f"fold_non_river={get_fold_to_non_river_bet(self.recon):.3f}",
            f"fold_river={get_fold_to_river_bet(self.recon):.3f}",
            f"non_river_bet_pct={get_non_river_bet_percentage(self.recon):.3f}",
            f"total_hands={self.recon.total_hands}",
            f"opp_last={opp_last_action!r}",
        ]
        self.logger.info(" | ".join(parts))

    def _log_street1a(self, hand_number, position, hand5, flop3,
                      opp_discard3, chosen_keep, chosen_discard,
                      discard_class, records):
        entry = self._entry_point
        pos_label = "SB" if position == 0 else "BB"

        top2 = sorted(records, key=lambda r: r.S1A, reverse=True)[:2]
        top2_str = ";".join(f"{r.S1A:.4f}" for r in top2)

        best = top2[0] if top2 else None
        parts = [
            STREET1_LOG_PREFIX,
            "1A",
            f"entry={entry}",
            f"hand={hand_number}",
            "street=1",
            f"position={pos_label}",
            f"hand5={hand5}",
            f"flop={flop3}",
            f"opp_discard={opp_discard3}",
            f"chosen_keep={chosen_keep}",
            f"chosen_discard={chosen_discard}",
            f"discard_class={discard_class}",
        ]
        if best:
            parts += [
                f"S1A={best.S1A:.4f}",
                f"C={best.C:.4f}",
                f"F={best.F:.4f}",
                f"D={best.D:.4f}",
                f"R={best.R:.4f}",
                f"A={best.A:.4f}",
                f"I={best.I_val:.4f}",
                f"B={best.B:.4f}",
            ]
        parts.append(f"top2_S1A={top2_str}")
        self.logger.info(" | ".join(parts))

    def _log_street1b(self, hand_number, bd, action_type, raise_amount,
                      size_bucket, observation):
        at = self.action_types
        action_name = at(action_type).name if action_type in range(5) else "UNKNOWN"
        my_bet = observation.get("my_bet", 0)
        opp_bet = observation.get("opp_bet", 0)
        opp_last_action = observation.get("opp_last_action", "") or ""
        entry = self._entry_point

        parts = [
            STREET1_LOG_PREFIX,
            "1B",
            f"entry={entry}",
            f"hand={hand_number}",
            "street=1",
            f"round={self._street1b_round_count}",
            f"S1B={bd.S1B:.4f}",
            f"H={bd.H:.4f}",
            f"F_prime={bd.F_prime:.4f}",
            f"E={bd.E:.4f}",
            f"T={bd.T:.4f}",
            f"P={bd.P:.4f}",
            f"X={bd.X:.4f}",
            f"action={action_name}",
            f"raise_amt={raise_amount}",
            f"size_bucket={size_bucket}",
            f"to_call={max(0, opp_bet - my_bet)}",
            f"pot={my_bet + opp_bet}",
            f"our_discard_class={self._street1_discard_class}",
            f"opp_discard_class={self.recon.opp_flop_discard_class}",
            f"texture={self._street1_flop_texture}",
            f"opp_last={opp_last_action!r}",
        ]
        self.logger.info(" | ".join(parts))

    def _log_street2(self, hand_number, breakdown, action_type, raise_amount,
                     size_bucket, observation):
        at = self.action_types
        action_name = at(action_type).name if action_type in range(5) else "UNKNOWN"
        my_bet = observation.get("my_bet", 0)
        opp_bet = observation.get("opp_bet", 0)
        opp_last_action = observation.get("opp_last_action", "") or ""
        entry = self._entry_point
        parts = [
            STREET2_LOG_PREFIX,
            f"entry={entry}",
            f"hand={hand_number}",
            "street=2",
            f"round={self._street2_round_count}",
            f"S2={breakdown.S2:.4f}",
            f"H={breakdown.H:.4f}",
            f"Rv={breakdown.Rv:.4f}",
            f"E={breakdown.E:.4f}",
            f"T={breakdown.T:.4f}",
            f"P={breakdown.P:.4f}",
            f"L={breakdown.L:.4f}",
            f"X={breakdown.X:.4f}",
            f"action={action_name}",
            f"raise_amt={raise_amount}",
            f"size_bucket={size_bucket}",
            f"to_call={max(0, opp_bet - my_bet)}",
            f"pot={my_bet + opp_bet}",
            f"our_discard_class={self._street1_discard_class}",
            f"opp_discard_class={self.recon.opp_flop_discard_class}",
            f"flop_texture={self._street1_flop_texture}",
            f"turn_texture={getattr(self.recon, 'turn_texture_this_hand', '') or ''}",
            f"flop_line={self._flop_line_summary or ''}",
            f"opp_last={opp_last_action!r}",
        ]
        self.logger.info(" | ".join(parts))

    def _log_street3(self, hand_number, breakdown, action_type, raise_amount,
                     size_bucket, observation):
        at = self.action_types
        action_name = at(action_type).name if action_type in range(5) else "UNKNOWN"
        my_bet = observation.get("my_bet", 0)
        opp_bet = observation.get("opp_bet", 0)
        opp_last_action = observation.get("opp_last_action", "") or ""
        entry = self._entry_point
        river_tex = getattr(self.recon, "river_texture_this_hand", "") or ""
        parts = [
            STREET3_LOG_PREFIX,
            f"entry={entry}",
            f"hand={hand_number}",
            "street=3",
            f"R3={breakdown.R3:.4f}",
            f"H3={breakdown.H3:.4f}",
            f"C3={breakdown.C3:.4f}",
            f"L3={breakdown.L3:.4f}",
            f"X3={breakdown.X3:.4f}",
            f"action={action_name}",
            f"raise_amt={raise_amount}",
            f"size_bucket={size_bucket}",
            f"to_call={max(0, opp_bet - my_bet)}",
            f"pot={my_bet + opp_bet}",
            f"river_texture={river_tex}",
            f"flop_line={self._flop_line_summary or ''}",
            f"turn_we_bet={bool(self._street2_size_bucket)}",
            f"turn_opp={self._turn_opp_action or ''}",
            f"opp_last={opp_last_action!r}",
        ]
        self.logger.info(" | ".join(parts))
