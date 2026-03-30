"""
Game model for 27-card poker: public sequences, infoset keys, and action encoding.

Represents the game as an extensive-form structure: streets (0-3), discard round
on street 1, and betting sequences. Defines "public sequence" (actions visible to
both, excluding private cards) and information set keys for CFR.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

# Action type values aligned with gym_env.PokerEnv.ActionType
FOLD = 0
RAISE = 1
CHECK = 2
CALL = 3
DISCARD = 4

DECK_SIZE = 27
NUM_STREETS = 4
SMALL_BLIND = 1
BIG_BLIND = 2
MAX_BET = 100
NUM_PLAYER_CARDS = 5
NUM_COMMUNITY_CARDS = 5


@dataclass(frozen=True)
class ActionRecord:
    """Single action in the public sequence (for history tracking and infoset keying)."""
    player: int  # 0 or 1
    action_type: int  # FOLD, RAISE, CHECK, CALL, DISCARD
    raise_amount: int = 0  # total bet after raise (for RAISE)
    keep_card_1: int = -1
    keep_card_2: int = -1

    def to_public_string(self) -> str:
        """Encode for public sequence (no private card info)."""
        if self.action_type == FOLD:
            return f"P{self.player}:F"
        if self.action_type == CALL:
            return f"P{self.player}:C"
        if self.action_type == CHECK:
            return f"P{self.player}:X"
        if self.action_type == RAISE:
            return f"P{self.player}:R{self.raise_amount}"
        if self.action_type == DISCARD:
            return f"P{self.player}:D{self.keep_card_1},{self.keep_card_2}"
        return f"P{self.player}:?"


class PublicSequence:
    """
    Sequence of public actions (and street boundaries) that define the game path.
    Used to build infoset keys and to match observation state to the game tree.
    """
    __slots__ = ("_actions", "_street", "_discard_done")

    def __init__(
        self,
        actions: Optional[List[ActionRecord]] = None,
        street: int = 0,
        discard_done: bool = False,
    ):
        self._actions: List[ActionRecord] = list(actions) if actions else []
        self._street = street
        self._discard_done = discard_done

    def copy(self) -> "PublicSequence":
        return PublicSequence(
            actions=list(self._actions),
            street=self._street,
            discard_done=self._discard_done,
        )

    def append(self, action: ActionRecord) -> None:
        self._actions.append(action)
        if action.action_type == DISCARD:
            self._discard_done = True
        elif action.action_type in (CALL, CHECK) or action.action_type == FOLD:
            # Street may end; caller can advance street
            pass
        elif action.action_type == RAISE:
            pass

    def advance_street(self) -> None:
        self._street += 1

    @property
    def street(self) -> int:
        return self._street

    @property
    def discard_done(self) -> bool:
        return self._discard_done

    @property
    def actions(self) -> List[ActionRecord]:
        return self._actions

    def to_key(self) -> str:
        """String key for this public sequence (for strategy lookup)."""
        parts = [f"S{self._street}", f"DD{1 if self._discard_done else 0}"]
        for a in self._actions:
            parts.append(a.to_public_string())
        return "|".join(parts)


def infoset_key(
    player: int,
    public_sequence: PublicSequence,
    private_hand: Union[Tuple[int, ...], int],
) -> str:
    """
    Build information set key for CFR: (player, public sequence, private hand/bucket).

    - Preflop: private_hand = tuple of 5 card indices (or bucket id int).
    - Post-discard (flop/turn/river): after discard, both hands are known so we use
      full state; for abstraction we can use bucket id (int) or hand tuple (2 cards).
    """
    pub = public_sequence.to_key()
    if isinstance(private_hand, (list, tuple)):
        hand_part = ",".join(str(c) for c in private_hand)
    else:
        hand_part = str(private_hand)
    return f"I{player}|{pub}|H{hand_part}"


def action_record_from_obs(
    player: int,
    action_type: int,
    raise_amount: int,
    keep_card_1: int,
    keep_card_2: int,
    prev_my_bet: int,
    prev_opp_bet: int,
) -> ActionRecord:
    """Build ActionRecord from agent action. For RAISE, raise_amount is total bet (my_bet after action)."""
    if action_type == RAISE:
        # Engine uses cumulative bet; we store total bet for this player after the raise
        total_bet = prev_my_bet + raise_amount if raise_amount > 0 else prev_opp_bet + raise_amount
        return ActionRecord(player=player, action_type=RAISE, raise_amount=total_bet)
    return ActionRecord(
        player=player,
        action_type=action_type,
        keep_card_1=keep_card_1,
        keep_card_2=keep_card_2,
    )


def parse_opponent_action_from_obs(
    player: int,
    opp_last_action: str,
    my_bet: int,
    opp_bet: int,
    observation: dict,
) -> Optional[ActionRecord]:
    """
    Infer opponent's last action from observation (opp_last_action, bets).
    Called in observe() to append opponent action to history.
    """
    opp = 1 - player
    name = (opp_last_action or "").strip().upper()
    if name == "FOLD":
        return ActionRecord(player=opp, action_type=FOLD)
    if name == "CALL":
        return ActionRecord(player=opp, action_type=CALL)
    if name == "CHECK":
        return ActionRecord(player=opp, action_type=CHECK)
    if name == "RAISE":
        return ActionRecord(player=opp, action_type=RAISE, raise_amount=opp_bet)
    if name == "DISCARD":
        # We see opp's 3 discarded cards; keep indices are private. Use placeholder for public key.
        return ActionRecord(player=opp, action_type=DISCARD, keep_card_1=0, keep_card_2=1)
    return None
