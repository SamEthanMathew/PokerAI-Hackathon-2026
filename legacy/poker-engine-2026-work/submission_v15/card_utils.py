"""
Card encoding/decoding and dead-card tracking utilities (v14 Alpha).
"""

from submission.constants import NUM_CARDS, NUM_RANKS, NUM_SUITS, RANKS, SUITS, RANK_A


class CardUtils:
    """Stateless utility class for card manipulation in the 27-card deck."""

    @staticmethod
    def card_to_int(rank_char: str, suit_char: str) -> int:
        """Convert rank+suit chars (e.g. 'A', 'h') to int 0-26."""
        return SUITS.index(suit_char) * NUM_RANKS + RANKS.index(rank_char)

    @staticmethod
    def int_to_rank_suit(card_int: int) -> tuple:
        """Returns (rank_index, suit_index) from card int."""
        return card_int % NUM_RANKS, card_int // NUM_RANKS

    @staticmethod
    def int_to_str(card_int: int) -> str:
        """Convert card int to human-readable string like '9h'."""
        rank_idx = card_int % NUM_RANKS
        suit_idx = card_int // NUM_RANKS
        return f"{RANKS[rank_idx]}{SUITS[suit_idx]}"

    @staticmethod
    def get_rank(card_int: int) -> int:
        """Return the rank index (0=2, 1=3, ..., 7=9, 8=A)."""
        return card_int % NUM_RANKS

    @staticmethod
    def get_suit(card_int: int) -> int:
        """Return the suit index (0=d, 1=h, 2=s)."""
        return card_int // NUM_RANKS

    @staticmethod
    def get_full_deck() -> set:
        return set(range(NUM_CARDS))

    @staticmethod
    def get_dead_cards(observation: dict) -> set:
        """All cards known to be out of the unknown pool."""
        dead = set()
        for card in observation.get("my_cards", []):
            if card != -1:
                dead.add(card)
        for card in observation.get("community_cards", []):
            if card != -1:
                dead.add(card)
        for card in observation.get("my_discarded_cards", []):
            if card != -1:
                dead.add(card)
        for card in observation.get("opp_discarded_cards", []):
            if card != -1:
                dead.add(card)
        return dead

    @staticmethod
    def get_unknown_cards(observation: dict) -> list:
        """Cards that could be in opponent's hand or undealt community cards."""
        dead = CardUtils.get_dead_cards(observation)
        return [c for c in range(NUM_CARDS) if c not in dead]

    @staticmethod
    def get_my_cards(observation: dict) -> list:
        """Filter out -1 sentinel values from my_cards."""
        return [c for c in observation.get("my_cards", []) if c != -1]

    @staticmethod
    def get_community_cards(observation: dict) -> list:
        """Filter out -1 sentinel values from community_cards."""
        return [c for c in observation.get("community_cards", []) if c != -1]

    @staticmethod
    def get_opp_discards(observation: dict) -> list:
        """Filter out -1 sentinel values from opp_discarded_cards."""
        return [c for c in observation.get("opp_discarded_cards", []) if c != -1]

    @staticmethod
    def count_pairs(cards: list) -> int:
        """Count number of pairs among a list of card ints."""
        ranks = [c % NUM_RANKS for c in cards]
        count = 0
        seen = {}
        for r in ranks:
            seen[r] = seen.get(r, 0) + 1
        for v in seen.values():
            count += v * (v - 1) // 2
        return count

    @staticmethod
    def count_trips(cards: list) -> int:
        """Count number of three-of-a-kinds."""
        ranks = [c % NUM_RANKS for c in cards]
        seen = {}
        for r in ranks:
            seen[r] = seen.get(r, 0) + 1
        return sum(1 for v in seen.values() if v >= 3)

    @staticmethod
    def max_suited_count(cards: list) -> int:
        """How many cards share the most common suit."""
        suits = [c // NUM_RANKS for c in cards]
        if not suits:
            return 0
        from collections import Counter
        return Counter(suits).most_common(1)[0][1]

    @staticmethod
    def max_connected_count(cards: list) -> int:
        """Longest run of consecutive ranks (Ace can be low or high)."""
        if not cards:
            return 0
        ranks = sorted(set(c % NUM_RANKS for c in cards))
        if RANK_A in ranks:
            ranks_ext = sorted(ranks + [-1, NUM_RANKS])
        else:
            ranks_ext = ranks
        best = 1
        run = 1
        for i in range(1, len(ranks_ext)):
            if ranks_ext[i] == ranks_ext[i - 1] + 1:
                run += 1
                best = max(best, run)
            elif ranks_ext[i] != ranks_ext[i - 1]:
                run = 1
        return best

    @staticmethod
    def high_card_score(cards: list) -> float:
        """Simple score based on rank values (A=8 is highest)."""
        if not cards:
            return 0.0
        ranks = [c % NUM_RANKS for c in cards]
        return sum(ranks) / len(ranks) / RANK_A
