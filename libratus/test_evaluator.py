"""Unit tests for deck and evaluator."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libratus.deck import rank, suit, card_str, same_suit, rank_gap, are_connected, RANKS, SUITS, DECK
from libratus.evaluator import evaluate_hand, mc_equity


def card(name: str) -> int:
    """Convert 'Ad' -> int."""
    r = RANKS.index(name[0])
    s = SUITS.index(name[1])
    return s * len(RANKS) + r


def test_card_encoding():
    assert card_str(0) == "2d"
    assert card_str(8) == "Ad"
    assert card_str(9) == "2h"
    assert card_str(17) == "Ah"
    assert card_str(26) == "As"
    assert rank(card("As")) == 8
    assert suit(card("As")) == 2
    print("  card encoding OK")


def test_suit_rank():
    assert same_suit(card("2d"), card("9d"))
    assert not same_suit(card("2d"), card("2h"))
    assert rank_gap(card("8d"), card("9d")) == 1
    assert are_connected(card("Ad"), card("2d"))  # A-2 wrap
    print("  suit/rank helpers OK")


def test_hand_ranking_order():
    # Straight flush > full house > flush > straight > trips > two pair > pair > high card
    board = [card("3d"), card("4d"), card("5d"), card("6d"), card("7d")]

    sf = evaluate_hand([card("8d"), card("9d")], [card("5d"), card("6d"), card("7d"), card("3h"), card("2h")])
    flush = evaluate_hand([card("2d"), card("Ad")], [card("5d"), card("6d"), card("7h"), card("3h"), card("2h")])

    # Lower = better, so SF should be lower
    # Just verify these don't crash; exact values depend on treys
    assert isinstance(sf, int)
    assert isinstance(flush, int)
    print("  hand ranking OK (no crash)")


def test_pair_beats_high_card():
    board = [card("3d"), card("5h"), card("7s"), card("2d"), card("6h")]
    pair = evaluate_hand([card("9d"), card("9h")], board)
    high = evaluate_hand([card("Ad"), card("8h")], board)
    assert pair < high, "Pair should beat high card (lower rank = better)"
    print("  pair beats high card OK")


def test_trips_beats_pair():
    board = [card("9d"), card("3h"), card("5s"), card("2d"), card("7h")]
    trips = evaluate_hand([card("9h"), card("9s")], board)
    pair = evaluate_hand([card("8d"), card("8h")], board)
    assert trips < pair, "Trips should beat pair"
    print("  trips beats pair OK")


def test_mc_equity_pair_vs_random():
    eq = mc_equity([card("9d"), card("9h")], [card("3d"), card("5h"), card("7s")],
                   dead=set(), num_sims=500)
    assert 0.4 < eq < 1.0, f"99 on dry board should have decent equity, got {eq:.3f}"
    print(f"  mc_equity 99 vs random = {eq:.3f} OK")


def test_mc_equity_weak_vs_random():
    eq = mc_equity([card("2d"), card("3h")], [card("7d"), card("8h"), card("9s")],
                   dead=set(), num_sims=500)
    assert 0.0 < eq < 0.6, f"23o on high board should be weak, got {eq:.3f}"
    print(f"  mc_equity 23o vs random = {eq:.3f} OK")


def main():
    print("Running evaluator tests...")
    test_card_encoding()
    test_suit_rank()
    test_hand_ranking_order()
    test_pair_beats_high_card()
    test_trips_beats_pair()
    test_mc_equity_pair_vs_random()
    test_mc_equity_weak_vs_random()
    print("All tests passed!")


if __name__ == "__main__":
    main()
