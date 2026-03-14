"""
Monte Carlo Odds Analyzer for the 27-card poker variant.

Standalone CLI tool that computes win/tie/loss equity at every street
of a hand (pre-flop, flop discard, post-discard, turn, river).

Usage:
    python monte_carlo.py --random --visualize
    python monte_carlo.py --random --sims 20000 --visualize
    python monte_carlo.py --my-cards "Ah 9d 5s 3h 2d" --visualize
    python monte_carlo.py --my-cards "Ah 9d 5s 3h 2d" --board "9s 8h 2h 6d As" --visualize
"""

import argparse
import json
import os
import random
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from treys import Card

from gym_env import WrappedEval

BATCH_JSON_PATH = "monte_carlo_batch.json"
BATCH_CHART_PATH = "monte_carlo_batch.png"

RANKS = "23456789A"
SUITS = "dhs"
DECK_SIZE = len(RANKS) * len(SUITS)  # 27


# ---------------------------------------------------------------------------
# Card conversion helpers
# ---------------------------------------------------------------------------

def int_to_str(card_int: int) -> str:
    rank = RANKS[card_int % len(RANKS)]
    suit = SUITS[card_int // len(RANKS)]
    return rank + suit


def str_to_int(card_str: str) -> int:
    rank, suit = card_str[0], card_str[1]
    return SUITS.index(suit) * len(RANKS) + RANKS.index(rank)


def int_to_treys(card_int: int) -> int:
    return Card.new(int_to_str(card_int))


def fmt_cards(card_ints) -> str:
    return " ".join(int_to_str(c) for c in card_ints)


# ---------------------------------------------------------------------------
# MonteCarloSimulator
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    def __init__(self):
        self.evaluator = WrappedEval()

    # ---- core evaluation helpers ----

    def _eval_rank(self, hole_ints: list[int], board_ints: list[int]) -> int:
        """Lower rank = better hand."""
        hand = [int_to_treys(c) for c in hole_ints]
        board = [int_to_treys(c) for c in board_ints]
        return self.evaluator.evaluate(hand, board)

    def _remaining_deck(self, known: set[int]) -> list[int]:
        return [i for i in range(DECK_SIZE) if i not in known]

    # ---- Monte Carlo equity (sampling) ----

    def compute_equity(
        self,
        my_cards: list[int],
        board: list[int],
        dead_cards: list[int],
        num_sims: int = 10_000,
    ) -> dict:
        """
        Monte Carlo equity for 2 hole cards against a random opponent hand.
        Samples unknown opponent cards and remaining board cards.

        Returns dict with win_pct, tie_pct, loss_pct, total.
        """
        known = set(my_cards) | set(board) | set(dead_cards)
        remaining = self._remaining_deck(known)
        board_needed = 5 - len(board)
        opp_needed = 2

        wins = ties = losses = 0
        for _ in range(num_sims):
            need = opp_needed + board_needed
            if need > len(remaining):
                break
            sample = random.sample(remaining, need)
            opp = sample[:opp_needed]
            full_board = list(board) + sample[opp_needed:]

            my_rank = self._eval_rank(my_cards, full_board)
            opp_rank = self._eval_rank(opp, full_board)

            if my_rank < opp_rank:
                wins += 1
            elif my_rank == opp_rank:
                ties += 1
            else:
                losses += 1

        total = wins + ties + losses
        if total == 0:
            return {"win_pct": 0, "tie_pct": 0, "loss_pct": 0, "total": 0}
        return {
            "win_pct": wins / total * 100,
            "tie_pct": ties / total * 100,
            "loss_pct": losses / total * 100,
            "total": total,
        }

    # ---- Exhaustive equity ----

    def compute_equity_exact(
        self,
        my_cards: list[int],
        board: list[int],
        dead_cards: list[int],
    ) -> dict:
        """
        Exhaustive equity: enumerate all opponent 2-card combos and (if needed)
        all remaining board completions.
        """
        known = set(my_cards) | set(board) | set(dead_cards)
        remaining = self._remaining_deck(known)
        board_needed = 5 - len(board)

        wins = ties = losses = 0

        if board_needed == 0:
            for opp in combinations(remaining, 2):
                my_rank = self._eval_rank(my_cards, board)
                opp_rank = self._eval_rank(list(opp), board)
                if my_rank < opp_rank:
                    wins += 1
                elif my_rank == opp_rank:
                    ties += 1
                else:
                    losses += 1
        else:
            for opp in combinations(remaining, 2):
                after_opp = [c for c in remaining if c not in opp]
                for extra_board in combinations(after_opp, board_needed):
                    full_board = list(board) + list(extra_board)
                    my_rank = self._eval_rank(my_cards, full_board)
                    opp_rank = self._eval_rank(list(opp), full_board)
                    if my_rank < opp_rank:
                        wins += 1
                    elif my_rank == opp_rank:
                        ties += 1
                    else:
                        losses += 1

        total = wins + ties + losses
        if total == 0:
            return {"win_pct": 0, "tie_pct": 0, "loss_pct": 0, "total": 0}
        return {
            "win_pct": wins / total * 100,
            "tie_pct": ties / total * 100,
            "loss_pct": losses / total * 100,
            "total": total,
        }

    # ---- Discard option ranking ----

    def rank_discard_options(
        self,
        my_5_cards: list[int],
        board: list[int],
        dead_cards: list[int],
        num_sims: int = 5_000,
    ) -> list[dict]:
        """
        Evaluate all C(5,2)=10 ways to keep 2 cards.
        Returns list sorted by win_pct descending, each entry:
        {keep_indices, keep_cards, win_pct, tie_pct, loss_pct, total}
        """
        results = []
        for i, j in combinations(range(5), 2):
            keep = [my_5_cards[i], my_5_cards[j]]
            discard = [my_5_cards[k] for k in range(5) if k != i and k != j]
            eq = self.compute_equity(keep, board, dead_cards + discard, num_sims)
            results.append({
                "keep_indices": (i, j),
                "keep_cards": keep,
                "discard_cards": discard,
                **eq,
            })
        results.sort(key=lambda r: r["win_pct"], reverse=True)
        return results

    # ---- Full hand walk-through ----

    def full_hand_analysis(
        self,
        my_5_cards: list[int],
        opp_5_cards: list[int],
        community_5: list[int],
        my_keep: tuple[int, int] | None = None,
        opp_keep: tuple[int, int] | None = None,
        num_sims: int = 10_000,
    ):
        """
        Walk through every street of a fully-dealt hand, printing equity
        at each stage.

        If my_keep / opp_keep are None, the simulator picks the best
        discard option (highest win_pct) automatically.
        """
        sep = "=" * 56

        print(sep)
        print("       MONTE CARLO HAND ANALYSIS")
        print(sep)
        print(f"  Your hole cards : {fmt_cards(my_5_cards)}")
        print(f"  Opp hole cards  : {fmt_cards(opp_5_cards)}")
        print(f"  Community cards : {fmt_cards(community_5)}")
        print(f"  Sims per stage  : {num_sims:,}")
        print(sep)

        # ---- PRE-FLOP: 5 hole cards, no board visible ----
        print("\n--- PRE-FLOP (5 hole cards, no board) ---")
        preflop_options = self.rank_discard_options(
            my_5_cards, board=[], dead_cards=list(opp_5_cards), num_sims=num_sims,
        )
        best_pre = preflop_options[0]
        for r in preflop_options:
            tag = "  *BEST*" if r is best_pre else ""
            print(
                f"  Keep {fmt_cards(r['keep_cards']):>5s}  ->  "
                f"Win {r['win_pct']:5.1f}%  Tie {r['tie_pct']:4.1f}%  "
                f"Lose {r['loss_pct']:5.1f}%{tag}"
            )
        print(f"\n  * Best pre-flop keep: {fmt_cards(best_pre['keep_cards'])} "
              f"({best_pre['win_pct']:.1f}% win)")

        # ---- FLOP DISCARD: 5 hole cards, 3 board visible ----
        board_flop = community_5[:3]
        print(f"\n--- FLOP DISCARD (board: {fmt_cards(board_flop)}) ---")
        flop_options = self.rank_discard_options(
            my_5_cards, board=board_flop, dead_cards=list(opp_5_cards), num_sims=num_sims,
        )
        best_flop = flop_options[0]
        for r in flop_options:
            tag = "  *BEST*" if r is best_flop else ""
            print(
                f"  Keep {fmt_cards(r['keep_cards']):>5s}  ->  "
                f"Win {r['win_pct']:5.1f}%  Tie {r['tie_pct']:4.1f}%  "
                f"Lose {r['loss_pct']:5.1f}%{tag}"
            )

        # Resolve keeps
        if my_keep is None:
            my_keep = best_flop["keep_indices"]
        if opp_keep is None:
            opp_flop_options = self.rank_discard_options(
                opp_5_cards, board=board_flop, dead_cards=list(my_5_cards), num_sims=num_sims,
            )
            opp_keep = opp_flop_options[0]["keep_indices"]

        my_kept = [my_5_cards[my_keep[0]], my_5_cards[my_keep[1]]]
        my_discarded = [my_5_cards[k] for k in range(5) if k not in my_keep]
        opp_kept = [opp_5_cards[opp_keep[0]], opp_5_cards[opp_keep[1]]]
        opp_discarded = [opp_5_cards[k] for k in range(5) if k not in opp_keep]

        dead = my_discarded + opp_discarded

        print(f"\n  -> You keep: {fmt_cards(my_kept)}  (discard {fmt_cards(my_discarded)})")
        print(f"  -> Opp keeps: {fmt_cards(opp_kept)}  (discard {fmt_cards(opp_discarded)})")

        # ---- FLOP EQUITY (after discard) ----
        print(f"\n--- FLOP EQUITY (after discard) ---")
        print(f"  Your hand: {fmt_cards(my_kept)}  |  Board: {fmt_cards(board_flop)}")
        print(f"  Dead cards: {fmt_cards(dead)}")
        flop_eq = self.compute_equity(my_kept, board_flop, dead + list(opp_kept), num_sims)
        print(
            f"  Win {flop_eq['win_pct']:5.1f}%  Tie {flop_eq['tie_pct']:4.1f}%  "
            f"Lose {flop_eq['loss_pct']:5.1f}%  ({flop_eq['total']:,} sims)"
        )

        # ---- TURN ----
        board_turn = community_5[:4]
        print(f"\n--- TURN (board: {fmt_cards(board_turn)}) ---")
        turn_eq = self.compute_equity_exact(my_kept, board_turn, dead + list(opp_kept))
        exact_tag = "[EXACT]" if turn_eq["total"] < 50_000 else f"[{turn_eq['total']:,} combos]"
        print(
            f"  Win {turn_eq['win_pct']:5.1f}%  Tie {turn_eq['tie_pct']:4.1f}%  "
            f"Lose {turn_eq['loss_pct']:5.1f}%  {exact_tag}  "
            f"({turn_eq['total']:,} combos evaluated)"
        )

        # ---- RIVER ----
        board_river = community_5[:5]
        print(f"\n--- RIVER (board: {fmt_cards(board_river)}) ---")
        river_eq = self.compute_equity_exact(my_kept, board_river, dead + list(opp_kept))
        print(
            f"  Win {river_eq['win_pct']:5.1f}%  Tie {river_eq['tie_pct']:4.1f}%  "
            f"Lose {river_eq['loss_pct']:5.1f}%  [EXACT]  "
            f"({river_eq['total']:,} opponent combos evaluated)"
        )

        print(f"\n{'=' * 56}")
        return {
            "preflop": preflop_options,
            "flop_discard": flop_options,
            "flop_equity": flop_eq,
            "turn_equity": turn_eq,
            "river_equity": river_eq,
            "my_kept": my_kept,
            "opp_kept": opp_kept,
            "my_5_cards": my_5_cards,
            "opp_5_cards": opp_5_cards,
            "community_5": community_5,
        }

    def run_one_hand_equities(
        self,
        my_5_cards: list[int],
        opp_5_cards: list[int],
        community_5: list[int],
        num_sims: int,
    ) -> tuple[float, float, float, float, float]:
        """
        Run full analysis for one hand (all discard options at pre-flop and flop,
        then flop/turn/river equity). Returns only the five win%% values:
        (preflop_best, flop_discard_best, flop_eq, turn_eq, river_eq).
        """
        preflop_options = self.rank_discard_options(
            my_5_cards, board=[], dead_cards=list(opp_5_cards), num_sims=num_sims,
        )
        preflop_best = preflop_options[0]["win_pct"]

        board_flop = community_5[:3]
        flop_options = self.rank_discard_options(
            my_5_cards, board=board_flop, dead_cards=list(opp_5_cards), num_sims=num_sims,
        )
        flop_discard_best = flop_options[0]["win_pct"]
        my_keep = flop_options[0]["keep_indices"]

        opp_flop_options = self.rank_discard_options(
            opp_5_cards, board=board_flop, dead_cards=list(my_5_cards), num_sims=num_sims,
        )
        opp_keep = opp_flop_options[0]["keep_indices"]

        my_kept = [my_5_cards[my_keep[0]], my_5_cards[my_keep[1]]]
        my_discarded = [my_5_cards[k] for k in range(5) if k not in my_keep]
        opp_kept = [opp_5_cards[opp_keep[0]], opp_5_cards[opp_keep[1]]]
        opp_discarded = [opp_5_cards[k] for k in range(5) if k not in opp_keep]
        dead = my_discarded + opp_discarded

        flop_eq = self.compute_equity(
            my_kept, board_flop, dead + list(opp_kept), num_sims,
        )["win_pct"]
        turn_eq = self.compute_equity_exact(
            my_kept, community_5[:4], dead + list(opp_kept),
        )["win_pct"]
        river_eq = self.compute_equity_exact(
            my_kept, community_5[:5], dead + list(opp_kept),
        )["win_pct"]

        return (preflop_best, flop_discard_best, flop_eq, turn_eq, river_eq)


# ---------------------------------------------------------------------------
# Batch run (many hands)
# ---------------------------------------------------------------------------

def run_batch(
    num_hands: int = 2000,
    sims_per_hand: int = 1000,
    verbose: bool = True,
    visualize: bool = True,
) -> dict:
    """
    Run num_hands random hands; for each, evaluate all options at pre-flop and
    flop discard, then flop/turn/river equity. Aggregate and optionally plot.
    """
    sim = MonteCarloSimulator()
    streets = ["Pre-Flop (best keep)", "Flop discard (best keep)", "Flop", "Turn", "River"]
    data = {s: [] for s in streets}

    if verbose:
        print(f"Running {num_hands:,} hands with {sims_per_hand:,} sims per stage...")
        print("(Pre-flop: 10 keep options; Flop: 10 keep options; then Flop/Turn/River equity)\n")

    for h in range(num_hands):
        my_5, opp_5, community = deal_random_hand()
        pre, flop_d, flop, turn, river = sim.run_one_hand_equities(
            my_5, opp_5, community, num_sims=sims_per_hand,
        )
        data[streets[0]].append(pre)
        data[streets[1]].append(flop_d)
        data[streets[2]].append(flop)
        data[streets[3]].append(turn)
        data[streets[4]].append(river)

        if verbose and (h + 1) % 200 == 0:
            print(f"  Hand {h + 1:,} / {num_hands:,}")

    # Summary stats
    summary = {}
    if verbose:
        print("\n" + "=" * 64)
        print("  AGGREGATE EQUITY ACROSS ALL HANDS (win % at each street)")
        print("=" * 64)
        print(f"  {'Street':<28}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        print("-" * 64)
    for s in streets:
        arr = np.array(data[s])
        summary[s] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                      "min": float(np.min(arr)), "max": float(np.max(arr))}
        if verbose:
            print(f"  {s:<28}  {summary[s]['mean']:>7.1f}%  {summary[s]['std']:>7.1f}%  "
                  f"{summary[s]['min']:>7.1f}%  {summary[s]['max']:>7.1f}%")
    if verbose:
        print("=" * 64)

    if visualize:
        visualize_batch(data, streets, summary, num_hands, sims_per_hand)

    # Save batch data for later re-plotting
    with open(BATCH_JSON_PATH, "w") as f:
        json.dump({
            "num_hands": num_hands,
            "sims_per_hand": sims_per_hand,
            "streets": streets,
            "data": data,
            "summary": summary,
        }, f, indent=0)
    if verbose:
        print(f"Batch data saved to {BATCH_JSON_PATH}")

    return {"data": data, "summary": summary}


def visualize_batch(
    data: dict,
    streets: list[str],
    summary: dict,
    num_hands: int,
    sims_per_hand: int,
):
    """Histograms and mean progression for batch run."""
    plt.rc("font", size=9)
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28,
                          left=0.06, right=0.98, top=0.88, bottom=0.08)

    fig.suptitle(
        f"Batch: {num_hands:,} hands  |  {sims_per_hand:,} sims per stage  |  "
        "All discard options evaluated at Pre-Flop and Flop",
        fontsize=12, fontweight="bold", color=TEXT, y=0.96,
    )

    # Left: mean equity by street (bar + error bar)
    ax0 = fig.add_subplot(gs[0, 0])
    _style_ax(ax0, "Average win % by street")
    x = np.arange(len(streets))
    means = [summary[s]["mean"] for s in streets]
    stds = [summary[s]["std"] for s in streets]
    ax0.bar(x, means, 0.6, color="#38bdf8", edgecolor=GRID)
    ax0.errorbar(x, means, yerr=stds, fmt="none", color=TEXT, capsize=4)
    ax0.axhline(50, color=GRID, linestyle="--", linewidth=1, alpha=0.7)
    ax0.set_xticks(x)
    ax0.set_xticklabels([s.replace(" (best keep)", "") for s in streets], color=TEXT, rotation=12, ha="right")
    ax0.set_ylim(0, 100)
    ax0.yaxis.set_major_formatter(mtick.PercentFormatter())
    for i, m in enumerate(means):
        ax0.annotate(f"{m:.1f}%", (i, m + stds[i] + 1.5), ha="center", fontsize=9, color=TEXT)

    # Right: distribution of equity at each street (box plot or histograms)
    ax1 = fig.add_subplot(gs[0, 1])
    _style_ax(ax1, "Distribution of win % by street")
    arrs = [np.array(data[s]) for s in streets]
    bp = ax1.boxplot(arrs, labels=[s.replace(" (best keep)", "") for s in streets],
                     patch_artist=True, notch=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor(PANEL)
        patch.set_edgecolor(GRID)
    ax1.axhline(50, color=GRID, linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_ylabel("Win %", color=TEXT)
    ax1.set_ylim(0, 100)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.setp(ax1.get_xticklabels(), rotation=12, ha="right", color=TEXT)

    # Bottom: 5 histograms (one per street)
    ax2 = fig.add_subplot(gs[1, :])
    _style_ax(ax2, "Histogram of win % at each street")
    bins = np.linspace(0, 100, 26)
    colors = ["#38bdf8", "#22c55e", "#eab308", "#f97316", "#ef4444"]
    for i, s in enumerate(streets):
        ax2.hist(data[s], bins=bins, alpha=0.5, label=s.replace(" (best keep)", ""),
                 color=colors[i], edgecolor=GRID, density=True, histtype="stepfilled")
    ax2.axvline(50, color=GRID, linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Win %", color=TEXT)
    ax2.set_ylabel("Density", color=TEXT)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.4, facecolor=PANEL, edgecolor=GRID)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter())

    plt.savefig(BATCH_CHART_PATH, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"\nBatch chart saved to {BATCH_CHART_PATH}")


# ---------------------------------------------------------------------------
# Visualization (single hand)
# ---------------------------------------------------------------------------

# Clean palette: green / amber / red
COLORS = {"win": "#22c55e", "tie": "#eab308", "loss": "#ef4444"}
BG = "#0f172a"
PANEL = "#1e293b"
GRID = "#334155"
TEXT = "#e2e8f0"


def visualize_results(results: dict):
    """Generate matplotlib charts from full_hand_analysis results (saves to file only)."""
    plt.rc("font", size=9)
    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    gs = fig.add_gridspec(3, 2, hspace=0.36, wspace=0.28,
                          left=0.06, right=0.98, top=0.88, bottom=0.06)

    my_5 = results["my_5_cards"]
    opp_5 = results["opp_5_cards"]
    board = results["community_5"]

    fig.suptitle(
        f"Monte Carlo Hand Analysis  |  You: {fmt_cards(my_5)}  |  Opp: {fmt_cards(opp_5)}  |  Board: {fmt_cards(board)}",
        fontsize=12, fontweight="bold", color=TEXT, y=0.96,
    )

    _plot_discard_bars(fig.add_subplot(gs[0, 0]), results["preflop"], "Pre-Flop (best 2 to keep)")
    _plot_discard_bars(fig.add_subplot(gs[0, 1]), results["flop_discard"],
                       f"Flop (board: {fmt_cards(board[:3])})")
    _plot_equity_progression(fig.add_subplot(gs[1, :]), results)
    _plot_street_stacked(fig.add_subplot(gs[2, 0]), results)
    _plot_river_summary(fig.add_subplot(gs[2, 1]), results)

    out_path = "monte_carlo_analysis.png"
    plt.savefig(out_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"\nChart saved to {out_path}")


def _style_ax(ax, title: str):
    ax.set_facecolor(PANEL)
    ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT, pad=8)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, axis="both", alpha=0.2, color=GRID)


def _plot_discard_bars(ax, options: list[dict], title: str):
    """Horizontal stacked bars for the 10 keep-pair options."""
    _style_ax(ax, title)

    labels = [fmt_cards(r["keep_cards"]) for r in reversed(options)]
    wins = [r["win_pct"] for r in reversed(options)]
    ties = [r["tie_pct"] for r in reversed(options)]
    losses = [r["loss_pct"] for r in reversed(options)]

    y = np.arange(len(labels))
    h = 0.72

    ax.barh(y, wins, h, label="Win", color=COLORS["win"], edgecolor="none")
    ax.barh(y, ties, h, left=wins, label="Tie", color=COLORS["tie"], edgecolor="none")
    ax.barh(y, losses, h, left=[w + t for w, t in zip(wins, ties)],
            label="Loss", color=COLORS["loss"], edgecolor="none")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontfamily="monospace", fontsize=9, color=TEXT)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="lower right", fontsize=8, framealpha=0.4, facecolor=PANEL, edgecolor=GRID)

    best_idx = len(options) - 1
    ax.annotate("best", xy=(wins[best_idx] / 2, best_idx), fontsize=8,
                fontweight="bold", color="white", ha="center", va="center")


def _plot_equity_progression(ax, results: dict):
    """Line chart: win % across streets."""
    _style_ax(ax, "Equity by street (win %)")

    best_pre = results["preflop"][0]["win_pct"]
    best_flop_disc = results["flop_discard"][0]["win_pct"]
    flop = results["flop_equity"]["win_pct"]
    turn = results["turn_equity"]["win_pct"]
    river = results["river_equity"]["win_pct"]

    streets = ["Pre-Flop", "Flop discard", "Flop", "Turn", "River"]
    equity = [best_pre, best_flop_disc, flop, turn, river]
    x = np.arange(len(streets))

    ax.plot(x, equity, "o-", color="#38bdf8", linewidth=2.2, markersize=9,
            markerfacecolor=PANEL, markeredgecolor="#38bdf8", markeredgewidth=1.8)
    ax.axhline(50, color=GRID, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    for i, eq in enumerate(equity):
        c = COLORS["win"] if eq >= 50 else COLORS["loss"]
        ax.annotate(f"{eq:.1f}%", (i, eq), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color=c)

    ax.set_xticks(x)
    ax.set_xticklabels(streets, color=TEXT)
    ax.set_ylabel("Win %", color=TEXT)


def _plot_street_stacked(ax, results: dict):
    """Stacked bars: win / tie / loss at each street."""
    _style_ax(ax, "Outcome mix by street")

    streets = ["Pre-Flop", "Flop disc", "Flop", "Turn", "River"]
    data = [
        results["preflop"][0],
        results["flop_discard"][0],
        results["flop_equity"],
        results["turn_equity"],
        results["river_equity"],
    ]
    wins = [d["win_pct"] for d in data]
    ties = [d["tie_pct"] for d in data]
    losses = [d["loss_pct"] for d in data]

    x = np.arange(len(streets))
    w = 0.52
    ax.bar(x, wins, w, label="Win", color=COLORS["win"])
    ax.bar(x, ties, w, bottom=wins, label="Tie", color=COLORS["tie"])
    ax.bar(x, losses, w, bottom=[wi + ti for wi, ti in zip(wins, ties)],
           label="Loss", color=COLORS["loss"])

    for i in range(len(streets)):
        if wins[i] > 8:
            ax.text(i, wins[i] / 2, f"{wins[i]:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(streets, color=TEXT)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="upper right", fontsize=8, framealpha=0.4, facecolor=PANEL, edgecolor=GRID)


def _plot_river_summary(ax, results: dict):
    """River outcome donut and kept hand."""
    _style_ax(ax, "River (exact)")

    river = results["river_equity"]
    sizes = [river["win_pct"], river["tie_pct"], river["loss_pct"]]
    labels_list = ["Win", "Tie", "Loss"]
    colors = [COLORS["win"], COLORS["tie"], COLORS["loss"]]

    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels_list, colors) if s > 0]
    if not non_zero:
        return
    sizes_nz, labels_nz, colors_nz = zip(*non_zero)

    wedges, _, autotexts = ax.pie(
        sizes_nz, labels=labels_nz, colors=colors_nz, autopct="%1.1f%%",
        startangle=90, pctdistance=0.72, textprops={"color": TEXT, "fontsize": 10},
        wedgeprops={"width": 0.5, "edgecolor": PANEL, "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")

    kept = results["my_kept"]
    ax.text(0, 0, fmt_cards(kept), ha="center", va="center",
            fontsize=11, fontweight="bold", color=TEXT, fontfamily="monospace")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def deal_random_hand() -> tuple[list[int], list[int], list[int]]:
    deck = list(range(DECK_SIZE))
    random.shuffle(deck)
    my_5 = deck[:5]
    opp_5 = deck[5:10]
    community = deck[10:15]
    return my_5, opp_5, community


def parse_card_string(s: str) -> list[int]:
    tokens = s.strip().split()
    return [str_to_int(t) for t in tokens]


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo odds analyzer for the 27-card poker variant."
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Deal a random hand and run full analysis.",
    )
    parser.add_argument(
        "--my-cards",
        type=str, default=None,
        help='Your 5 hole cards, e.g. "Ah 9d 5s 3h 2d".',
    )
    parser.add_argument(
        "--opp-cards",
        type=str, default=None,
        help='Opponent 5 hole cards (optional; random if omitted).',
    )
    parser.add_argument(
        "--board",
        type=str, default=None,
        help='Community cards (up to 5), e.g. "9s 8h 2h 6d As".',
    )
    parser.add_argument(
        "--sims", type=int, default=20_000,
        help="Monte Carlo simulations per stage (default: 20000).",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate matplotlib charts of the analysis.",
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="Batch mode: run N random hands (all options at each street). Default 2000 when used.",
    )
    parser.add_argument(
        "--batch-sims", type=int, default=1000,
        help="Sims per stage when using --rounds (default: 1000).",
    )
    parser.add_argument(
        "--no-batch-viz", action="store_true",
        help="Skip saving batch chart when using --rounds.",
    )
    parser.add_argument(
        "--visualize-batch", action="store_true",
        help="Load saved batch data from monte_carlo_batch.json and regenerate chart.",
    )
    args = parser.parse_args()

    if args.visualize_batch:
        if not os.path.isfile(BATCH_JSON_PATH):
            print(f"Missing {BATCH_JSON_PATH}. Run with --rounds N first to generate batch data.")
            return
        with open(BATCH_JSON_PATH) as f:
            saved = json.load(f)
        visualize_batch(
            saved["data"], saved["streets"], saved["summary"],
            saved["num_hands"], saved["sims_per_hand"],
        )
        return

    sim = MonteCarloSimulator()

    if args.rounds is not None:
        num_hands = args.rounds if args.rounds > 0 else 2000
        run_batch(
            num_hands=num_hands,
            sims_per_hand=args.batch_sims,
            verbose=True,
            visualize=not args.no_batch_viz,
        )
        return

    if args.random:
        my_5, opp_5, community = deal_random_hand()
        results = sim.full_hand_analysis(my_5, opp_5, community, num_sims=args.sims)
        if args.visualize:
            visualize_results(results)
        return

    if args.my_cards:
        my_5 = parse_card_string(args.my_cards)
        if len(my_5) != 5:
            parser.error("--my-cards must specify exactly 5 cards.")

        known = set(my_5)

        if args.opp_cards:
            opp_5 = parse_card_string(args.opp_cards)
            if len(opp_5) != 5:
                parser.error("--opp-cards must specify exactly 5 cards.")
            known |= set(opp_5)
        else:
            remaining = [c for c in range(DECK_SIZE) if c not in known]
            random.shuffle(remaining)
            opp_5 = remaining[:5]
            known |= set(opp_5)

        if args.board:
            community_partial = parse_card_string(args.board)
            known |= set(community_partial)
            remaining = [c for c in range(DECK_SIZE) if c not in known]
            random.shuffle(remaining)
            need = 5 - len(community_partial)
            community = community_partial + remaining[:need]
        else:
            remaining = [c for c in range(DECK_SIZE) if c not in known]
            random.shuffle(remaining)
            community = remaining[:5]

        results = sim.full_hand_analysis(my_5, opp_5, community, num_sims=args.sims)
        if args.visualize:
            visualize_results(results)
        return

    parser.print_help()
    print("\nExamples:")
    print('  python monte_carlo.py --random --visualize')
    print('  python monte_carlo.py --rounds 2000              # 2000 hands, aggregate odds')
    print('  python monte_carlo.py --rounds 2000 --batch-sims 500  # faster, fewer sims/hand')
    print('  python monte_carlo.py --my-cards "Ah 9d 5s 3h 2d" --visualize')


if __name__ == "__main__":
    main()
