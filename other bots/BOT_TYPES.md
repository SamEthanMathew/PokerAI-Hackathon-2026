# Other Bots — Bot Types Overview

Quick reference for all files in `other bots/`: which are **playable bots** vs **support modules**, and what each bot does.

---

## Playable bots (entry points)

| File | Agent class | Type / lineage | Short description |
|------|-------------|----------------|-------------------|
| **ensemble.py** | `EnsembleAgent` | **Ensemble** — shared-context switcher | Wraps all other bots. Every bot observes every action (parallel context), so switching mid-match is seamless. Active bot set via `POST /switch_bot` or a hand-number schedule. See file docstring. |
| **ALPHANiT.py** | `PlayerAgent` | **ALPHANiT V2** — Nit / MC | Nit-style (tight/cautious): premium-hand preflop filter, **Monte Carlo equity** for discard (all 10 keep combos, threat-suit blocking). Postflop: board-paired equity penalty when weak, river call cap. Tunable via `bot_profile.json`. |
| **blambot.py** | `PlayerAgent` | **Blambot** — Meta chassis | Meta chassis with opponent modeling: VPIP/PFR, fold-to-bet, aggression factor, net profit. Full postflop (hand classification, made hands vs draws). Same stat-tracking structure as Omicron V1.3. |
| **delta.py** | `PlayerAgent` | **DELTA V1** — Frankenstein | Merges OMICRoN V1 (LUT + exact solver), Genesis V2 (equity, river, tiered draws), Meta V5 (safety caps, overrides), plus **adaptive opponent learning**: EMA stats, Bayesian archetypes, per-street profiling, regime detection, bet-sizing tells. |
| **genesis.py** | `GenesisAgent` | **Genesis** — modular street engine | Street-by-street: preflop (street0_score + street0_bet_sizing), discard (street1a), flop betting (street1b), turn (street2), river (street3). Uses **OpponentRecon** for VPIP/PFR, fold rates, aggression, texture reactions. |
| **OmicronV1.2.py** | `PlayerAgent` | **OMICRoN V1** — fork of ALPHANiTV8 | Exact subgame solver for discard + adaptive opponent modeling. Full postflop: range-weighted equity, board texture, semi-bluff, dynamic sizing, opponent profiling. Speed-optimized (lookup arrays, no Counter in hot paths). |
| **player.py** (root) | `PlayerAgent` | **Genesis wrapper** | Thin wrapper: `GenesisAgent` with `entry_point='player.py'` for logging. Same behavior as Genesis. |
| **submission/player.py** | `PlayerAgent` | **Omicron V1.3** — Meta chassis | “Meta chassis × Alpha killer instinct.” Tracks VPIP/PFR, fold-to-bet, net profit/loss; uses Omicron-style discard/postflop logic. |
| **submission/lambdaV2.py** | `PlayerAgent` | **Lambda V2** — Meta × Alpha | Uses **Libratus tables** (POLICY, KEEP_EQUITY, POSTERIOR, MATCHUPS) + **discard_engine**. Gradient mode (GTO/AGRO/VALUE/TRAP), semi-bluff fixes, equity penalty cap, river call/bluff tuning. |

---

## Support modules (not bots) — in `support/`

Used by one or more bots; no `PlayerAgent` / `GenesisAgent` entry point. All live under **`support/`**.

| File | Purpose |
|------|--------|
| **support/opponent_recon.py** | `OpponentRecon`: central opponent modeling — VPIP, PFR, fold-to-bet (river vs non-river), aggression, preflop response by our sizing, board texture / discard reaction stats. Used by Genesis (and others). |
| **support/street0_score.py** | Preflop hand scoring (0–1), `OpponentProfile`, `ScoreBreakdown`. Used by Genesis (street0). |
| **support/street0_bet_sizing.py** | Preflop bet sizing from recon; `Street0Context`, `get_street0_action_from_recon`. Used by Genesis. |
| **support/street1a_discard.py** | Flop discard: `Street1AContext`, `KeepEvalRecord`, `choose_keep_with_controlled_mixing`, texture classification. Used by Genesis. |
| **support/street1b_betting.py** | Flop (multi-round) betting: `Street1BContext`, `get_street1b_action`. Used by Genesis. |
| **support/street2_turn.py** | Turn logic: `Street2Context`, `get_street2_action`, turn texture. Used by Genesis. |
| **support/street3_river.py** | River logic: `Street3Context`, `get_street3_action`. Used by Genesis. |
| **support/discard_engine.py** | Post-flop discard: 6-step pipeline (made hand, boat potential, flush/straight draws, dead outs, heuristics). Shared by METAV4, ALPHANiTV5, LambdaV2. |
| **support/libratus_tables.py** | Lookup tables: `POLICY`, `KEEP_EQUITY`, `POSTERIOR`, `MATCHUPS`. Used by LambdaV2. |

**Running:** Put the `other bots` directory on `PYTHONPATH` when running these bots (e.g. `PYTHONPATH="other bots:$PYTHONPATH"` from project root). Genesis, player.py, and submission/lambdaV2.py also add `other bots` to `sys.path` when loaded so `support` is found.

---

## By “family”

- **ALPHANiT:** `ALPHANiT.py` — nit-style, Monte Carlo discard, premium preflop, board-paired penalty.
- **Blambot:** `blambot.py` — meta chassis, opponent stats, full postflop (no shared support).
- **Genesis family:** `genesis.py`, `player.py` (root), plus `support/street*` and `support/opponent_recon.py`.
- **OMICRoN / Alpha family:** `OmicronV1.2.py`, `submission/player.py` (Omicron V1.3), `blambot.py` (same chassis style).
- **Lambda / Libratus family:** `submission/lambdaV2.py`, `support/discard_engine.py`, `support/libratus_tables.py`.
- **Delta:** `delta.py` — combines OMICRoN, Genesis, Meta, and its own adaptive learning.

---

## One-line summary by bot type

| Bot | One-line |
|-----|----------|
| **ALPHANiT V2** | Nit: premium preflop, Monte Carlo discard (10 keeps, threat blocking), board-paired equity penalty, river cap. |
| **Blambot** | Meta chassis: VPIP/PFR, fold stats, aggression, net profit; full postflop hand classification. |
| **Delta** | Frankenstein: OMICRoN + Genesis + Meta + adaptive opponent learning (EMA, Bayesian, per-street, regime, sizing tells). |
| **Genesis** | Modular street engine (preflop → discard → flop → turn → river) + OpponentRecon. |
| **OMICRoN V1.2** | Exact discard solver + opponent modeling + full postflop (equity, texture, semi-bluff, sizing). |
| **Omicron V1.3** (submission/player) | Meta chassis with Omicron-style logic; tracks VPIP/PFR, folds, net profit. |
| **Lambda V2** | Libratus tables + discard_engine; gradient GTO/AGRO/VALUE/TRAP; semi-bluff and river tuning. |
| **Player (root)** | Genesis under the hood; entry_point for logging only. |
