# Libratus-Lite Poker Bot

A loose-aggressive, table-driven, mixed-strategy poker bot for the 27-card Texas Hold'em variant.

## Architecture

```
libratus/                    # Offline tools (run before deployment)
  deck.py                    # Card/deck utilities
  evaluator.py               # Hand evaluator wrapper + MC equity
  abstractions.py            # Bucketing: preflop, flop, opp discard, keep archetype
  discard_eval.py            # Keep-selection scoring engine
  simulate.py                # Offline MC simulation pipeline
  generate_tables.py         # Generates submission/libratus_tables.py
  posterior_model.py          # Opponent discard posterior model
  policy_train.py            # Betting policy table construction
  log_analysis.py            # Match log analysis / self-patching
  test_evaluator.py          # Unit tests
  odds_tables.csv            # Generated equity/matchup data

submission/                  # Runtime files (deployed to competition)
  Libratus.py                # PlayerAgent class
  libratus_tables.py         # Pre-computed lookup tables (auto-generated)
```

## Quick Start

### 1. Run Tests

```bash
python libratus/test_evaluator.py
```

### 2. Generate Lookup Tables

Fast mode (~20 seconds):
```bash
python libratus/generate_tables.py --fast
```

Full simulation (~2-5 minutes):
```bash
python libratus/generate_tables.py
```

This produces `submission/libratus_tables.py` with:
- **POLICY**: Betting policy table (400 entries)
- **KEEP_EQUITY**: Equity by keep archetype
- **POSTERIOR**: Opponent discard posterior probabilities
- **MATCHUPS**: Head-to-head win rates

### 3. Run the Bot

The runtime bot is `submission/Libratus.py`. It loads automatically when the match server uses it as an agent.

### 4. Analyze Match Logs

Place match CSV files in `logs/`, then:
```bash
python libratus/log_analysis.py
```

Or specify files directly:
```bash
python libratus/log_analysis.py logs/M1.CSV logs/M2.CSV
```

Use `--team1` flag if we are team 1:
```bash
python libratus/log_analysis.py --team1
```

### 5. Run Simulations Standalone

```bash
python libratus/simulate.py
```

### 6. Build Posterior Model Standalone

```bash
python libratus/posterior_model.py
```

## How It Works

### Discard Decision (Most Important)
- Evaluates all C(5,2)=10 possible 2-card keeps
- Scores each by: **MC equity** (weight 3.0) + **structural bonus** (1.5) + **board interaction** (1.0) + **opponent inference** (0.5)
- Near-ties broken by seeded randomization

### Betting Decision
- Computes MC equity for current hand vs random opponent
- Buckets state into (street, position, strength, board_texture, to_call)
- Looks up mixed-strategy action probabilities from POLICY table
- Samples action from distribution using seeded PRNG
- Legalizes action against valid_actions

### Mixed Strategy
All randomness is seeded from `hash(hand_number, street, cards)` for reproducibility.

### Self-Improvement Loop
1. Play match -> get CSV log
2. Run `python libratus/log_analysis.py` to find leaks
3. Adjust parameters in `policy_train.py`
4. Re-run `python libratus/generate_tables.py`
5. New tables are used automatically on next match
