# CMU PokerAI 26' - Team Ctrl+Alt+Defeat

**A full-stack competition research codebase for CMU’s PokerAI competition:** custom Hold’em variant, tournament engine, many generations of bots (from tight “Nit” play to hybrid solver + meta-learning stacks), human-in-the-loop training apps, RL pipelines, and analysis tooling—organized for clarity and reuse.

---

## At a glance

| | |
|---|---|
| **What** | Multi-agent poker engine (`gym`-style env), HTTP bot API, tournament runner, and primary submission **`submission/player.py` (OMICRoN V2)** |
| **Variant** | 27-card deck (2–9, A × ♦♥♠), **5 hole cards**, mandatory **discard-to-2** after the flop (revealed discards), 4 betting streets |
| **Match format** | 1000 hands / side, bankroll scoring (see `config/RULES.txt`) |
| **Stack** | Python 3.12+, FastAPI/uvicorn agents, NumPy/Treys/PyTorch where used, Next.js trainer UI |
| **This repo** | Single **`main`** line of development; historical trees under `archive/` and `legacy/` |

---

## Table of contents

1. [Why this project exists](#why-this-project-exists)  
2. [The game variant](#the-game-variant)  
3. [Strategies we explored](#strategies-we-explored)  
4. [What we built (systems)](#what-we-built-systems)  
5. [Repository layout](#repository-layout)  
6. [Getting started](#getting-started)  
7. [Running matches](#running-matches)  
8. [Configuration](#configuration)  
9. [Testing](#testing)  
10. [Training & data](#training--data)  
11. [Analysis & visualization](#analysis--visualization)  
12. [Archive, legacy, and history](#archive-legacy-and-history)  
13. [CI & infrastructure hooks](#ci--infrastructure-hooks)  
14. [Credits](#credits)  

---

## Why this project exists

This repository supports **CMU PokerAI** work: teams implement bots that plug into a **shared engine**, compete under fixed rules, and iterate on **preflop sizing, discard policy, postflop logic, and opponent modeling**. The codebase grew across design generations—**heuristic and Monte Carlo lines, modular “street” engines, meta-chassis bots with live stats, Libratus-style table lookups, and combined “Frankenstein” stacks**—before converging on the current tournament submission while preserving earlier experiments for study and RL opponent pools.

If you are reading this as a **portfolio / showcase** repo: the interesting story is not only the final agent, but the **traceable evolution of ideas** (what we tried, what we kept, where code still lives).

---

## The game variant

The engine implements a **non-standard Hold’em** documented for players in `config/RULES.txt`. In short:

- **Deck:** 27 cards (no clubs, no faces; Ace high in the shortened rank set).  
- **Deal:** Each player receives **5** private cards.  
- **Flop:** Three community cards, then a **mandatory discard round**: each player keeps **2** of 5 and discards **3**; discards are **visible** to the opponent and removed for the hand.  
- **Betting:** Pre-flop, flop, turn, river—with engine-enforced action types, raises, and time/bankroll semantics used in scrimmages.

The environment implementation lives in **`gym_env.py`** (`PokerEnv`): observations, action tuples, and hand evaluation (including Ace-as-low handling where applicable via `WrappedEval`).

---

## Strategies we explored

Below is a **conceptual map** of major directions (details and file pointers live in `archive/other-bots/BOT_TYPES.md` and throughout `submission/`).

### Nit / Monte Carlo line (ALPHANiT-style)

Tight preflop selection, **Monte Carlo equity** for discard (evaluating keep combinations), threat-aware blocking, and conservative postflop adjustments (e.g. board-paired penalties, river caps). Good baseline for **low-variance** play and interpretable discard math.

### Exact / solver-leaning discard + rich postflop (OMICRoN family)

**Discard** treated as a serious optimization problem (exact / structured search where feasible), paired with **range- and board-aware** postflop: texture, semi-bluffing, dynamic sizing, and **opponent modeling**. OMICRoN V1.x lines informed later “Alpha killer” style play.

### Modular street engine (Genesis)

**Street-by-street** decomposition: dedicated logic for preflop scoring (`street0_score`), bet sizing, discard (street 1a), flop betting (1b), turn, river—wired to a shared **`OpponentRecon`** (VPIP/PFR, fold-to-bet, aggression, texture-linked reactions). See `docs/opponent-tracking.md` for how recon feeds Street 0.

### Meta chassis (METAV / Blambot / Omicron V1.3-style)

A **statistics-first** shell: track opponent tendencies and bankroll context, apply **safety caps and mode switches** (value / trap / aggro regimes), and integrate with strong postflop machinery. Useful when you want **explicit** exploitability signals.

### Libratus tables + discard engine (Lambda V2)

Hybrid of **lookup tables** (policy / equity / posteriors / matchups) and a shared **`discard_engine`** pipeline (made hands, draws, dead outs, heuristics). Explores **GTO-ish gradients** (modes like GTO/AGRO/VALUE/TRAP) with engineering focus on **speed and table-driven** decisions.

### DELTA (“Frankenstein”)

A deliberate **merge** of strengths: OMICRoN-style substrate, Genesis-style equity and river thinking, Meta-style safety and overrides, plus **adaptive learning** (EMA stats, Bayesian-style archetypes, per-street profiles, regime detection, bet-sizing tells). High complexity; useful as a **research maximum** in the archive.

### Ensemble / A/B switching

**Ensemble** wrappers allow running multiple bot “brains” with shared observation context and controlled switching—useful for **mid-match experiments** and safe rollouts.

### Current submission: **OMICRoN V2** (`submission/player.py`)

Documented in-file as **algorithmic fixes + online learning systems**, building on V1-class infrastructure with:

- Granular hand categories and **equity-aware** strength  
- Re-raise awareness  
- **EMA opponent stats** with regime detection  
- Per-street fold rates, **bet-sizing tells**, equity calibration  
- **Action-sequence danger** modeling  

Supporting assets (e.g. evaluation tables) live alongside the submission where applicable.

### Blueprint CFR / Libratus research (`agents/libratus/`)

An in-repo **abstraction + MCCFR** path (`scripts/train_libratus.py`) can generate a **blueprint strategy JSON** (large; gitignored by default). This is **research infrastructure**, not necessarily the same as the HTTP tournament agent.

### Reinforcement learning (`apps/poker-rl-trainer/`)

A separate pipeline with **behavioral cloning warm-up** and **PPO**-style settings (see `TrainingConfig` in `apps/poker-rl-trainer/config.py`), feature engineering from the env, and opponent diversity from archived bots—aimed at **learning policies** from data and self-play rather than hand-authored rules alone.

---

## What we built (systems)

| System | Role |
|--------|------|
| **Engine** | `gym_env.py`, `match.py`, `run.py` — env, API match loop, process orchestration |
| **Agents framework** | `agents/agent.py`, test agents, probability agent, **Libratus** modules |
| **Tournament submission** | `submission/player.py` (+ alternates, genesis modules, older subs) |
| **Human trainer app** | `apps/poker-bot-trainer/` — Next.js UI + Python `training/` scripts, session logging |
| **RL trainer** | `apps/poker-rl-trainer/` — config-driven BC + PPO training, opponent pool from `archive/other-bots/` |
| **Analysis** | `scripts/analyze_*.py`, `tools/monte_carlo.py`, `visualizer/` + Streamlit shim |
| **Docs** | `docs/MATCH_LOGGING.md`, `docs/opponent-tracking.md`, `config/RULES.txt` |
| **CI** | `.github/workflows/tests.yml` — pytest on core tests |

---

## Repository layout

| Path | Purpose |
|------|---------|
| **`run.py`**, **`match.py`**, **`gym_env.py`** | Core entrypoints; keep `gym_env` importable from repo root |
| **`config/`** | `agent_config.json` (bots, ports, CSV path), `RULES.txt` |
| **`outputs/`** | Default match CSV output directory (artifacts gitignored via `*.csv`) |
| **`submission/`** | **Canonical bot** (`player.py`) + variants, genesis files, `oldersubs/` |
| **`agents/`** | Shared agent base, heuristics, RL helpers, **Libratus** package |
| **`tests/`** | `api_test.py`, `engine_test.py`, `agent_test.py`, tournaments / validation |
| **`scripts/`** | Analyzers, short matches (`run_50_hands.py`), `train_libratus.py`, `create_release.sh`, manual checks |
| **`docs/`** | Deep dives (logging, opponent tracking) |
| **`genesis/`** | Preserved genesis-line experiments |
| **`apps/poker-bot-trainer/`** | Trainer web app + training scripts |
| **`apps/poker-rl-trainer/`** | RL training code + shell helpers |
| **`archive/`** | Old bots, `other-bots` pool, scratch — see `archive/README.md` |
| **`legacy/poker-engine-2026-work/`** | Full snapshot of parallel team exploration (Phoenix, bleed logic, alternate submissions, etc.) |
| **`tools/`** | Monte Carlo CLI, Streamlit log viewer source |
| **`visualizer/`** | Streamlit-style analysis app (`app.py`, parsers) |
| **`visualizer.py`** | Root shim → `streamlit run tools/visualizer_streamlit.py` |

**Note:** Older checkouts sometimes used a top-level `submissions/` folder; merged work lives on **`main`**, with historical trees under **`legacy/`** and **`archive/`**.

---

## Getting started

**Requirements:** Python **3.12+** (`pyproject.toml`).

```bash
python3.12 -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate

pip install -r requirements.txt
# or: uv sync
```

Optional: Node.js for the Next.js trainer (`apps/poker-bot-trainer/` — `npm install` / `npm run dev`).

---

## Running matches

1. **Edit** `config/agent_config.json` — set `bot0` / `bot1` `file_path` (Python module path to agent class), ports, and `match_settings` (hands, `csv_output_path`).  
2. **Run:**

```bash
python run.py
```

Default CSV path is typically `outputs/match.csv` (directory is created as needed).

**Short scrimmage (50 hands)** with the same config:

```bash
python scripts/run_50_hands.py
```

**Opponent mix example:** point `bot1` at `agents.libratus_agent.LibratusAgent` or another registered class.

---

## Configuration

| File | Use |
|------|-----|
| `config/agent_config.json` | Who plays, ports, match length, CSV path |
| `config/RULES.txt` | Human-readable tournament rules |

Rollout-related env toggles are set from `run.py` when present in config (`OMICRON_LIVE_STAGE`, shadow flags, etc.).

---

## Testing

**Focused CI-style tests:**

```bash
pytest tests/api_test.py tests/engine_test.py -v
```

**Coverage (example):**

```bash
pytest --cov=gym_env --cov-report=term-missing --cov-report=html --cov-branch
```

**Submission smoke test** (multi-opponent short runs):

```bash
python tests/agent_test.py
```

Pytest collection is scoped via `pyproject.toml` (`testpaths = ["tests"]`). Long-running tournament scripts under `tests/` can be invoked explicitly when needed.

---

## Training & data

### Poker Bot Trainer (human sessions)

- **Path:** `apps/poker-bot-trainer/`  
- **Idea:** Collect human decisions through a web UI, save sessions, and feed downstream training (see `training/` Python modules).  
- **Dev:** standard Next.js workflow (`npm run dev`).

### Poker RL Trainer

- **Path:** `apps/poker-rl-trainer/`  
- **Idea:** Behavioral cloning from logged behavior, then PPO-style RL against env + diverse opponents from `archive/other-bots/`.  
- **Entry:** `train.py`, `config.py` for hyperparameters; `run_training_detached.sh` for long jobs (paths assume repo layout).

---

## Analysis & visualization

| Tool | Command / note |
|------|----------------|
| **Monte Carlo / charts** | `python tools/monte_carlo.py --help` (artifacts default under `tools/`) |
| **Match CSV analyzers** | `python scripts/analyze_match.py`, `analyze_logs.py`, `analyze_for_bot.py` |
| **Streamlit log tail** | `streamlit run tools/visualizer_streamlit.py` or `python visualizer.py` |
| **Visualizer package** | `visualizer/app.py` — richer parsing of logs + CSVs |
| **Logging reference** | `docs/MATCH_LOGGING.md` |

---

## Archive, legacy, and history

- **`archive/`** — Intentionally **not** the default submission path; holds old bots, opponent-pool sources, and scratch. See [`archive/README.md`](archive/README.md).  
- **`legacy/poker-engine-2026-work/`** — Preserves a **whole parallel codebase** (alternate submissions, HRT paths, “bleed” experiments, tooling). Use it when you need **historical fidelity**, not day-to-day tournament runs.  
- **Repository hygiene** — Tracked layout was reorganized so the root stays readable (`config/`, `outputs/`, `scripts/`, `tests/`, `apps/`, `archive/`, `tools/`).  

---

## CI & infrastructure hooks

- **Tests workflow:** `.github/workflows/tests.yml` runs pytest on `tests/api_test.py` and `tests/engine_test.py`.  
- **Deploy hook:** `.github/workflows/deploy-engine.yml` can trigger **`repository-dispatch`** on `cmu-dsc/poker-infra-2026` when the **`ENGINE_PAT`** secret is configured; otherwise the job is skipped so forks and personal clones stay green.

---

## Credits

See **[CONTRIBUTORS.md](CONTRIBUTORS.md)** for human-readable credits. Git history remains the source of truth for authorship.

---

## License / course context

This project was built in an **academic competition** setting. If you reuse code or ideas, respect **course and partner attribution** requirements and any **third-party** licenses (e.g. Treys, PyTorch, Next.js).
