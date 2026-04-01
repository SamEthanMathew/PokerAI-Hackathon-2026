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

### Deep Learning line (data-generation + policy-learning pipeline)

Alongside the rule-based and solver-leaning bots, we also explored a **deep learning-oriented pipeline** built around one simple idea: use strong, diverse bots to generate large amounts of structured gameplay data, then train a model to imitate and generalize from those decisions.

This was not a pure end-to-end "throw states into a neural net and hope it works" system. Instead, the DL line was designed as a **supervised policy learner** on top of a strong engineered foundation. The handcrafted bots produced the strategy, the simulator produced the volume, and the model tried to compress those patterns into something faster, smoother, and potentially more adaptable.

#### Core idea

The main bottleneck in this game was not a lack of possible decisions, but the difficulty of making good decisions consistently under uncertainty. The discard phase, the reduced 27-card deck, revealed information, and the reset-every-hand structure all created a setting where local mistakes compounded quickly.

To address that, we used multiple bot variants as **teachers**:

- conservative / low-variance bots
- more aggressive exploitative bots
- variants with different discard preferences
- variants with different postflop thresholds and response logic

By letting these bots play large numbers of games against each other and against other baselines, we generated a dataset of decision points paired with strong actions, outcomes, and contextual features. The deep learning model then learned from that distribution.

#### Why we used multiple bots instead of one

Training on a single bot would have made the model mostly an imitator of one fixed style. That would have limited both generalization and robustness. Using multiple strong variants gave us a broader decision distribution:

- some spots were played for safety
- some were played for value extraction
- some were played as pressure or denial
- some were only good because of opponent context

That variety mattered. It gave the model exposure to a richer set of strategic patterns and reduced the chance that it would overfit to one narrow style of play.

In practice, this made the data-generation loop far more valuable than expected. The bots were originally deployed to produce training data, but at one point those same variants were also occupying the top three leaderboard spots, which was a strong signal that the generated data was coming from genuinely competitive policies rather than weak synthetic play.

#### Data generation pipeline

The DL workflow started with self-play and cross-play.

##### 1. Run bot leagues

We deployed multiple bot variants and had them play repeated matches against:

- each other
- previous archived versions
- simpler baselines
- targeted opponent styles when available

This produced a large set of state-action trajectories across many strategic regimes.

##### 2. Log structured decision states

For each decision point, we recorded the game state in a machine-readable form. The exact schema can vary, but conceptually it included:

- private hand information
- public board state
- discard information and revealed signals
- betting history and action sequence
- pot / bankroll context
- street identifier
- opponent behavior summaries
- equity estimates or strength proxies from the engine
- final chosen action
- eventual hand outcome when relevant

This mattered because raw cards alone were not enough. A good action depended heavily on context: street, opponent type, prior aggression, reveal signals, and risk posture.

##### 3. Build labels from high-quality actions

The training labels came from the actions selected by the strongest available bot logic at the time. Depending on the experiment, this could mean:

- direct action labels such as `fold`, `check`, `call`, `bet_small`, `bet_large`, `raise`
- discard-choice labels for which subset of cards to keep
- value targets such as estimated EV, win probability, or calibrated equity bucket
- auxiliary labels such as opponent archetype or board texture class

This let the model learn both **what to do** and, in some experiments, **why the state looked favorable or dangerous**.

#### Model objective

The deep learning line was most naturally framed as a **policy approximation** problem.

Given a state representation `s`, the model predicts either:

1. the best action directly, or  
2. a distribution over actions, where higher probability corresponds to stronger choices under that state.

In some variants, the model could also predict auxiliary outputs such as:

- hand strength bucket
- opponent aggression class
- probability of continuing profitably
- discard quality score
- risk level for the current line

These auxiliary targets were useful because they forced the model to learn internal structure instead of blindly memorizing surface patterns.

#### Input representation

A major design choice was how to represent game state. The strongest version was not just "cards in, action out." It combined several feature groups:

##### Hand and board features
- encoded private cards
- encoded board cards
- made-hand indicators
- draw indicators
- blockers / suit structure / pair structure
- discard-related combinational features

##### Betting context
- current street
- pot size
- effective commitment / risk posture
- prior action sequence
- bet-size ratios
- whether the line showed strength, weakness, or capped range behavior

##### Opponent features
- VPIP / PFR-style behavior stats
- fold-to-bet tendencies
- aggression metrics
- discard tendencies
- sizing tells
- recent regime shifts or short-term EMA-based behavior summaries

##### Engine-derived features
- Monte Carlo equity estimates
- calibrated equity bucket
- danger / pressure score
- board texture class
- sequence-based threat indicators

This hybrid representation was important. The model worked best when it did not have to rediscover everything from scratch. Engine features gave it strong priors, while learned layers helped combine them more flexibly.

#### Training loop

The training process followed a practical cycle:

##### Step 1: Generate data
Run a large volume of matches using multiple bot variants.

##### Step 2: Filter and clean
Remove broken logs, inconsistent states, and low-information samples. In some cases, downweight trivial spots and upweight higher-leverage decisions.

##### Step 3: Train supervised models
Train a model to predict the teacher action from the recorded state. Typical loss functions would be cross-entropy for action selection and MSE or ranking loss for value-style targets.

##### Step 4: Evaluate against held-out matches
Check whether the learned model reproduced strong decisions on unseen samples and whether it could compete in live play.

##### Step 5: Redeploy and iterate
Use the improved model or updated bot population to generate new data, then retrain. This created a feedback loop where better policies generated better training examples.

That feedback loop was one of the most valuable parts of the system. It made progress feel compounding rather than linear.

#### Where the DL approach helped

The learned model was most useful in places where purely handwritten logic became brittle or too fragmented.

##### 1. Smoother decision boundaries
Handwritten thresholds often create discontinuities. A learned model can interpolate between similar states more naturally.

##### 2. Better compression of many interacting signals
Opponent stats, board texture, sequence danger, and equity estimates all interact. A neural model can combine these signals without requiring hundreds of manually tuned if-statements.

##### 3. Faster approximation of expensive reasoning
If trained well, the model can approximate the output of heavier logic more quickly, which matters in repeated or latency-sensitive settings.

##### 4. Better generalization across adjacent states
Instead of memorizing exact hand categories, the model can learn strategic shape: when pressure is credible, when thin value is justified, and when revealed information changes the range interaction.

#### Where the DL approach was limited

The deep learning line was promising, but it was not magic.

##### Data quality ceiling
The model could not surpass the quality of the data distribution unless it learned genuinely better abstractions. If the teacher bots had blind spots, the model could inherit them.

##### Distribution shift
A model trained on one population of opponents might degrade against very different styles.

##### Interpretability
Compared to handcrafted logic, neural policies are harder to debug. When performance drops, it is often less obvious whether the issue comes from the representation, label quality, sampling bias, or the model itself.

##### Adversarial environments
Because this is a strategic game, strong opponents can adapt. A learned policy that performs well against the training mix can still be exploited if it becomes too predictable.

#### Best way to think about this line

The DL algorithm worked best not as a replacement for the engineered system, but as an additional layer on top of it.

The rule-based and solver-inspired bots gave us:

- structure
- strong priors
- interpretable decisions
- reliable data generation

The deep learning layer gave us:

- policy compression
- smoother generalization
- the ability to learn from many interacting signals
- a framework for turning large-scale bot self-play into a reusable model

So the real value of the DL line was not that it magically solved poker by itself. The value was that it converted a growing archive of competitive gameplay into a trainable decision model. In that sense, it acted as a bridge between handcrafted game-theoretic intuition and scalable learned policy behavior.

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
