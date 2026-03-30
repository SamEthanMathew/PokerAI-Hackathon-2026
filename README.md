# Poker engine 2026 (CMU PokerAI)

## How to run the engine

1. Create a virtual environment:

   ```bash
   python3.12 -m venv .venv
   ```

2. Activate the virtual environment:
   - On Windows:

     ```bash
     .venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source .venv/bin/activate
     ```

3. Install dependencies (pick one):

   ```bash
   pip install -r requirements.txt
   ```

   Or with uv:

   ```bash
   uv sync
   ```

Python **3.12+** is required (`pyproject.toml`).

## Repository layout

| Area | Purpose |
|------|---------|
| **Core (repo root)** | `run.py`, `match.py`, `gym_env.py`, `agent_test.py`, `agent_config.json` — engine and quick runs. |
| `submission/` | Tournament entry (`player.py` = OMICRON V2) plus alternate bots (`OMICRoN_V1.py`, genesis modules, etc.). |
| `agents/` | Built-in agents, RL/probability helpers, **Libratus** (`libratus_agent.py`, `libratus/`). |
| `tests/`, `scripts/`, `docs/` | Tests, tooling scripts, documentation. |
| `genesis/` | Genesis-line experiment code preserved from earlier work. |
| `legacy/poker-engine-2026-work/` | Full snapshot of the second team repo (Phoenix / bleed / `submission_v15`, HRT, validation, etc.). |
| `apps/poker-bot-trainer/` | Next.js trainer UI + Python training scripts (`training/`). |
| `apps/poker-rl-trainer/` | RL training pipeline; reads human session data from `apps/poker-bot-trainer/…` and loads opponents from `archive/other-bots/`. |
| `archive/` | Historical bots, scratch notes, and experiments — **not** the default submission path. See `archive/README.md`. |
| `tools/` | Standalone utilities: `monte_carlo.py` (equity / batch charts), `visualizer_streamlit.py` (Streamlit log viewer). |
| `visualizer/` | Separate analysis app (`app.py`, parsers). |
| `visualizer.py` (root) | Thin shim: runs `streamlit run tools/visualizer_streamlit.py`. |

**Packaged bot variants** that lived under a top-level `submissions/` folder in older checkouts now live under **`legacy/poker-engine-2026-work/`** (and similar paths inside `archive/` where copied). Search the tree if you are looking for a specific bot file.

**Consolidation:** All team work lives on **`main`**. Former GitHub feature branches were merged then removed so `main` does not drift from stale branch tips.

**`.claude/`:** Not tracked (see `.gitignore`); removed from Git history in a one-time cleanup.

**Credits:** See [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Running tests

```bash
pytest --cov=gym_env --cov-report=term-missing --cov-report=html --cov-branch
```

### Testing your submission

1. Quick multi-bot check (5 hands each):

   ```bash
   python agent_test.py
   ```

2. Full match (e.g. 1000 hands) via config:

   ```bash
   python run.py
   ```

Configure opponents in `agent_config.json` (module paths for each bot).

## Optional tools

- **Monte Carlo CLI** (outputs under `tools/`):

  ```bash
  python tools/monte_carlo.py --random --visualize
  ```

- **Streamlit log viewer** (expects `logs/engine_log.txt`):

  ```bash
  streamlit run tools/visualizer_streamlit.py
  ```

  Or: `python visualizer.py` from the repo root (shim).
