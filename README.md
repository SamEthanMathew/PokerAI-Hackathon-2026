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
| `submission/player.py` | Primary tournament agent (OMICRON V2). |
| `submission/` | Alternate and historical bots (`OMICRoN_V1.py`, genesis modules, etc.). |
| `submissions/` | Additional packaged bot variants. |
| `agents/` | Engine agents, RL/probability helpers, **Libratus** (`libratus_agent.py`, `libratus/`). |
| `legacy/poker-engine-2026-work/` | Full snapshot of the second team repo (`poker-engine-2026-work`), including Phoenix / bleed / `submission_v15`, `HRT_submission`, tools, and validation. |
| `genesis/` | Genesis-line experiments preserved from earlier branches. |
| `poker-bot-trainer/` | Next.js trainer UI and training scripts (from genesis line). |
| `tests/`, `scripts/` | Tests and utilities. |

`main` merges the histories of **poker-engine-2026-work** (under `legacy/…`) and feature branches **genesis-v1**, **ML/RL**, **libratus-agent**, **monte-carlo-analysis**, **omimax**, plus work branch **sigedit** (integrated via the legacy import and merge).

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

## Human vs Bot (CLI)

From the project root:

```bash
python -m human_vs_bot.run
```

Session logs go under `human_vs_bot/logs/`.
