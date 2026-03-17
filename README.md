#Team Bot

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

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

1. Basic coverage test:

```bash
pytest --cov=gym_env --cov-report=term-missing --cov-report=html --cov-branch
```

### Testing Your Submission

1. To test your agent (5 hands per bot) against ProbabilityAgent, AllInAgent, FoldAgent, CallingStationAgent, RandomAgent:

```bash
python agent_test.py
```

2. To run a full match (1000 hands) of your agent against a specific agent:

```bash
python run.py
```

You can modify which bots play by modifying the agent config file. Write the file path to the corresponding agent for that bot to play.

## Performance (match time limit)

Preflop (Street 0) scoring uses parallel workers to stay within the match time limit. Set **`POKER_N_WORKERS`** in the environment: use **2** for the current 2 vCPU setup (default), and **4** in the next phase when 4 vCPU are available.
