# Match data logging in the bot engine

This doc explains **where** match data is logged and **how** to add or use it for any bot (including ALPHANiTV6/V7, DELTA V2, METAV5).

---

## 1. Engine-level logging (every match)

**File:** `match.py`  
**When:** Every match run via `run.py` (or any caller of `run_api_match()`).

The **match runner** writes a single CSV file with one row per **action** (not per hand):

- **Config:** `config/agent_config.json` → `match_settings.csv_output_path` (e.g. `outputs/match.csv`).
- **Contents:** `hand_number`, `street`, `active_team`, `team_0_bankroll`, `team_1_bankroll`, `action_type`, `action_amount`, `action_keep_1`, `action_keep_2`, `team_0_cards`, `team_1_cards`, `board_cards`, `team_0_discarded`, `team_1_discarded`, `team_0_bet`, `team_1_bet`, plus telemetry columns through `final_action_amount`.
- **Written in:** `play_hand()` → `writer.writerow(current_state)` after each env.step().

So **every** scrimmage (any two bots) produces this CSV. No bot code is required; it’s the engine’s log.

---

## 2. Base Agent logger (all bots)

**File:** `agents/agent.py` → `_setup_logger()`  
**When:** Each bot process starts (e.g. when `run.py` spawns bot processes).

Every bot that extends `Agent` gets:

- **`self.logger`** – a Python `logging.Logger` with:
  - **File handler:** `agent_logs/match_{MATCH_ID}_{PLAYER_ID}.log`
  - **Env:** `MATCH_ID` and `PLAYER_ID` from environment. `run.py` does **not** set `MATCH_ID`; it only passes `player_id` into `Agent.run(..., player_id=...)`, which sets `PLAYER_ID`. So typically: `agent_logs/match_unknown_bot0.log`, `agent_logs/match_unknown_bot1.log` (or whatever `player_id` is in config).

So **any** bot can log match-related data by calling:

- `self.logger.info("...")`  → goes to that file (and console if `stream=True`).

**Who uses it today:**

- **METAV5** (`submissions/METAV5.py`): logs hand start (mode, pnl, stats), every 50-hand stats, overrides (check_raise, reraise), and postflop RAISE/FOLD with equity. Human-readable lines, not a structured “HAND_RESULT” format.
- **ALPHANiTV6 / ALPHANiTV7** (`submissions/ALPHANiTV6.py`, `ALPHANiTV7.py`): do **not** call `self.logger.info()` in the current code. So they don’t write bot-side match data to a file; the only record of the match from their perspective is the **engine CSV** above.

So “ALPHANiTV7 and V6 logged data” in practice means: when you run a match with them, the **engine** logs the CSV. The bots themselves don’t write their own log lines unless we add them.

---

## 3. DELTA V2 custom logger (bot-specific)

**File:** `submission/player.py` (DELTA V2) – class **`_MatchLogger`** (lines ~25–115).

**When:** Only when the **DELTA V2** bot is used in a match.

- **Per-decision:** appends one JSON line per decision to **`logs/delta_v2_decisions.jsonl`** (hand, street, pot, my_bet, opp_bet, equity, hand_cat, rules, action, raise_amt, dt_ms).
- **Per-hand:** `log_hand_result(pnl, hand_num)` updates in-memory stats (hands, pnl, showdowns_won/lost).
- **End of match / every 100 hands:** `flush_summary()` writes **`logs/delta_v2_summary.json`** (hands, pnl, rule_fires, folds_by_street, raises_by_street, showdowns_won/lost, total_time).

**Usage inside DELTA V2:**

- `act()`: `start_decision()` at start, `log_rule()` when a rule fires, `finish_decision(action, elapsed_ms)` before each return.
- `observe()`: `log_hand_result(int(reward), self._hands_completed - 1)` when `terminated`, and `flush_summary()` every 100 hands.

This is the only bot that writes **structured** match data (JSON/JSONL) to `logs/`.

---

## 4. How to log match data from any bot (e.g. ALPHANiTV6/V7)

You have two straightforward options.

### Option A: Use the base logger (no new files)

In **`observe()`**, when **`terminated`** is True, log whatever you need for “match data”:

```python
def observe(self, observation, reward, terminated, truncated, info):
    if terminated:
        self._running_pnl += int(reward)
        self._hands_completed += 1  # if you track it
        # Log one line per hand for later parsing / analysis
        self.logger.info(
            f"HAND_RESULT | hand={self._hands_completed} | reward={int(reward)} | "
            f"pnl={self._running_pnl}"
            # add position, street_ended, end_type, etc. if your obs/info provide them
        )
```

- **Destination:** `agent_logs/match_{MATCH_ID}_{PLAYER_ID}.log`.
- To get **per-bot** logs when both bots are ALPHANiT (or mixed), use different `player_id` in config (e.g. `bot0` / `bot1`) so each process gets its own file.

You can standardize the format (e.g. pipe-separated `key=value`) so a script or visualizer can parse it later (e.g. “HAND_RESULT” lines).

### Option B: Add a small file logger (like DELTA V2)

- Add a small helper (e.g. a `_MatchLogger` or a single function) that:
  - Opens a file in `logs/` (e.g. `logs/alphanit_v7_decisions.jsonl` or `.log`) in append mode.
  - On each hand end in `observe(..., terminated=True)`, write one line (JSON or pipe-separated).
- Call it from **`observe()`** when **`terminated`** is True (and optionally from **`act()`** if you want per-decision lines).
- Keep the same pattern as DELTA V2: one line per hand (and optionally per decision), plus optional summary file if you want.

That way ALPHANiTV6/V7 (or any bot) can “log data as well” in a structured way for scrimmages.

---

## 5. Summary table

| Source              | What is logged                          | Where                                      | When / which bot      |
|---------------------|-----------------------------------------|--------------------------------------------|------------------------|
| **match.py**        | One row per action (full state)         | CSV path from config (e.g. `outputs/match.csv`) | Every match, any bots  |
| **Agent base**      | Whatever bot writes with `self.logger`  | `agent_logs/match_{MATCH_ID}_{PLAYER_ID}.log` | Any bot that calls it  |
| **METAV5**          | Mode, stats, RAISE/FOLD lines            | Same as Agent base                         | When METAV5 is used    |
| **ALPHANiTV6/V7**   | Nothing (no logger calls)               | —                                          | Only engine CSV        |
| **DELTA V2**        | Per-decision JSON + summary             | `logs/delta_v2_decisions.jsonl`, `delta_v2_summary.json` | When DELTA V2 is used   |

So: the **engine** always logs the match CSV; **DELTA V2** also logs structured bot data to `logs/`; **METAV5** logs to the base Agent log file; **ALPHANiTV6/V7** currently only “log” via the engine CSV, unless we add `self.logger.info(...)` or a small file logger in their `observe()` (and optionally `act()`).
