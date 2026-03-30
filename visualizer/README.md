# Match analysis visualizer

Drop match data files and run the dashboard to see where you lost, opponent recon over time, strategy shift triggers, and read accuracy.

## Files to drop

1. **Match runner log** – e.g. `match_25757.txt`  
   From the match runner: lines like `Hand number: 50, Team0 bankroll: -432, Team1 bankroll: 432`, `Final results`, `Time used`.

2. **Match CSV** – e.g. `match_25770.csv`  
   From [match.py](https://github.com/.../match.py): per-action CSV with `hand_number`, `street`, `active_team`, `action_type`, `team_0_bankroll`, `team_1_bankroll`, cards, bets. First line can be `# Team 0: X, Team 1: Y`.

3. **Bot log** – e.g. `match_25757_0.log`  
   From `agent_logs/match_{match_id}_{player_id}.log`: genesis logs with `HAND_RESULT`, `OPP_RECON`, and optionally `STREET0`/`STREET1`/`STREET2`/`STREET3` (pipe-separated `key=value`).

Put them in `visualizer/data/` or set the paths in the app.

## Run

```bash
# From project root
pip install -r visualizer/requirements.txt
streamlit run visualizer/app.py
```

Or from the visualizer directory:

```bash
cd visualizer
pip install -r requirements.txt
streamlit run app.py
```

## Config

- **We are Team 0 / Team 1**: Our bot does not log its player id. Choose which side we are so bankroll and “opponent” align with the CSV. Our bankroll = that team’s bankroll; opponent actions = rows where `active_team` is the other team.

## Tabs

- **Overview**: Bankroll over hands (from match log), final result, time used, hand counts.
- **Losses & mistakes**: Invalid-action hands, we-folded-in-big-pot list, loss breakdown by street/end_type/position, table of losing hands.
- **Opponent recon**: VPIP, PFR, AF over hands from OPP_RECON; annotated strategy shifts (e.g. vpip jump).
- **Read accuracy**: Our VPIP/PFR vs actual (from CSV, trailing 50 hands); table of our read vs actual.
- **Hand drill-down**: Pick a hand; show HAND_RESULT, OPP_RECON, and CSV actions for that hand.

## Match semantics (for correct parsing)

- Blind rotation: hand N has SB = `N % 2`, BB = `1 - N % 2`.
- CSV bankrolls in each row are at **start** of that hand. Bankroll **after** hand N = first row of hand N+1 (or match log checkpoint).
- Match log “Hand number: K” = bankrolls **after** hand K completed.
