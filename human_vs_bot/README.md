# Human vs Bot

Play poker against any submission bot from the CLI. Every decision is logged for analysis and RL training.

## Run

From the **project root**:

```bash
python -m human_vs_bot.run
```

You will be asked to:
1. Select a bot (numbered list from `submission/`).
2. Enter number of hands (or 0 for "until you quit").

Human is **player 0**; the bot is **player 1**.

## Features

- **Discard prompt**: On the flop discard, you see board, your 5 cards, opponent discards (if you are SB), known card count (e.g. "You know 11/27 cards"), and analysis lines for each possible 2-card keep (hand category, flush/straight outs). Optional: "What do you think they kept?" and "Why this keep?" (tags: flush_draw, straight_draw, trips, top_pair, blocker, draw_dead).
- **Betting prompt**: Street, pot, bets, valid actions. Optional: reason (value/bluff/pot_odds/…), confidence (1–5), opponent read (weak/strong/neutral/unknown).
- **Post-hand review**: After each hand you can say if you would change a decision and annotate which street and how.
- **Logging**:
  - **Decisions CSV** (`human_vs_bot/logs/decisions_<timestamp>.csv`): All match.py-style columns plus actor, decision_time_sec, time_used_player_0/1, human_reason, human_confidence, human_read, human_discard_reason, opp_keep_inference, valid_actions, cards_known_count, cards_known_list, bot_recommendation_keep.
  - **RL JSONL** (`human_vs_bot/logs/rl_<timestamp>.jsonl`): One JSON object per step with hand_id, step_in_hand, player, obs, action, reward, done, **derived_state** (hand_category, flush_outs, straight_outs, cards_known, opp_discard_bucket, board_texture, position, pot_odds, effective_stack), is_human, info.
  - **Session summary** (`human_vs_bot/logs/session_<timestamp>_summary.json`): total_hands, bankrolls, time_used per player, post_hand_annotations.

## Derived state (RL)

Each RL row includes a `derived_state` object so the model can use precomputed features: hand_category, flush_outs, straight_outs, cards_known, opp_discard_bucket, board_texture, position, pot_odds, effective_stack. This supports training a small discard policy and a small betting policy separately.
