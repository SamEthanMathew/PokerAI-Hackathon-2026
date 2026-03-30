# Opponent Tracking (genesis)

This doc explains how the `genesis` agent keeps track of opponent behavior, where those stats live, how they are updated during play, and how they feed into Street 0 scoring/strategy.

Primary files:
- `submission/opponent_recon.py` (data structure + update/getter functions)
- `submission/genesis.py` (calls recon update functions during the hand)
- `submission/street0_score.py` (turns a subset of recon stats into `OpponentProfile`, then adjusts Street 0 scoring)

## 1. Data flow overview

### 1.1 Where opponent stats are stored

- A single `OpponentRecon` instance is created per agent/match in `GenesisAgent.__init__`:
  - `self.recon = OpponentRecon()`
- All opponent tracking counters are stored as fields on that `OpponentRecon` object.

### 1.2 When those stats are updated

`GenesisAgent.act()` and `GenesisAgent.observe()` update recon based on `observation["street"]` and `observation["opp_last_action"]`.

Key update calls in `submission/genesis.py`:
- Start-of-hand/state:
  - `start_new_hand(self.recon, hand_number)` when `hand_number` changes
  - `start_new_street(self.recon, street)` when `street` changes
- Ongoing generic updates (every `act()` call):
  - `update_opponent_actions(self.recon, opp_last_action)`
  - `update_vpip_pfr(self.recon, observation, street)`
  - `update_aggression_flags(self.recon, observation, street)`
- Street-specific updates when betting actions occur:
  - Street 1:
    - `update_opponent_flop_response(self.recon, opp_last_action)` (response to our flop bet)
    - `update_opponent_flop_aggression(self.recon, opp_last_action)` (their bet vs check on flop, keyed by their inferred discard class)
  - Street 2:
    - `update_opponent_turn_response(self.recon, opp_last_action, self._street1_discard_class)`
  - Street 3:
    - `update_opponent_river_response(self.recon, opp_last_action, self._street1_discard_class)`
    - `update_opponent_river_aggression(self.recon, True, opp_last_action)` when we checked river
- Hand termination:
  - `update_fold_on_terminate(self.recon, street, opp_last_action, terminated)` to increment fold-to-bet counts

### 1.3 How recon becomes Street 0 opponent profile

Before choosing the Street 0 action, `GenesisAgent._act_street0()` computes:
- `opponent_profile = to_opponent_profile(self.recon)`

Then `final_street0_score()` uses that `OpponentProfile` to adjust the base Street 0 hand score (via `opponent_adjusted_score()` and a confidence-weighted blend).

`to_opponent_profile()` in `submission/opponent_recon.py` maps a subset of `OpponentRecon` fields into `street0_score.OpponentProfile`.

## 2. What we track: `OpponentRecon` stats

`OpponentRecon` stores raw counts and per-hand attribution state. Stats are grouped below by how they are used by getters/strategy.

### 2.1 Preflop participation + aggression (lifetime counters)

These are updated by:
- `update_vpip_pfr()` (with per-hand “counted this hand” guards)
- `update_opponent_actions()` (raise/call counters)

Tracked fields:
- `total_hands`
  - Set in `start_new_hand()` (note: it is set to `hand_number`)
- VPIP:
  - `opp_vpip_count`
  - Used denominator: `total_hands` (in `get_vpip()`)
- PFR:
  - `opp_pfr_count`
  - Used denominator: `total_hands` (in `get_pfr()`)
- Aggression (raw):
  - `opp_raise_count`
  - `opp_call_count`
  - Used by `get_aggression_factor()` as `opp_raise_count / opp_call_count` (capped; special-case when no calls)

How they are detected:
- VPIP success:
  - If `opp_last_action` is `"raise"` on street 0 (counts as VPIP)
  - If `opp_last_action` is `"call"` on street 0 (with a blind-aware condition)
- PFR success:
  - If `opp_last_action` is `"raise"` on street 0 (counts via a per-hand guard)
- Raises/calls:
  - `update_opponent_actions()` increments counters whenever `opp_last_action` contains `"raise"` or `"call"` (no per-hand guard)

### 2.2 Non-river betting frequency + fold-to-bet (lifetime counters)

These come from:
- `update_vpip_pfr()` for “seen” and “bet/raise on flop/turn”
- `record_our_bet()` + `update_fold_on_terminate()` for fold-to-our-bet

Tracked fields:
- Non-river “seen” opportunities:
  - `opp_non_river_streets_seen` (incremented once per hand per street, in `update_vpip_pfr()`)
- Non-river betting:
  - `opp_non_river_bet_count` (counts opponent bet/raise actions on flop/turn)
  - Getter: `get_non_river_bet_percentage()` = `opp_non_river_bet_count / opp_non_river_streets_seen` (smoothed/defaulted)
- Facing our bet on flop/turn:
  - `opp_non_river_bets_faced` incremented by `record_our_bet()` the first time we bet/raise each street this hand
- Folding to our bet on flop/turn:
  - `opp_non_river_folds` incremented by `update_fold_on_terminate()` when opponent’s last action includes `"fold"` and we had bet this street
- Facing our bet on river:
  - `opp_river_bets_faced` incremented by `record_our_bet()` when we bet/raise river
- Folding to our bet on river:
  - `opp_river_folds` incremented by `update_fold_on_terminate()`

Derived getters used by strategy:
- `get_fold_to_non_river_bet()`
- `get_fold_to_river_bet()`
- `get_non_river_bet_percentage()`
- `get_opponent_type()` (tight/loose/balanced based on fold-to-non-river)

### 2.3 “We checked river” aggression (lifetime counters)

Tracked by:
- `update_opponent_river_aggression()`

Fields:
- `river_we_checked_opp_bet_count`
- `river_we_checked_opp_check_count`

Used by:
- `get_river_bet_when_checked_to()` = bet rate when we checked river

### 2.4 Flop response to our bet (per-discard-class, per-texture, per-size)

These are updated when:
- our action is on flop (Genesis calls `record_our_flop_discard_class()` and `record_flop_texture()`)
- and later, when opponent makes a betting action on flop:
  - `update_opponent_flop_response(self.recon, opp_last_action)`

What gets incremented:

1) Keyed by our inferred flop discard class:
- `react_<our_discard_class>_flop_faced`
- `react_<our_discard_class>_flop_fold`
- `react_<our_discard_class>_flop_call`
- `react_<our_discard_class>_flop_raise`

2) Keyed by flop texture bucket (`_texture_bucket()` maps to `paired/suited/connected/disconnected`):
- `flop_tex_<tex>_faced`
- `flop_tex_<tex>_fold`
- `flop_tex_<tex>_raise`

3) Keyed by our flop bet size bucket (`small/medium/large`):
- `flop_bet_<bucket>_faced`
- `flop_bet_<bucket>_fold`
- `flop_bet_<bucket>_call`
- `flop_bet_<bucket>_raise`

Important limitation:
- `OpponentRecon` only has counters for specific discard classes in `_DISCARD_CLASS_LABELS`:
  - `flush_transparent`, `straight_transparent`, `pair_transparent`, `weak_transparent`, `ambiguous`, `capped`
  - If Street 1A’s discard classification ever returns a class outside this set, those counters won’t exist and the related flop exploitation features will fall back to default rates (0.5).

### 2.5 Flop aggression keyed by opponent’s inferred discard class

Updated by:
- `update_opponent_flop_aggression(self.recon, opp_last_action)`

Tracked fields (pattern):
- `opp_<their_discard_class>_discard_bet_count`
- `opp_<their_discard_class>_discard_check_count`

Getter:
- `get_opponent_flop_aggression_after_discard(recon, their_discard_class)`

Important limitation:
- `their_discard_class` is inferred from revealed opponent discard cards, and recon only classifies opponent discard when `opp_discard3` is available in observations.
  - If the agent cannot see opponent’s discard cards in a given situation (depends on blind position and what the observation includes), the keyed counters for `opp_<their_discard_class>_*` may not be updated.

### 2.6 Turn response to our bet

Updated by:
- `update_opponent_turn_response(self.recon, opp_last_action, our_discard_class=...)`

The turn response is attributed to:
1) Our turn bet size bucket:
- `turn_bet_<small|medium|large>_faced`
- `turn_bet_<...>_fold`
- `turn_bet_<...>_call`
- `turn_bet_<...>_raise`

2) Our discard class (from Street 1A):
- `turn_react_<our_discard_class>_faced`
- `turn_react_<...>_fold`
- `turn_react_<...>_call`
- `turn_react_<...>_raise`

3) Turn texture bucket (`paired/flush/straight/blank`):
- `turn_tex_<bucket>_faced`
- `turn_tex_<bucket>_fold`
- `turn_tex_<bucket>_raise`

4) Turn aggression after the flop action (hand-state attribution):
- `turn_after_flop_call_fold`
- `turn_after_flop_call_bet`
- `turn_after_flop_bet_called_fold`
- `turn_after_flop_bet_called_bet`
- `turn_after_flop_check_fold`
- `turn_after_flop_check_bet`

### 2.7 River response to our bet

Updated by:
- `update_opponent_river_response(self.recon, opp_last_action, our_discard_class=...)`

The river response is attributed to:
1) Our river bet size bucket:
- `river_bet_<small|medium|large>_faced`
- `river_bet_<...>_fold`
- `river_bet_<...>_call`
- `river_bet_<...>_raise`

2) Our discard class:
- `river_react_<our_discard_class>_faced`
- `river_react_<...>_fold`
- `river_react_<...>_call`
- `river_react_<...>_raise`

3) River texture bucket:
- `river_tex_<flush_river|straight_river|standard_river>_faced`
- `river_tex_<...>_fold`
- `river_tex_<...>_raise`

## 3. What we track: `OpponentProfile` stats (Street 0)

`submission/street0_score.py` defines `OpponentProfile` as a dataclass containing:
- Raw preflop and aggression stats (derived rates come from smoothed Beta-posteriors)
- Fold-to-bet stats (non-river and river)
- A few “profile indices” that adjust Street 0 score:
  - pressure / foldability / stickiness / transparency_exploitation / discard_weakness / volatility
- An overall model confidence `opponent_confidence()`

### 3.1 How recon fields map into `OpponentProfile`

`to_opponent_profile()` in `submission/opponent_recon.py` sets only a subset of `OpponentProfile` fields.

Mapped fields include:
- VPIP/PFR:
  - `vpip_opportunities = recon.total_hands`
  - `vpip_successes = recon.opp_vpip_count`
  - `pfr_opportunities = recon.total_hands`
  - `pfr_successes = recon.opp_pfr_count`
- Aggression:
  - `raise_count = recon.opp_raise_count`
  - `call_count = recon.opp_call_count`
- Non-river bet/ fold-to-bet:
  - `non_river_bet_opportunities = recon.opp_non_river_streets_seen`
  - `non_river_bet_successes = recon.opp_non_river_bet_count`
  - `fold_non_river_opportunities = recon.opp_non_river_bets_faced`
  - `fold_non_river_successes = recon.opp_non_river_folds`
  - `fold_river_opportunities = recon.opp_river_bets_faced`
  - `fold_river_successes = recon.opp_river_folds`
- Some additional “discard reaction” and board-texture fields:
  - `paired_board_*`, `suited_board_*`
  - `react_<pair|flush|ambiguous>_keep_*`
  - `preflop_<limp/open>_<faced|fold|call|raise>`

### 3.2 Important implementation gap (affects effectiveness)

In the current `opponent_recon.py`, the additional `OpponentProfile` fields that relate to:
- `paired_board_faced` / `suited_board_faced`
- `react_pair_keep_*`, `react_flush_keep_*`, `react_ambiguous_faced` / fold / raise
- `preflop_*_faced` and their fold/call/raise splits

are present as dataclass fields, but there are no explicit update increments found elsewhere in `opponent_recon.py` for those exact counters.

As a result, those parts of `OpponentProfile` are effectively “untrained” (stay near defaults), so Street 0’s adjustment indices that depend on them will often behave close to priors.

## 4. How the opponent stats turn into decisions

### 4.1 Street 0 scoring adjustment

`final_street0_score()` blends:
- base hand score (`s_base`)
- opponent-adjusted hand score (`s_opp`)

with a confidence-weighted mixing:
- `s_final = (1 - lam) * s_base + lam * s_opp`
- where `lam = opponent_confidence(opponent_profile)` (capped to 0.80)

`OpponentProfile` influences `s_opp` through:
- `pressure_index()`
- `foldability_index()`
- `stickiness_index()`
- `transparency_exploitation_index()` (depends on the “keep reaction” counters noted above)
- `discard_weakness_index()`
- `volatility_index()`

### 4.2 Street 1/2/3 strategy exploitation inputs

Street 1B / Street 2 / Street 3 strategies query recon through its getters, especially:
- VPIP/PFR, fold-to-bet, non-river bet frequency, aggression factor (Street 0 sizing + scoring)
- flop/turn/river fold/raise rates conditioned on:
  - our discard class
  - flop/turn/river texture bucket
  - our bet size bucket
  - turn aggression after flop line

Defaults:
- Most getters return 0.5 (or `_DEFAULT_RATE=0.5`) when no data exists yet.
- Flop texture/size keyed getters use smoothed Beta-posteriors to avoid 0/1 extremes early in the match.

## 5. Effectiveness score (subjective)

Score: **6/10**

Why it’s decent:
- The core “classic” opponent descriptors (VPIP, PFR, aggression factor, fold-to-bet rates) are tracked end-to-end and used in Street 0 scoring/sizing.
- Street 1B/2/3 exploit stats are richer than just a single VPIP/AF number: they are conditioned on discard class, bet size bucket, and board/texture buckets.
- The recon update model resets per-hand/per-street state to prevent obvious double-counting.

Why it’s not higher:
- Several `OpponentProfile` fields (notably board-texture/discard-reaction “keep” counters) appear not to be incremented anywhere in `OpponentRecon`, so Street 0 indices that depend on them likely stay near priors.
- Some flop aggression counters keyed by opponent discard class depend on opponent discard being visible in observations; when opponent discard is unavailable, those exploit features don’t learn.
- Any discard class values outside `_DISCARD_CLASS_LABELS` won’t map to counters and will silently fall back to 0.5.

## 6. Suggested follow-ups (if you want to improve it)

1. Implement the missing updates for `OpponentProfile`-only counters:
   - `paired_board_*`, `suited_board_*`, and `react_*_keep_*`
   - `preflop_*` split counters by limp/open size (if they are intended)
2. Ensure opponent discard classification happens in all relevant observable scenarios (or add alternate inference when opponent discard cards are hidden).
3. Add a debug/log step that reports which recon counters are non-zero after N hands, so missing-stat issues don’t go unnoticed.

