# Genesis match analysis report (M1)

- Total hands: 1000
- Total reward: -221.0

## Loss by (street_ended, end_type)

| street_ended | end_type | count | total_reward |
|--------------|----------|-------|--------------|
| 0 | 'they_fold' | 449 | 488.0 |
| 1 | 'they_fold' | 270 | 551.0 |
| 4 | 'showdown' | 151 | -1528.0 |
| 3 | 'they_fold' | 93 | 509.0 |
| 3 | 'we_fold' | 14 | -144.0 |
| 0 | 'we_fold' | 10 | -20.0 |
| 1 | 'we_fold' | 7 | -48.0 |
| 4 | 'showdown_tie' | 4 | 0.0 |
| 2 | 'we_fold' | 2 | -29.0 |

## Loss by position

| position | count | total_reward |
|----------|-------|--------------|
| 'SB' | 500 | -158.0 |
| 'BB' | 500 | -63.0 |

## We folded in big pots (pot >= 50)

| hand | pot | reward | position | street_ended | flop_line | our_discard_class |
|------|-----|--------|----------|--------------|-----------|-------------------|
| 1 | 114 | -14.0 | 'BB' | 1 | 'we_checked_opp_raise' | 'flush_transparent' |
| 24 | 113 | -13.0 | 'SB' | 3 | 'we_checked_opp_check' | 'pair_transparent' |
| 50 | 110 | -10.0 | 'SB' | 3 | 'we_bet_small_opp_check' | 'straight_transparent' |
| 54 | 104 | -4.0 | 'SB' | 3 | 'we_checked_opp_check' | 'pair_transparent' |
| 88 | 121 | -21.0 | 'SB' | 3 | 'we_checked_opp_raise' | 'capped' |
| 106 | 108 | -8.0 | 'SB' | 1 | 'we_checked_opp_raise' | 'pair_transparent' |
| 107 | 107 | -7.0 | 'BB' | 3 | 'we_bet_small_opp_unknown' | 'ambiguous' |
| 118 | 104 | -4.0 | 'SB' | 1 | 'we_checked_opp_raise' | 'ambiguous' |
| 124 | 104 | -4.0 | 'SB' | 3 | 'we_checked_opp_check' | 'ambiguous' |
| 127 | 124 | -24.0 | 'BB' | 3 | 'we_checked_opp_raise' | 'ambiguous' |
| 146 | 110 | -10.0 | 'SB' | 2 | 'we_checked_opp_raise' | 'straight_transparent' |
| 160 | 104 | -4.0 | 'SB' | 1 | 'we_checked_opp_raise' | 'pair_transparent' |
| 166 | 102 | -2.0 | 'SB' | 1 | 'we_checked_opp_raise' | 'pair_transparent' |
| 172 | 119 | -19.0 | 'SB' | 2 | 'we_checked_opp_raise' | 'pair_transparent' |

## Showdown losses

| hand | pot | reward | position | our_discard_class | flop_line | flop_texture | turn_texture | river_texture |
|------|-----|--------|----------|-------------------|-----------|--------------|--------------|---------------|
| 2 | 14 | -7.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_raise' | 'monotone' | 'turn_disconnected' | 'river_standard' |
| 4 | 200 | -100.0 | 'SB' | 'straight_transparent' | 'we_bet_small_opp_check' | 'rainbow_disconnected' | 'turn_connected' | 'river_standard' |
| 9 | 28 | -14.0 | 'BB' | 'straight_transparent' | 'we_checked_opp_raise' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 10 | 14 | -7.0 | 'SB' | 'ambiguous' | 'we_bet_small_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 11 | 26 | -13.0 | 'BB' | 'pair_transparent' | 'we_checked_opp_unknown' | 'paired_flop' | 'turn_connected' | 'river_standard' |
| 14 | 34 | -17.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_raise' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 15 | 28 | -14.0 | 'BB' | 'straight_transparent' | 'we_bet_medium_opp_unknown' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 16 | 12 | -6.0 | 'SB' | 'ambiguous' | 'we_bet_small_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 17 | 200 | -100.0 | 'BB' | 'pair_transparent' | 'we_checked_opp_raise' | 'connected_flop' | 'turn_connected' | 'river_standard' |
| 18 | 100 | -50.0 | 'SB' | 'ambiguous' | 'we_bet_small_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 28 | 200 | -100.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_raise' | 'monotone' | 'turn_connected' | 'river_standard' |
| 31 | 48 | -24.0 | 'BB' | 'pair_transparent' | 'we_checked_opp_raise' | 'rainbow_disconnected' | 'turn_connected' | 'river_standard' |
| 32 | 98 | -49.0 | 'SB' | 'ambiguous' | 'we_bet_small_opp_check' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 33 | 14 | -7.0 | 'BB' | 'straight_transparent' | 'we_bet_small_opp_unknown' | 'monotone' | 'turn_disconnected' | 'river_standard' |
| 36 | 14 | -7.0 | 'SB' | 'pair_transparent' | 'we_bet_small_opp_check' | 'connected_flop' | 'turn_connected' | 'river_standard' |
| 37 | 66 | -33.0 | 'BB' | 'ambiguous' | 'we_checked_opp_raise' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 43 | 40 | -20.0 | 'BB' | 'ambiguous' | 'we_checked_opp_raise' | 'connected_flop' | 'turn_connected' | 'river_standard' |
| 47 | 40 | -20.0 | 'BB' | 'flush_transparent' | 'we_checked_opp_raise' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 48 | 200 | -100.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_raise' | 'paired_flop' | 'turn_connected' | 'river_standard' |
| 51 | 42 | -21.0 | 'BB' | 'ambiguous' | 'we_checked_opp_raise' | 'rainbow_disconnected' | 'turn_connected' | 'river_standard' |
| 52 | 60 | -30.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_raise' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 61 | 14 | -7.0 | 'BB' | 'ambiguous' | 'we_checked_opp_unknown' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 69 | 24 | -12.0 | 'BB' | 'pair_transparent' | 'we_checked_opp_raise' | 'rainbow_disconnected' | 'turn_disconnected' | 'river_standard' |
| 76 | 28 | -14.0 | 'SB' | 'ambiguous' | 'we_bet_small_opp_check' | 'paired_flop' | 'turn_connected' | 'river_standard' |
| 84 | 200 | -100.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 86 | 14 | -7.0 | 'SB' | 'ambiguous' | 'we_checked_opp_raise' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 89 | 38 | -19.0 | 'BB' | 'ambiguous' | 'we_bet_small_opp_unknown' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 92 | 8 | -4.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 97 | 8 | -4.0 | 'BB' | 'straight_transparent' | 'we_checked_opp_raise' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 98 | 14 | -7.0 | 'SB' | 'pair_transparent' | 'we_bet_small_opp_check' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 102 | 8 | -4.0 | 'SB' | 'ambiguous' | 'we_bet_small_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 109 | 14 | -7.0 | 'BB' | 'pair_transparent' | 'we_checked_opp_raise' | 'paired_flop' | 'turn_connected' | 'river_standard' |
| 110 | 26 | -13.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 117 | 20 | -10.0 | 'BB' | 'straight_transparent' | 'we_bet_small_opp_unknown' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 122 | 14 | -7.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 129 | 24 | -12.0 | 'BB' | 'ambiguous' | 'we_bet_small_opp_unknown' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 130 | 14 | -7.0 | 'SB' | 'ambiguous' | 'we_bet_small_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 135 | 40 | -20.0 | 'BB' | 'straight_transparent' | 'we_bet_small_opp_unknown' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 138 | 8 | -4.0 | 'SB' | 'ambiguous' | 'we_checked_opp_raise' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 139 | 200 | -100.0 | 'BB' | 'straight_transparent' | 'we_bet_medium_opp_unknown' | 'monotone' | 'flush_completing_turn' | 'river_standard' |
| 142 | 74 | -37.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 145 | 20 | -10.0 | 'BB' | 'ambiguous' | 'we_bet_small_opp_unknown' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 148 | 14 | -7.0 | 'SB' | 'ambiguous' | 'we_bet_small_opp_check' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 150 | 200 | -100.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 151 | 8 | -4.0 | 'BB' | 'straight_transparent' | 'we_bet_small_opp_unknown' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 153 | 200 | -100.0 | 'BB' | 'ambiguous' | 'we_bet_small_opp_unknown' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 155 | 14 | -7.0 | 'BB' | 'ambiguous' | 'we_bet_small_opp_unknown' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 156 | 8 | -4.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 159 | 14 | -7.0 | 'BB' | 'pair_transparent' | 'we_checked_opp_raise' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 163 | 24 | -12.0 | 'BB' | 'pair_transparent' | 'we_checked_opp_raise' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 168 | 8 | -4.0 | 'SB' | 'ambiguous' | 'we_bet_small_opp_check' | 'connected_flop' | 'turn_connected' | 'river_standard' |
| 170 | 8 | -4.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'rainbow_disconnected' | 'turn_connected' | 'river_standard' |
| 175 | 114 | -57.0 | 'BB' | 'pair_transparent' | 'we_checked_opp_raise' | 'paired_flop' | 'turn_connected' | 'river_standard' |
| 177 | 78 | -39.0 | 'BB' | 'pair_transparent' | 'we_bet_small_opp_unknown' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 181 | 40 | -20.0 | 'BB' | 'ambiguous' | 'we_checked_opp_raise' | 'connected_flop' | 'turn_connected' | 'river_standard' |
| 182 | 14 | -7.0 | 'SB' | 'pair_transparent' | 'we_bet_small_opp_check' | 'connected_flop' | 'turn_connected' | 'river_standard' |
| 184 | 34 | -17.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_raise' | 'rainbow_disconnected' | 'turn_connected' | 'river_standard' |
| 187 | 58 | -29.0 | 'BB' | 'ambiguous' | 'we_checked_opp_raise' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 189 | 200 | -100.0 | 'BB' | 'ambiguous' | 'we_checked_opp_raise' | 'monotone' | 'flush_completing_turn' | 'flush_completed_river' |
| 205 | 22 | -11.0 | 'BB' | 'ambiguous' | 'we_bet_small_opp_unknown' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 210 | 14 | -7.0 | 'SB' | 'pair_transparent' | 'we_bet_small_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 211 | 26 | -13.0 | 'BB' | 'straight_transparent' | 'we_checked_opp_raise' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 216 | 26 | -13.0 | 'SB' | 'ambiguous' | 'we_bet_small_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 220 | 74 | -37.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 222 | 30 | -15.0 | 'SB' | 'ambiguous' | 'we_checked_opp_raise' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 230 | 4 | -2.0 | 'SB' | 'flush_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 240 | 4 | -2.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_check' | 'rainbow_disconnected' | 'turn_disconnected' | 'river_standard' |
| 246 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 248 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'rainbow_disconnected' | 'turn_connected' | 'river_standard' |
| 274 | 4 | -2.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 282 | 4 | -2.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_check' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 320 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 334 | 4 | -2.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_check' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 402 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 412 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 460 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 474 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 496 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 498 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 516 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 534 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 536 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 538 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 540 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_connected' | 'river_standard' |
| 556 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 592 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 598 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 642 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 652 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 654 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 662 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 668 | 4 | -2.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_check' | 'rainbow_disconnected' | 'turn_connected' | 'river_standard' |
| 672 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 690 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 714 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 746 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_connected' | 'river_standard' |
| 752 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'rainbow_disconnected' | 'turn_connected' | 'river_standard' |
| 758 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 766 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 770 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 778 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 780 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 782 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 784 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 788 | 4 | -2.0 | 'SB' | 'straight_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 838 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 846 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_connected' | 'river_standard' |
| 850 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 860 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'paired_flop' | 'turn_disconnected' | 'river_standard' |
| 920 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'rainbow_disconnected' | 'turn_connected' | 'river_standard' |
| 964 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_disconnected' | 'river_standard' |
| 976 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 984 | 4 | -2.0 | 'SB' | 'capped' | 'we_checked_opp_check' | 'connected_two_suited' | 'turn_connected' | 'river_standard' |
| 990 | 4 | -2.0 | 'SB' | 'pair_transparent' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |
| 996 | 4 | -2.0 | 'SB' | 'ambiguous' | 'we_checked_opp_check' | 'two_suited_flop' | 'turn_connected' | 'river_standard' |

## By opp_type (at hand end)

| opp_type | count | total_reward |
|----------|-------|--------------|
| 'balanced' | 614 | 109.0 |
| 'loose' | 200 | -569.0 |
| 'tight' | 186 | 239.0 |
