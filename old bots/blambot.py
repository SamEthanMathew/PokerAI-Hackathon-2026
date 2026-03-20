from agents.agent import Agent
from gym_env import PokerEnv


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.opp_raise_count = 0
        self.opp_call_count = 0
        self.hand_type = None  # Track if we have 'made_hand' or 'draw' after discard

        # Track opponent aggression by street (resets each hand)
        self.preflop_aggressor = False
        self.flop_aggressor = False
        self.turn_aggressor = False
        self.last_hand_number = -1  # To detect new hands

        # Track opponent fold tendencies
        self.opp_non_river_bets_faced = 0
        self.opp_non_river_folds = 0
        self.opp_river_bets_faced = 0
        self.opp_river_folds = 0

        # Track our betting actions (to detect when opponent faces our bet)
        self.we_bet_this_street = False
        self.last_street = -1

        # Track VPIP, PFR, and non-river betting stats
        self.total_hands = 0
        self.opp_vpip_count = 0  # Times opponent voluntarily put money in preflop
        self.opp_pfr_count = 0   # Times opponent raised preflop
        self.opp_non_river_streets_seen = 0  # Times opponent saw flop or turn
        self.opp_non_river_bet_count = 0  # Times opponent bet/raised on flop or turn

        # Manual bankroll tracking: track net profit/loss from 0
        # Net profit starts at 0, incremented when we win, decremented when we lose
        self.net_profit_loss = 0  # Net gain/loss throughout the match

    def __name__(self):
        return "PlayerAgent"

    def observe(self, observation, reward, terminated, truncated, info) -> None:
        """
        Called when we receive an observation (including terminal observations).
        Use this to track net profit/loss and fold stats.
        """
        # Update net profit/loss when hand completes (terminated=True)
        if terminated and reward != 0:
            my_bet = observation.get("my_bet", 0)
            opp_bet = observation.get("opp_bet", 0)

            if reward > 0:
                # We won - add opponent's bet to net profit
                self.net_profit_loss += opp_bet
                self.logger.info(f"OBSERVE WON: +{opp_bet} | net={self.net_profit_loss:+d}")
            else:
                # We lost - subtract our bet from net profit
                self.net_profit_loss -= my_bet
                self.logger.info(f"OBSERVE LOST: -{my_bet} | net={self.net_profit_loss:+d}")

        # Update fold stats when hand terminates and opponent folded
        if terminated:
            street = observation.get("street", -1)
            opp_last_action = observation.get("opp_last_action", "")

            self.logger.info(f"OBSERVE TERMINATED: street={street}, opp_last_action='{opp_last_action}', we_bet_this_street={self.we_bet_this_street}")

            if opp_last_action and "fold" in opp_last_action.lower():
                if self.we_bet_this_street:
                    if street == 3:  # River
                        self.opp_river_folds += 1
                        self.logger.info(f"OBSERVE: Opponent folded to river bet | fold_rate={self.get_fold_to_river_bet():.0%}")
                    elif street in [1, 2]:  # Flop or Turn
                        self.opp_non_river_folds += 1
                        self.logger.info(f"OBSERVE: Opponent folded to non-river bet | folds={self.opp_non_river_folds} faced={self.opp_non_river_bets_faced} rate={self.get_fold_to_non_river_bet():.0%}")
                    else:
                        self.logger.info(f"OBSERVE: Opponent folded but street={street} not in [1,2,3]")
                else:
                    self.logger.info(f"OBSERVE: Opponent folded but we_bet_this_street=False")
            else:
                if opp_last_action:
                    self.logger.info(f"OBSERVE: Terminated but no fold detected in action: '{opp_last_action}'")

    def card_rank(self, card: int):
        """Returns rank index 0-8 for ranks 2-9, A"""
        if card == -1:
            return -1
        return card % 9

    def card_suit(self, card: int):
        """Returns suit index 0-2 for suits d, h, s"""
        if card == -1:
            return -1
        return card // 9

    def update_opponent_stats(self, opp_last_action: str):
        """Track opponent's actions to calculate aggression factor"""
        if opp_last_action:
            if "raise" in opp_last_action.lower():
                self.opp_raise_count += 1
            elif "call" in opp_last_action.lower():
                self.opp_call_count += 1

    def get_aggression_factor(self):
        """Calculate opponent's aggression factor (raises/calls)"""
        if self.opp_call_count == 0:
            return float('inf') if self.opp_raise_count > 0 else 0
        return self.opp_raise_count / self.opp_call_count

    def get_fold_to_non_river_bet(self):
        """Calculate opponent's fold frequency on non-river streets when facing a bet"""
        if self.opp_non_river_bets_faced == 0:
            return 0.5  # Default to 50% if no data
        return self.opp_non_river_folds / self.opp_non_river_bets_faced

    def get_fold_to_river_bet(self):
        """Calculate opponent's fold frequency on river when facing a bet"""
        if self.opp_river_bets_faced == 0:
            return 0.5  # Default to 50% if no data
        return self.opp_river_folds / self.opp_river_bets_faced

    def update_fold_stats(self, street, opp_last_action, terminated):
        """Update opponent fold statistics when they face our bet"""
        # Check if opponent folded
        if opp_last_action and "fold" in opp_last_action.lower() and terminated:
            if self.we_bet_this_street:
                if street == 3:  # River
                    self.opp_river_folds += 1
                elif street in [1, 2]:  # Flop or Turn
                    self.opp_non_river_folds += 1

    def record_our_bet(self, street):
        """Record that we bet/raised on this street (only count once per street)"""
        self.we_bet_this_street = True

        # Only increment bets_faced once per street
        if street == 3:  # River
            if not hasattr(self, '_counted_river_bet_this_hand'):
                self._counted_river_bet_this_hand = True
                self.opp_river_bets_faced += 1
                self.logger.info(f"RECORD_BET: River bet recorded, total faced={self.opp_river_bets_faced}")
        elif street in [1, 2]:  # Flop or Turn
            street_name = "flop" if street == 1 else "turn"
            flag_name = f'_counted_{street_name}_bet_faced_this_hand'
            if not hasattr(self, flag_name):
                setattr(self, flag_name, True)
                self.opp_non_river_bets_faced += 1
                self.logger.info(f"RECORD_BET: {street_name.capitalize()} bet recorded, total faced={self.opp_non_river_bets_faced}")

    def get_opponent_type(self):
        """
        Classify opponent based on their fold frequency.

        Returns:
            str: "tight" if fold_to_non_river_bet > 70%, "loose" if < 30%, "balanced" otherwise
        """
        fold_rate = self.get_fold_to_non_river_bet()

        if fold_rate > 0.70:
            return "tight"
        elif fold_rate < 0.30:
            return "loose"
        else:
            return "balanced"

    def get_vpip(self):
        """
        Calculate opponent's VPIP (Voluntary Put In Pot) percentage.
        VPIP is how often they voluntarily put money in preflop (excluding blinds).

        Returns:
            float: VPIP percentage (0.0 to 1.0)
        """
        if self.total_hands == 0:
            return 0.5  # Default to 50% if no data
        return self.opp_vpip_count / self.total_hands

    def get_pfr(self):
        """
        Calculate opponent's PFR (Preflop Raise) percentage.
        PFR is how often they raise preflop.

        Returns:
            float: PFR percentage (0.0 to 1.0)
        """
        if self.total_hands == 0:
            return 0.5  # Default to 50% if no data
        return self.opp_pfr_count / self.total_hands

    def get_non_river_bet_percentage(self):
        """
        Calculate opponent's non-river bet percentage.
        How often they bet or raise on flop or turn when they see those streets.

        Returns:
            float: Non-river bet percentage (0.0 to 1.0)
        """
        if self.opp_non_river_streets_seen == 0:
            return 0.5  # Default to 50% if no data
        return self.opp_non_river_bet_count / self.opp_non_river_streets_seen

    def update_vpip_pfr_stats(self, observation, street):
        """
        Update VPIP, PFR, and non-river bet tracking based on opponent's actions.

        Args:
            observation: Game observation
            street: Current street
        """
        opp_last_action = observation.get("opp_last_action", "")

        # Track preflop actions (street 0) - only count once per hand
        if street == 0 and opp_last_action:
            if "raise" in opp_last_action.lower():
                # Opponent raised preflop - counts for both VPIP and PFR
                if not hasattr(self, '_counted_pfr_this_hand'):
                    self._counted_pfr_this_hand = True
                    self.opp_vpip_count += 1
                    self.opp_pfr_count += 1
            elif "call" in opp_last_action.lower():
                # Opponent called preflop - counts for VPIP only (not PFR)
                # But need to make sure they're not just calling from big blind
                my_blind_position = observation.get("blind_position", -1)
                opp_is_big_blind = (my_blind_position == 0)  # If we're small blind, they're big blind

                # Only count as VPIP if they called an actual bet (not just checking as BB)
                if not hasattr(self, '_counted_vpip_this_hand'):
                    if not opp_is_big_blind or observation.get("my_bet", 0) > observation.get("opp_bet", 0):
                        self._counted_vpip_this_hand = True
                        self.opp_vpip_count += 1

        # Track non-river actions (streets 1 and 2: flop and turn)
        if street in [1, 2]:
            # Check if opponent saw this street (only count once per hand per street)
            street_name = "flop" if street == 1 else "turn"
            flag_name = f'_counted_{street_name}_seen_this_hand'

            if not hasattr(self, flag_name):
                setattr(self, flag_name, True)
                self.opp_non_river_streets_seen += 1

            # Track if they bet or raised on this street
            if opp_last_action and ("bet" in opp_last_action.lower() or "raise" in opp_last_action.lower()):
                bet_flag_name = f'_counted_{street_name}_bet_this_hand'
                if not hasattr(self, bet_flag_name):
                    setattr(self, bet_flag_name, True)
                    self.opp_non_river_bet_count += 1

    def has_completed_flush(self, my_cards, community_cards):
        """Check if we have a completed flush (5+ cards of same suit)"""
        all_cards = my_cards + community_cards
        for suit in range(3):
            suit_cards = [c for c in all_cards if self.card_suit(c) == suit]
            if len(suit_cards) >= 5:
                return True
        return False

    def has_completed_straight(self, my_cards, community_cards):
        """Check if we have a completed straight (5 consecutive ranks, including A2345 and 6789A)"""
        all_cards = my_cards + community_cards
        all_ranks = set(self.card_rank(c) for c in all_cards)

        # Check for A2345 (Ace low): ranks {8, 0, 1, 2, 3}
        if {8, 0, 1, 2, 3}.issubset(all_ranks):
            return True

        # Check for 6789A (Ace high): ranks {4, 5, 6, 7, 8}
        if {4, 5, 6, 7, 8}.issubset(all_ranks):
            return True

        # Check for regular consecutive straights (no Ace wrap)
        sorted_ranks = sorted(all_ranks)
        for i in range(len(sorted_ranks) - 4):
            # Ensure all 5 ranks are consecutive
            if (sorted_ranks[i+1] - sorted_ranks[i] == 1 and
                sorted_ranks[i+2] - sorted_ranks[i+1] == 1 and
                sorted_ranks[i+3] - sorted_ranks[i+2] == 1 and
                sorted_ranks[i+4] - sorted_ranks[i+3] == 1):
                return True

        return False

    def has_flush_draw(self, my_cards, community_cards):
        """Check if we have a flush draw (exactly 4 cards of same suit)"""
        all_cards = my_cards + community_cards
        for suit in range(3):
            suit_cards = [c for c in all_cards if self.card_suit(c) == suit]
            if len(suit_cards) == 4:
                return True
        return False

    def evaluate_flush_draw_equity(self, my_cards, community_cards, my_discarded, opp_discarded):
        """
        Evaluate if we have a high or low equity flush draw.

        High equity flush draw:
        - Four-liner (4 board cards of same suit): 1st or 2nd highest flush
        - Non-four-liner: 1st, 2nd, 3rd, or 4th highest flush

        Returns:
            tuple: (has_flush_draw, is_high_equity, suit, our_highest_rank, unavailable_higher_ranks)
                   or (False, False, -1, -1, []) if no flush draw
        """
        all_my_cards = my_cards + community_cards

        # Check each suit for a flush draw
        for suit in range(3):
            suit_cards_combined = [c for c in all_my_cards if self.card_suit(c) == suit]

            if len(suit_cards_combined) == 4:
                # We have a flush draw in this suit

                # Check if it's a four-liner (4 board cards of same suit)
                board_suit_cards = [c for c in community_cards if self.card_suit(c) == suit]
                is_four_liner = len(board_suit_cards) >= 4

                # Get our cards in this suit
                my_suit_cards = [c for c in my_cards if self.card_suit(c) == suit]
                our_highest_rank = max([self.card_rank(c) for c in my_suit_cards]) if my_suit_cards else -1

                # Collect all known cards in this suit (including discarded)
                all_known_cards = all_my_cards + my_discarded + opp_discarded
                all_known_suit_cards = [c for c in all_known_cards if self.card_suit(c) == suit]
                known_ranks = set(self.card_rank(c) for c in all_known_suit_cards)

                # Find which ranks higher than ours are unavailable (already used/discarded)
                unavailable_higher_ranks = [r for r in known_ranks if r > our_highest_rank]
                num_unavailable_higher = len(unavailable_higher_ranks)

                # Determine if this is high equity
                if is_four_liner:
                    # Four-liner: high equity if we have 1st or 2nd highest (≤1 higher cards unavailable)
                    is_high_equity = num_unavailable_higher <= 1
                else:
                    # Non-four-liner: high equity if we have 1st, 2nd, 3rd, or 4th highest (≤3 higher cards unavailable)
                    is_high_equity = num_unavailable_higher <= 3

                return True, is_high_equity, suit, our_highest_rank, unavailable_higher_ranks

        return False, False, -1, -1, []

    def has_open_ended_straight_draw(self, my_cards, community_cards):
        """Check if we have an open-ended straight draw (4 consecutive ranks, can complete on both ends)"""
        all_cards = my_cards + community_cards
        all_ranks = set(self.card_rank(c) for c in all_cards)

        # Check for open-ended draws including Ace
        # 2345 (can complete with A or 6): {0, 1, 2, 3}
        if {0, 1, 2, 3}.issubset(all_ranks):
            return True

        # 6789 (can complete with 5 or A): {4, 5, 6, 7}
        if {4, 5, 6, 7}.issubset(all_ranks):
            return True

        # Check for regular open-ended draws (4 consecutive, no Ace wrap)
        sorted_ranks = sorted(all_ranks)
        for i in range(len(sorted_ranks) - 3):
            # Ensure all 4 ranks are consecutive (not a gutshot)
            if (sorted_ranks[i+1] - sorted_ranks[i] == 1 and
                sorted_ranks[i+2] - sorted_ranks[i+1] == 1 and
                sorted_ranks[i+3] - sorted_ranks[i+2] == 1):
                low_rank = sorted_ranks[i]
                high_rank = sorted_ranks[i+3]
                # Open-ended means we can extend on both ends (not at extremes)
                if low_rank > 0 and high_rank < 8:
                    return True

        return False

    def has_straight_flush(self, my_cards, community_cards):
        """Check if we have a straight flush (straight AND flush in same suit)"""
        all_cards = my_cards + community_cards

        # Check each suit for both flush and straight
        for suit in range(3):
            suit_cards = [c for c in all_cards if self.card_suit(c) == suit]
            if len(suit_cards) >= 5:
                # We have 5+ cards of this suit, check if they form a straight
                suit_ranks = set(self.card_rank(c) for c in suit_cards)

                # Check for A2345 straight flush
                if {8, 0, 1, 2, 3}.issubset(suit_ranks):
                    return True

                # Check for 6789A straight flush
                if {4, 5, 6, 7, 8}.issubset(suit_ranks):
                    return True

                # Check for regular consecutive straights
                sorted_suit_ranks = sorted(suit_ranks)
                for i in range(len(sorted_suit_ranks) - 4):
                    # Ensure all 5 ranks are consecutive
                    if (sorted_suit_ranks[i+1] - sorted_suit_ranks[i] == 1 and
                        sorted_suit_ranks[i+2] - sorted_suit_ranks[i+1] == 1 and
                        sorted_suit_ranks[i+3] - sorted_suit_ranks[i+2] == 1 and
                        sorted_suit_ranks[i+4] - sorted_suit_ranks[i+3] == 1):
                        return True

        return False

    def has_full_house(self, my_cards, community_cards):
        """Check if we have a full house (3 of one rank + 2 of another)"""
        all_cards = my_cards + community_cards
        all_ranks = [self.card_rank(c) for c in all_cards]

        # Count occurrences of each rank
        rank_counts = {}
        for rank in all_ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        # Check for full house: need at least one set (3+) and another pair (2+)
        has_three = any(c >= 3 for c in rank_counts.values())
        num_pairs = len([c for c in rank_counts.values() if c >= 2])

        # Full house requires a set AND at least 2 different ranks with pairs
        if has_three and num_pairs >= 2:
            return True

        return False

    def has_trips_or_set(self, my_cards, community_cards):
        """Check if we have trips or a set (3 of a kind)"""
        all_cards = my_cards + community_cards
        all_ranks = [self.card_rank(c) for c in all_cards]

        # Count occurrences of each rank
        rank_counts = {}
        for rank in all_ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        # Check if any rank appears 3+ times
        return any(count >= 3 for count in rank_counts.values())

    def has_two_pair_or_better(self, my_cards, community_cards):
        """
        Check if we have two pair or better (two pair, trips, straight, flush, full house, straight flush).

        Returns:
            bool: True if we have two pair or better
        """
        # First check for made hands better than two pair
        if (self.has_straight_flush(my_cards, community_cards) or
            self.has_full_house(my_cards, community_cards) or
            self.has_completed_flush(my_cards, community_cards) or
            self.has_completed_straight(my_cards, community_cards) or
            self.has_trips_or_set(my_cards, community_cards)):
            return True

        # Check for two pair
        all_cards = my_cards + community_cards
        all_ranks = [self.card_rank(c) for c in all_cards]

        # Count occurrences of each rank
        rank_counts = {}
        for rank in all_ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        # Count how many pairs we have
        num_pairs = sum(1 for count in rank_counts.values() if count >= 2)

        # Two pair means at least 2 different ranks with pairs
        return num_pairs >= 2

    def board_has_flush_possible(self, community_cards):
        """Check ONLY community cards for flush possible (3+ cards of same suit)"""
        if len(community_cards) < 3:
            return False

        for suit in range(3):
            suit_cards = [c for c in community_cards if self.card_suit(c) == suit]
            if len(suit_cards) >= 3:
                return True
        return False

    def board_has_straight_possible(self, community_cards):
        """Check ONLY community cards for straight possible (3+ cards that could make a straight)"""
        if len(community_cards) < 3:
            return False

        board_ranks = [self.card_rank(c) for c in community_cards]
        board_rank_set = set(board_ranks)

        # Check for A2345 possibility: need 3+ of {8, 0, 1, 2, 3}
        if len(board_rank_set & {8, 0, 1, 2, 3}) >= 3:
            return True

        # Check for 6789A possibility: need 3+ of {4, 5, 6, 7, 8}
        if len(board_rank_set & {4, 5, 6, 7, 8}) >= 3:
            return True

        # Check for any 3+ consecutive ranks
        sorted_board_ranks = sorted(board_rank_set)
        for i in range(len(sorted_board_ranks) - 2):
            if sorted_board_ranks[i+2] - sorted_board_ranks[i] <= 4:
                # 3 cards within a 5-card window could make a straight
                return True

        return False

    def board_has_double_paired(self, community_cards):
        """Check ONLY community cards for double paired board (2 different pairs on the board)"""
        if len(community_cards) < 4:
            return False

        board_ranks = [self.card_rank(c) for c in community_cards]
        rank_counts = {}
        for rank in board_ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        num_pairs = sum(1 for count in rank_counts.values() if count >= 2)
        return num_pairs >= 2

    def board_has_trips(self, community_cards):
        """Check ONLY community cards for trips (3+ of same rank on the board)"""
        if len(community_cards) < 3:
            return False

        board_ranks = [self.card_rank(c) for c in community_cards]
        rank_counts = {}
        for rank in board_ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        # Check if any rank appears 3+ times
        return any(count >= 3 for count in rank_counts.values())

    def board_has_four_liner_flush(self, community_cards):
        """Check ONLY community cards for four-liner flush (4 cards of same suit on board)"""
        if len(community_cards) < 4:
            return False

        for suit in range(3):
            suit_cards = [c for c in community_cards if self.card_suit(c) == suit]
            if len(suit_cards) >= 4:
                return True
        return False

    def board_has_four_liner_straight(self, community_cards):
        """Check ONLY community cards for four-liner straight (4 cards that make a straight draw on board)"""
        if len(community_cards) < 4:
            return False

        board_ranks = set(self.card_rank(c) for c in community_cards)

        # Check for A2345 four-liner: {8, 0, 1, 2} or {0, 1, 2, 3}
        if {8, 0, 1, 2}.issubset(board_ranks) or {0, 1, 2, 3}.issubset(board_ranks):
            return True

        # Check for 6789A four-liner: {5, 6, 7, 8} or {4, 5, 6, 7}
        if {5, 6, 7, 8}.issubset(board_ranks) or {4, 5, 6, 7}.issubset(board_ranks):
            return True

        # Check for any 4 consecutive ranks
        sorted_board_ranks = sorted(board_ranks)
        for i in range(len(sorted_board_ranks) - 3):
            # Ensure all 4 ranks are consecutive
            if (sorted_board_ranks[i+1] - sorted_board_ranks[i] == 1 and
                sorted_board_ranks[i+2] - sorted_board_ranks[i+1] == 1 and
                sorted_board_ranks[i+3] - sorted_board_ranks[i+2] == 1):
                return True

        return False

    def update_aggression_tracking(self, observation, street):
        """Update opponent aggression tracking based on their last action"""
        opp_last_action = observation.get("opp_last_action", "")

        if opp_last_action and ("raise" in opp_last_action.lower() or "bet" in opp_last_action.lower()):
            if street == 0:
                self.preflop_aggressor = True
            elif street == 1:
                self.flop_aggressor = True
            elif street == 2:
                self.turn_aggressor = True

    def get_aggression_street_count(self):
        """Count how many streets opponent was the aggressor"""
        count = sum([self.preflop_aggressor, self.flop_aggressor, self.turn_aggressor])
        return count

    def identify_made_hand(self, my_cards, community_cards):
        """
        Identify the best made hand we have.

        Returns:
            tuple: (has_sf, has_fh, has_flush, has_straight, has_trips, hand_name)
        """
        has_sf = self.has_straight_flush(my_cards, community_cards)
        has_fh = self.has_full_house(my_cards, community_cards)
        has_flush = self.has_completed_flush(my_cards, community_cards)
        has_straight = self.has_completed_straight(my_cards, community_cards)
        has_trips = self.has_trips_or_set(my_cards, community_cards)

        if has_sf:
            hand_name = "straight flush"
        elif has_fh:
            hand_name = "full house"
        elif has_flush:
            hand_name = "flush"
        elif has_straight:
            hand_name = "straight"
        elif has_trips:
            hand_name = "trips/set"
        else:
            hand_name = "unknown"

        return has_sf, has_fh, has_flush, has_straight, has_trips, hand_name

    def get_board_texture(self, community_cards):
        """
        Get all board texture information.

        Returns:
            dict: Dictionary with board texture flags
        """
        return {
            'double_paired': self.board_has_double_paired(community_cards),
            'flush_possible': self.board_has_flush_possible(community_cards),
            'straight_possible': self.board_has_straight_possible(community_cards),
            'four_flush': self.board_has_four_liner_flush(community_cards),
            'four_straight': self.board_has_four_liner_straight(community_cards)
        }

    def should_bet_river_defensive(self, has_sf, has_fh, has_flush, has_straight, has_trips, board_texture):
        """
        Decide if we should bet on river in defensive mode (opponent aggressed 2+ streets).

        Returns:
            bool: True if we should bet pot
        """
        # Always bet with straight flush, full house, flush
        if has_sf or has_fh or has_flush:
            return True

        # Straight: bet if no four-liner flush and board is safe
        if has_straight:
            if board_texture['four_flush']:
                return False
            if not board_texture['double_paired'] and not board_texture['flush_possible']:
                return True
            return False

        # Trips/set: bet if no four-liners and board is very safe
        if has_trips:
            if board_texture['four_flush'] or board_texture['four_straight']:
                return False
            if (not board_texture['double_paired'] and
                not board_texture['flush_possible'] and
                not board_texture['straight_possible']):
                return True
            return False

        return False

    def should_bet_river_aggressive(self, has_sf, has_fh, has_flush, has_straight, has_trips, board_texture):
        """
        Decide if we should bet on river in aggressive mode (opponent aggressed ≤1 street).

        Returns:
            bool: True if we should bet half pot
        """
        # Always bet with straight flush, full house, flush
        if has_sf or has_fh or has_flush:
            return True

        # Straight: bet if no four-liner flush
        if has_straight:
            return not board_texture['four_flush']

        # Trips/set: bet if no four-liners and board is safe
        if has_trips:
            if board_texture['four_flush'] or board_texture['four_straight']:
                return False
            if not board_texture['double_paired'] and not board_texture['flush_possible']:
                return True
            return False

        return False

    def execute_pot_bet_with_opp_check(self, hand_name, pot_size, amount_to_call, valid_actions, observation, bet_size, street):
        """
        Execute pot-sized betting logic with opponent bet size consideration.

        Args:
            hand_name: Name of the hand for logging
            pot_size: Current pot size
            amount_to_call: Amount we need to call
            valid_actions: Valid actions array
            observation: Game observation
            bet_size: Either pot_size or pot_size // 2
            street: Current street (for tracking fold stats)

        Returns:
            tuple: (action_type, amount, card1, card2)
        """
        if amount_to_call > 0:
            # Opponent has bet - check their bet size relative to initial pot
            initial_pot = pot_size - amount_to_call
            opp_bet_ratio = amount_to_call / initial_pot if initial_pot > 0 else float('inf')

            if opp_bet_ratio > 0.75:
                # Opponent bet >75% of initial pot - just call
                self.logger.info(f"{hand_name}: Opponent bet {amount_to_call} ({opp_bet_ratio:.1%} of pot), calling")
                if valid_actions[self.action_types.CALL.value]:
                    return self.action_types.CALL.value, -1, -1, -1
                elif valid_actions[self.action_types.CHECK.value]:
                    return self.action_types.CHECK.value, -1, -1, -1
                else:
                    return self.action_types.FOLD.value, -1, -1, -1
            else:
                # Opponent bet ≤75% of initial pot - raise to bet_size
                self.logger.info(f"{hand_name}: Opponent bet {amount_to_call} ({opp_bet_ratio:.1%} of pot), raising")
                if valid_actions[self.action_types.RAISE.value]:
                    max_raise = observation["max_raise"]
                    bet_amount = min(bet_size, max_raise)
                    if bet_size < pot_size:  # Half pot bet
                        bet_amount = max(1, bet_amount)
                    # Track that we bet on this street
                    self.record_our_bet(street)
                    return self.action_types.RAISE.value, bet_amount, -1, -1
                elif valid_actions[self.action_types.CALL.value]:
                    return self.action_types.CALL.value, -1, -1, -1
                elif valid_actions[self.action_types.CHECK.value]:
                    return self.action_types.CHECK.value, -1, -1, -1
        else:
            # Opponent hasn't bet - we can bet
            self.logger.info(f"{hand_name}: Betting {bet_size}")
            if valid_actions[self.action_types.RAISE.value]:
                max_raise = observation["max_raise"]
                bet_amount = min(bet_size, max_raise)
                if bet_size < pot_size:  # Half pot bet
                    bet_amount = max(1, bet_amount)
                # Track that we bet on this street
                self.record_our_bet(street)
                return self.action_types.RAISE.value, bet_amount, -1, -1
            elif valid_actions[self.action_types.CALL.value]:
                return self.action_types.CALL.value, -1, -1, -1
            elif valid_actions[self.action_types.CHECK.value]:
                return self.action_types.CHECK.value, -1, -1, -1

        return self.action_types.FOLD.value, -1, -1, -1

    def try_check_fold(self, valid_actions, reason=""):
        """
        Try to check, otherwise fold.

        Returns:
            tuple: (action_type, amount, card1, card2)
        """
        if reason:
            self.logger.info(reason)
        if valid_actions[self.action_types.CHECK.value]:
            return self.action_types.CHECK.value, -1, -1, -1
        else:
            return self.action_types.FOLD.value, -1, -1, -1

    def should_bet_non_river(self, has_sf, has_fh, has_flush, has_straight, has_trips, board_texture):
        """
        Decide if we should bet pot on non-river streets.

        Returns:
            bool: True if we should bet pot
        """
        # Always bet with SF, FH, flush
        if has_sf or has_fh or has_flush:
            return True

        # For straight or trips, check board texture
        if has_straight or has_trips:
            if board_texture['double_paired'] or board_texture['flush_possible']:
                return False
            return True

        return False

    def act(self, observation, reward, terminated, truncated, info):
        """
        Preflop: tries to get in cheap. Postflop: adjusts to opponent's aggression.

        Args (Gym-style step callback):
          observation: Dict with game state for this player (street, my_cards, community_cards,
                       my_bet, opp_bet, valid_actions, my_discarded_cards, opp_discarded_cards, etc.).
          reward: Chip reward from the previous step (e.g. +2 if you won the pot).
          terminated: True if the hand has ended (fold or showdown).
          truncated: True if the episode was cut off (e.g. time limit); usually False per hand.
          info: Extra dict (e.g. hand_number, and at showdown player_0_cards, player_1_cards, community_cards).
        """
        current_hand = info.get('hand_number', -1)
        street = observation["street"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]

        # Reset aggression tracking at the start of a new hand
        if current_hand != self.last_hand_number:
            self.preflop_aggressor = False
            self.flop_aggressor = False
            self.turn_aggressor = False
            self.last_hand_number = current_hand
            self.total_hands = current_hand  # Sync with actual hand number
            # Reset per-hand tracking flags
            for flag in ['_counted_flop_seen_this_hand', '_counted_turn_seen_this_hand',
                        '_counted_flop_bet_this_hand', '_counted_turn_bet_this_hand',
                        '_counted_vpip_this_hand', '_counted_pfr_this_hand',
                        '_counted_river_bet_this_hand', '_counted_flop_bet_faced_this_hand',
                        '_counted_turn_bet_faced_this_hand']:
                if hasattr(self, flag):
                    delattr(self, flag)
            self.logger.info(f"NEW HAND {current_hand}")

        # Reset bet tracking at start of new street
        if street != self.last_street:
            self.we_bet_this_street = False
            self.last_street = street

        self.logger.info(f"Hand {current_hand} street {street}")

        # Update fold stats if opponent folded to our bet on previous street
        self.update_fold_stats(street, observation.get("opp_last_action", ""), terminated)

        # Update aggression tracking based on opponent's last action
        self.update_aggression_tracking(observation, street)

        # Update VPIP, PFR, and flop bet stats
        self.update_vpip_pfr_stats(observation, street)

        # Track opponent's actions for aggression factor
        self.update_opponent_stats(observation.get("opp_last_action", ""))
        af = self.get_aggression_factor()
        vpip = self.get_vpip()
        pfr = self.get_pfr()
        non_river_bet = self.get_non_river_bet_percentage()
        fold_non_river = self.get_fold_to_non_river_bet()
        fold_river = self.get_fold_to_river_bet()

        # Compact HUD stats based on street
        af_str = f"{af:.2f}" if af != float('inf') else "Inf"
        if street == 0:
            # Preflop: show VPIP, PFR
            self.logger.info(f"HUD: VPIP={vpip:.0%} PFR={pfr:.0%} AF={af_str}")
        elif street in [1, 2]:
            # Non-river: show VPIP, AF, non_river_bet%, fold_to_non_river%
            self.logger.info(f"HUD: VPIP={vpip:.0%} AF={af_str} Bet={non_river_bet:.0%} Fold={fold_non_river:.0%}")
        else:
            # River: show AF, river_bet stats (we don't track river bet %, only fold to river bet)
            self.logger.info(f"HUD: AF={af_str} FoldRiver={fold_river:.0%}")

        valid_actions = observation["valid_actions"]
        amount_to_call = opp_bet - my_bet

        # CRITICAL: Check chip lead BEFORE any other actions (including discard)
        # Use net profit/loss to determine if we should fold every hand
        hands_remaining = 1000 - current_hand
        threshold = hands_remaining * 1.5 + 10

        # DEBUG: Log chip lead status on preflop only
        if street == 0:
            self.logger.info(f"CHIP LEAD: Hand {current_hand} | net={self.net_profit_loss:+d} thresh={threshold:.1f} | {self.net_profit_loss}>{threshold:.1f}={self.net_profit_loss > threshold}")

        if self.net_profit_loss > threshold:
            if street == 0:
                self.logger.info(f"!!! BIG LEAD ACTIVE (net={self.net_profit_loss:+d} > {threshold:.1f}): FOLDING ALL HANDS !!!")
            # If it's a discard action, just discard highest 2 cards quickly
            if valid_actions[self.action_types.DISCARD.value]:
                my_cards = [c for c in observation["my_cards"] if c != -1]
                my_cards_indexed = [(i, c) for i, c in enumerate(my_cards)]
                my_cards_indexed.sort(key=lambda x: self.card_rank(x[1]), reverse=True)
                keep_idx0, keep_idx1 = my_cards_indexed[0][0], my_cards_indexed[1][0]
                return self.action_types.DISCARD.value, 0, keep_idx0, keep_idx1
            # Preflop: always try to fold (even if we can check)
            elif street == 0:
                if valid_actions[self.action_types.FOLD.value]:
                    return self.action_types.FOLD.value, -1, -1, -1
                elif valid_actions[self.action_types.CHECK.value]:
                    return self.action_types.CHECK.value, -1, -1, -1
            # Postflop: check/fold
            elif street > 0:
                if valid_actions[self.action_types.CHECK.value]:
                    return self.action_types.CHECK.value, -1, -1, -1
                else:
                    return self.action_types.FOLD.value, -1, -1, -1

        # Handle discard action
        if valid_actions[self.action_types.DISCARD.value]:
            my_cards = [c for c in observation["my_cards"] if c != -1]
            community_cards = [c for c in observation["community_cards"] if c != -1]
            all_cards = my_cards + community_cards

            # Helper function to get highest 2 cards from a list of (index, card) tuples
            def get_best_two_cards(card_list):
                card_list.sort(key=lambda x: self.card_rank(x[1]), reverse=True)
                return card_list[0][0], card_list[1][0]

            # Priority 1: Check for straight flush (highest hand!)
            if self.has_straight_flush(my_cards, community_cards):
                # Find which suit has the straight flush and keep our best 2 cards in that suit
                for suit in range(3):
                    suit_cards_all = [c for c in all_cards if self.card_suit(c) == suit]
                    if len(suit_cards_all) >= 5:
                        suit_ranks = set(self.card_rank(c) for c in suit_cards_all)
                        my_suit_ranks = set(self.card_rank(c) for c in my_cards if self.card_suit(c) == suit)

                        # Find which straight flush this suit has
                        sf_ranks = None
                        if {8, 0, 1, 2, 3}.issubset(suit_ranks):
                            sf_ranks = {8, 0, 1, 2, 3}
                        elif {4, 5, 6, 7, 8}.issubset(suit_ranks):
                            sf_ranks = {4, 5, 6, 7, 8}
                        else:
                            sorted_suit_ranks = sorted(suit_ranks)
                            for i in range(len(sorted_suit_ranks) - 4):
                                # Ensure all 5 ranks are consecutive
                                if (sorted_suit_ranks[i+1] - sorted_suit_ranks[i] == 1 and
                                    sorted_suit_ranks[i+2] - sorted_suit_ranks[i+1] == 1 and
                                    sorted_suit_ranks[i+3] - sorted_suit_ranks[i+2] == 1 and
                                    sorted_suit_ranks[i+4] - sorted_suit_ranks[i+3] == 1):
                                    sf_ranks = set(sorted_suit_ranks[i:i+5])
                                    break

                        if sf_ranks:
                            # Count how many of our cards are in this straight flush
                            num_my_cards_in_sf = len(my_suit_ranks & sf_ranks)

                            # Only valid if we use 1-2 cards from hand (poker rules: max 2 from hand)
                            if 1 <= num_my_cards_in_sf <= 2:
                                my_sf_cards = [(i, c) for i, c in enumerate(my_cards) if self.card_suit(c) == suit and self.card_rank(c) in sf_ranks]
                                if len(my_sf_cards) >= 2:
                                    keep_idx0, keep_idx1 = get_best_two_cards(my_sf_cards)
                                elif len(my_sf_cards) == 1:
                                    # Keep the 1 SF card + best other card
                                    keep_idx0 = my_sf_cards[0][0]
                                    other_cards = [(i, c) for i, c in enumerate(my_cards) if i != keep_idx0]
                                    keep_idx1 = max(other_cards, key=lambda x: self.card_rank(x[1]))[0]
                                else:
                                    # 0 cards in SF (board SF) - keep 2 best cards
                                    keep_idx0, keep_idx1 = get_best_two_cards([(i, c) for i, c in enumerate(my_cards)])

                                self.logger.info(f"Straight flush found (using {num_my_cards_in_sf} hand cards)! Keeping indices {keep_idx0}, {keep_idx1}")
                                self.hand_type = 'made_hand'
                                return self.action_types.DISCARD.value, 0, keep_idx0, keep_idx1
                            else:
                                # Straight flush uses >2 hole cards (invalid!)
                                self.logger.warning(f"Have straight flush but using {num_my_cards_in_sf} hole cards (>2)! Skipping...")

            # Priority 2: Check for full house
            if self.has_full_house(my_cards, community_cards):
                # Keep the two highest cards that contribute to the full house
                all_ranks = [self.card_rank(c) for c in all_cards]
                my_ranks = [self.card_rank(c) for c in my_cards]

                rank_counts = {}
                for rank in all_ranks:
                    rank_counts[rank] = rank_counts.get(rank, 0) + 1

                # Find the set (3+) ranks and pair (2+) ranks
                set_ranks = [r for r, c in rank_counts.items() if c >= 3]
                pair_ranks = [r for r, c in rank_counts.items() if c >= 2]

                if set_ranks and len(pair_ranks) >= 2:
                    # Full house is made from the highest set + highest other pair
                    best_set_rank = max(set_ranks)
                    # Find the second rank for the pair (could be another set or just a pair)
                    pair_ranks_for_fh = [r for r in pair_ranks if r != best_set_rank]
                    if pair_ranks_for_fh:
                        best_pair_rank = max(pair_ranks_for_fh)

                        # Count how many of our cards are used in this full house
                        num_my_cards_in_fh = my_ranks.count(best_set_rank) + my_ranks.count(best_pair_rank)

                        # Only valid if we use 1-2 cards from hand (poker rules: max 2 from hand)
                        if 1 <= num_my_cards_in_fh <= 2:
                            # Keep cards from the full house (prefer set, then pair)
                            my_fh_cards = [(i, c) for i, c in enumerate(my_cards) if self.card_rank(c) in [best_set_rank, best_pair_rank]]
                            if len(my_fh_cards) >= 2:
                                keep_idx0, keep_idx1 = get_best_two_cards(my_fh_cards)
                            elif len(my_fh_cards) == 1:
                                keep_idx0 = my_fh_cards[0][0]
                                other_cards = [(i, c) for i, c in enumerate(my_cards) if i != keep_idx0]
                                keep_idx1 = max(other_cards, key=lambda x: self.card_rank(x[1]))[0]
                            else:
                                # 0 cards in FH (board FH) - keep 2 best cards
                                keep_idx0, keep_idx1 = get_best_two_cards([(i, c) for i, c in enumerate(my_cards)])

                            self.logger.info(f"Full house found (using {num_my_cards_in_fh} hand cards)! Keeping indices {keep_idx0}, {keep_idx1}")
                            self.hand_type = 'made_hand'
                            return self.action_types.DISCARD.value, 0, keep_idx0, keep_idx1
                        else:
                            # Full house uses >2 hole cards (invalid!) or 0 cards
                            self.logger.warning(f"Have full house but using {num_my_cards_in_fh} hole cards! Skipping...")

            # Priority 3: Check for flush (5+ cards of same suit)
            for suit in range(3):
                suit_cards_all = [c for c in all_cards if self.card_suit(c) == suit]
                if len(suit_cards_all) >= 5:
                    # Count how many board cards and hole cards are in this suit
                    board_suit_cards = [c for c in community_cards if self.card_suit(c) == suit]
                    my_flush_cards = [(i, c) for i, c in enumerate(my_cards) if self.card_suit(c) == suit]
                    num_my_cards_in_flush = len(my_flush_cards)

                    # To make a flush we need 5 cards of same suit
                    # If board has X cards, we need (5-X) from hand
                    # Only valid if we use 1-2 cards from hand (poker rules: max 2 from hand)
                    num_board_in_suit = len(board_suit_cards)
                    cards_needed_from_hand = max(0, 5 - num_board_in_suit)

                    if cards_needed_from_hand <= 2:
                        if num_my_cards_in_flush >= 2:
                            keep_idx0, keep_idx1 = get_best_two_cards(my_flush_cards)
                        elif num_my_cards_in_flush == 1:
                            keep_idx0 = my_flush_cards[0][0]
                            other_cards = [(i, c) for i, c in enumerate(my_cards) if i != keep_idx0]
                            keep_idx1 = max(other_cards, key=lambda x: self.card_rank(x[1]))[0]
                        else:
                            # 0 cards in flush (board flush) - keep 2 best cards
                            keep_idx0, keep_idx1 = get_best_two_cards([(i, c) for i, c in enumerate(my_cards)])

                        self.logger.info(f"Flush found (board:{num_board_in_suit}, need {cards_needed_from_hand} from hand)! Keeping indices {keep_idx0}, {keep_idx1}")
                        self.hand_type = 'made_hand'
                        return self.action_types.DISCARD.value, 0, keep_idx0, keep_idx1
                    else:
                        # Flush needs >2 hole cards (invalid!)
                        self.logger.warning(f"Have flush but needs {cards_needed_from_hand} hole cards (>2)! Skipping...")

            # Priority 4: Check for straight (5 consecutive ranks)
            if self.has_completed_straight(my_cards, community_cards):
                all_ranks = sorted(set(self.card_rank(c) for c in all_cards))
                my_ranks = set(self.card_rank(c) for c in my_cards)

                # Find ALL possible straights, then keep the highest one where we have ≥2 hole cards
                possible_straights = []

                # Check 6789A
                if {4, 5, 6, 7, 8}.issubset(all_ranks):
                    num_my_cards = len(my_ranks & {4, 5, 6, 7, 8})
                    possible_straights.append(({4, 5, 6, 7, 8}, num_my_cards, 8))  # (ranks, count, high_rank)

                # Check for regular consecutive straights
                for i in range(len(all_ranks) - 4):
                    if (all_ranks[i+1] - all_ranks[i] == 1 and
                        all_ranks[i+2] - all_ranks[i+1] == 1 and
                        all_ranks[i+3] - all_ranks[i+2] == 1 and
                        all_ranks[i+4] - all_ranks[i+3] == 1):
                        straight_ranks = set(all_ranks[i:i+5])
                        num_my_cards = len(my_ranks & straight_ranks)
                        high_rank = all_ranks[i+4]
                        possible_straights.append((straight_ranks, num_my_cards, high_rank))

                # Check A2345
                if {8, 0, 1, 2, 3}.issubset(all_ranks):
                    num_my_cards = len(my_ranks & {8, 0, 1, 2, 3})
                    possible_straights.append(({8, 0, 1, 2, 3}, num_my_cards, 3))  # High rank is 3 (5) for A2345

                # Filter to straights where we use 1-2 hole cards (poker rules: max 2 from hand)
                valid_straights = [(ranks, count, high) for ranks, count, high in possible_straights if 1 <= count <= 2]

                if valid_straights:
                    # Sort by high rank (descending) to get the highest straight
                    valid_straights.sort(key=lambda x: x[2], reverse=True)
                    straight_ranks, count, _ = valid_straights[0]

                    my_straight_cards = [(i, c) for i, c in enumerate(my_cards) if self.card_rank(c) in straight_ranks]

                    if len(my_straight_cards) >= 2:
                        keep_idx0, keep_idx1 = get_best_two_cards(my_straight_cards)
                    else:
                        # Only 1 card in straight, keep it + highest other card
                        keep_idx0 = my_straight_cards[0][0]
                        other_cards = [(i, c) for i, c in enumerate(my_cards) if i != keep_idx0]
                        keep_idx1 = max(other_cards, key=lambda x: self.card_rank(x[1]))[0]

                    self.logger.info(f"Straight found (ranks {straight_ranks}, using {count} hand cards)! Keeping indices {keep_idx0}, {keep_idx1}")
                    self.hand_type = 'made_hand'
                    return self.action_types.DISCARD.value, 0, keep_idx0, keep_idx1
                else:
                    # We have a straight but using > 2 hole cards (invalid!)
                    self.logger.warning("Have straight but using >2 hole cards! Falling through...")

            # Priority 5: Check for trips/set
            if self.has_trips_or_set(my_cards, community_cards):
                all_ranks = [self.card_rank(c) for c in all_cards]
                my_ranks = [self.card_rank(c) for c in my_cards]
                rank_counts = {}
                for rank in all_ranks:
                    rank_counts[rank] = rank_counts.get(rank, 0) + 1

                # Find the trips rank(s)
                trips_ranks = [r for r, c in rank_counts.items() if c >= 3]
                self.logger.info(f"DISCARD Priority 5: Trips detected. trips_ranks={trips_ranks}")
                if trips_ranks:
                    # Keep cards from the highest trips
                    best_trips_rank = max(trips_ranks)
                    my_trips_cards = [(i, c) for i, c in enumerate(my_cards) if self.card_rank(c) == best_trips_rank]
                    num_my_cards_in_trips = len(my_trips_cards)

                    self.logger.info(f"  best_trips_rank={best_trips_rank}, my_trips_cards count={num_my_cards_in_trips}, my_cards={[(i, self.card_rank(c)) for i, c in enumerate(my_cards)]}")

                    # Only valid if we use 1-2 cards from hand (poker rules: max 2 from hand)
                    if 1 <= num_my_cards_in_trips <= 2:
                        if num_my_cards_in_trips >= 2:
                            keep_idx0, keep_idx1 = get_best_two_cards(my_trips_cards)
                            self.logger.info(f"  KEEPING: 2 trips cards at indices {keep_idx0}, {keep_idx1}")
                        else:  # num_my_cards_in_trips == 1
                            # We have one card of trips, keep it + highest other card
                            keep_idx0 = my_trips_cards[0][0]
                            other_cards = [(i, c) for i, c in enumerate(my_cards) if i != keep_idx0]
                            if other_cards:
                                keep_idx1 = max(other_cards, key=lambda x: self.card_rank(x[1]))[0]
                                self.logger.info(f"  KEEPING: 1 trips card at idx {keep_idx0} + highest other at idx {keep_idx1}")

                        self.hand_type = 'made_hand'
                        return self.action_types.DISCARD.value, 0, keep_idx0, keep_idx1
                    elif num_my_cards_in_trips == 0:
                        # All trips cards are on board, none in hand
                        self.logger.info(f"  FALLING THROUGH: All trips cards on board, none in hand")
                    else:
                        # Trips uses >2 hole cards (invalid!)
                        self.logger.warning(f"  Have trips but using {num_my_cards_in_trips} hole cards (>2)! Skipping...")

            # Priority 6: Check for flush draw (4 cards of same suit)
            my_discarded = [c for c in observation.get("my_discarded_cards", []) if c != -1]
            opp_discarded = [c for c in observation.get("opp_discarded_cards", []) if c != -1]

            has_fd, is_high_equity, fd_suit, our_rank, unavail_ranks = self.evaluate_flush_draw_equity(
                my_cards, community_cards, my_discarded, opp_discarded
            )

            if has_fd:
                self.logger.info(f"DISCARD Priority 6: Flush draw detected (suit={fd_suit}, high_equity={is_high_equity})")

                # Check how many board cards are in the flush draw suit
                board_suit_cards = [c for c in community_cards if self.card_suit(c) == fd_suit]
                my_draw_cards = [(i, c) for i, c in enumerate(my_cards) if self.card_suit(c) == fd_suit]
                num_board_in_suit = len(board_suit_cards)
                num_my_cards_in_draw = len(my_draw_cards)

                # For a 4-card flush draw, we need 4 total: (4-board_count) from hand
                cards_needed_from_hand = 4 - num_board_in_suit

                self.logger.info(f"  my_draw_cards count={num_my_cards_in_draw}, board_count={num_board_in_suit}, need {cards_needed_from_hand} from hand")

                # Only valid if we use 1-2 cards from hand (poker rules: max 2 from hand)
                if cards_needed_from_hand <= 2:
                    if num_my_cards_in_draw >= 2:
                        keep_idx0, keep_idx1 = get_best_two_cards(my_draw_cards)
                        equity_type = "HIGH EQUITY" if is_high_equity else "LOW EQUITY"
                        self.logger.info(f"  KEEPING: {equity_type} flush draw at indices {keep_idx0}, {keep_idx1}")
                        self.hand_type = 'draw'
                        return self.action_types.DISCARD.value, 0, keep_idx0, keep_idx1
                    else:
                        self.logger.info(f"  FALLING THROUGH: Flush draw detected but < 2 cards in hand")
                else:
                    # Flush draw needs >2 hole cards (invalid!)
                    self.logger.warning(f"  Have flush draw but needs {cards_needed_from_hand} hole cards (>2)! Skipping...")

            # Priority 7: Check for open-ended straight draw
            if self.has_open_ended_straight_draw(my_cards, community_cards):
                all_ranks = sorted(set(self.card_rank(c) for c in all_cards))
                my_ranks = set(self.card_rank(c) for c in my_cards)

                self.logger.info(f"DISCARD Priority 7: OESD detected. all_ranks={all_ranks}, my_ranks={my_ranks}")

                # Find ALL possible OESDs, then keep the highest one where we have ≥2 hole cards
                possible_draws = []

                # Check 6789 (can make 56789 or 6789A)
                if {4, 5, 6, 7}.issubset(all_ranks):
                    num_my_cards = len(my_ranks & {4, 5, 6, 7})
                    possible_draws.append(({4, 5, 6, 7}, num_my_cards, 7))  # (ranks, count, high_rank)

                # Check for regular OESDs
                for i in range(len(all_ranks) - 3):
                    if (all_ranks[i+1] - all_ranks[i] == 1 and
                        all_ranks[i+2] - all_ranks[i+1] == 1 and
                        all_ranks[i+3] - all_ranks[i+2] == 1):
                        low_rank, high_rank = all_ranks[i], all_ranks[i+3]
                        if low_rank > 0 and high_rank < 8:
                            draw_ranks = set(all_ranks[i:i+4])
                            num_my_cards = len(my_ranks & draw_ranks)
                            possible_draws.append((draw_ranks, num_my_cards, high_rank))

                # Check 2345 (can make A2345 or 23456)
                if {0, 1, 2, 3}.issubset(all_ranks):
                    num_my_cards = len(my_ranks & {0, 1, 2, 3})
                    possible_draws.append(({0, 1, 2, 3}, num_my_cards, 3))  # High rank is 3 (5)

                # Filter to draws where we use 1-2 hole cards (poker rules: max 2 from hand)
                valid_draws = [(ranks, count, high) for ranks, count, high in possible_draws if 1 <= count <= 2]
                self.logger.info(f"  possible_draws={possible_draws}, valid_draws count={len(valid_draws)}")

                if valid_draws:
                    # Sort by high rank (descending) to get the highest draw
                    valid_draws.sort(key=lambda x: x[2], reverse=True)
                    draw_ranks, count, _ = valid_draws[0]

                    my_draw_cards = [(i, c) for i, c in enumerate(my_cards) if self.card_rank(c) in draw_ranks]

                    if len(my_draw_cards) >= 2:
                        keep_idx0, keep_idx1 = get_best_two_cards(my_draw_cards)
                    else:
                        # Only 1 card in draw, keep it + highest other card
                        keep_idx0 = my_draw_cards[0][0]
                        other_cards = [(i, c) for i, c in enumerate(my_cards) if i != keep_idx0]
                        keep_idx1 = max(other_cards, key=lambda x: self.card_rank(x[1]))[0]

                    self.logger.info(f"  KEEPING: OESD cards (ranks {draw_ranks}, using {count} hand cards) at indices {keep_idx0}, {keep_idx1}")
                    self.hand_type = 'draw'
                    return self.action_types.DISCARD.value, 0, keep_idx0, keep_idx1
                else:
                    # We have an OESD but using > 2 hole cards (invalid!)
                    self.logger.info(f"  FALLING THROUGH: OESD detected but using >2 hole cards!")

            # Priority 8: Default - keep highest 2 cards
            my_cards_indexed = [(i, c) for i, c in enumerate(my_cards)]
            my_cards_with_ranks = [(i, c, self.card_rank(c)) for i, c in enumerate(my_cards)]
            self.logger.info(f"DISCARD Priority 8: No made hand or draw. my_cards (idx, card, rank)={my_cards_with_ranks}")
            keep_idx0, keep_idx1 = get_best_two_cards(my_cards_indexed)
            self.logger.info(f"  KEEPING: Highest 2 cards at indices {keep_idx0}, {keep_idx1} (cards: rank {self.card_rank(my_cards[keep_idx0])}, rank {self.card_rank(my_cards[keep_idx1])})")
            self.hand_type = None
            return self.action_types.DISCARD.value, 0, keep_idx0, keep_idx1

        # Preflop strategy (street 0): try to get in cheap
        if street == 0:
            self.logger.info(f"Preflop - Amount to call: {amount_to_call}")

            # If we can check, check
            if valid_actions[self.action_types.CHECK.value]:
                self.logger.info("Checking preflop")
                return self.action_types.CHECK.value, -1, -1, -1

            # IMPORTANT: Check for high PFR opponents FIRST before other logic
            # Against very aggressive opponents (high PFR), don't fold
            if pfr > 0.50:
                # Call if amount to call is under 20
                if amount_to_call <= 20 and valid_actions[self.action_types.CALL.value]:
                    self.logger.info(f"Calling {amount_to_call} preflop against high PFR opponent (PFR={pfr:.1%})")
                    return self.action_types.CALL.value, -1, -1, -1
                # Against extremely aggressive opponents (PFR > 80%), call anything
                elif pfr > 0.80:
                    if valid_actions[self.action_types.CALL.value]:
                        self.logger.info(f"Calling against very aggressive opponent preflop (PFR={pfr:.1%})")
                        return self.action_types.CALL.value, -1, -1, -1
                    elif valid_actions[self.action_types.CHECK.value]:
                        self.logger.info("Checking against very aggressive opponent preflop")
                        return self.action_types.CHECK.value, -1, -1, -1

            # Against very passive opponents (low VPIP), we can raise more preflop
            if vpip < 0.40:
                import random
                do_raise = random.random() < 0.5
                if do_raise and valid_actions[self.action_types.RAISE.value]:
                    raise_amount = min(10, observation["max_raise"])  # Raise up to 10 preflop against passive
                    self.logger.info(f"Raising {raise_amount} preflop against passive opponent (VPIP={vpip:.1%})")
                    self.record_our_bet(street)
                    return self.action_types.RAISE.value, raise_amount, -1, -1
                else:
                    self.logger.info("Not raising preflop against passive opponent")
                    if valid_actions[self.action_types.CALL.value]:
                        return self.action_types.CALL.value, -1, -1, -1
                    elif valid_actions[self.action_types.CHECK.value]:
                        return self.action_types.CHECK.value, -1, -1, -1

            # Call if amount to call is under 10
            if amount_to_call <= 10 and valid_actions[self.action_types.CALL.value]:
                self.logger.info(f"Calling {amount_to_call} preflop (under 10)")
                return self.action_types.CALL.value, -1, -1, -1

            # Otherwise fold
            self.logger.info("Folding preflop (bet too high)")
            return self.try_check_fold(valid_actions, "Preflop: Folding/checking as fallback")

        # Postflop strategy (streets 1, 2, 3): adjust to opponent's aggression
        if street > 0:
            street_name = {1: "FLOP", 2: "TURN", 3: "RIVER"}.get(street, f"St{street}")
            self.logger.info(f"{street_name} - To call: {amount_to_call}, Pot: {my_bet + opp_bet}, AF: {af}")

            my_cards = [c for c in observation["my_cards"] if c != -1]
            community_cards = [c for c in observation["community_cards"] if c != -1]
            pot_size = my_bet + opp_bet  # Calculate pot size

            # RE-EVALUATE actual hand strength (don't rely on hand_type from discard!)
            # Check for made hands in priority order
            actual_hand_type = None
            if self.has_straight_flush(my_cards, community_cards):
                actual_hand_type = 'made_hand'
                self.logger.info("Current hand: STRAIGHT FLUSH")
            elif self.has_full_house(my_cards, community_cards):
                actual_hand_type = 'made_hand'
                self.logger.info("Current hand: FULL HOUSE")
            elif self.has_completed_flush(my_cards, community_cards):
                actual_hand_type = 'made_hand'
                self.logger.info("Current hand: FLUSH")
            elif self.has_completed_straight(my_cards, community_cards):
                actual_hand_type = 'made_hand'
                self.logger.info("Current hand: STRAIGHT")
            elif self.has_trips_or_set(my_cards, community_cards):
                actual_hand_type = 'made_hand'
                self.logger.info("Current hand: TRIPS/SET")
            # Check for draws ONLY on non-river streets (on river, draws either completed or are nothing)
            elif street < 3:
                # Get discarded cards for flush draw evaluation
                my_discarded = [c for c in observation.get("my_discarded_cards", []) if c != -1]
                opp_discarded = [c for c in observation.get("opp_discarded_cards", []) if c != -1]

                has_fd, is_high_equity, fd_suit, our_rank, unavail_ranks = self.evaluate_flush_draw_equity(
                    my_cards, community_cards, my_discarded, opp_discarded
                )

                if has_fd:
                    actual_hand_type = 'draw'
                    equity_type = "HIGH EQUITY" if is_high_equity else "LOW EQUITY"
                    self.logger.info(f"Current hand: {equity_type} FLUSH DRAW (suit={fd_suit}, our_rank={our_rank}, unavailable_higher={unavail_ranks})")
                elif self.has_open_ended_straight_draw(my_cards, community_cards):
                    actual_hand_type = 'draw'
                    self.logger.info("Current hand: STRAIGHT DRAW")
                else:
                    actual_hand_type = None
                    self.logger.info("Current hand: NOTHING")
            else:
                # On river, if no made hand, we have nothing
                actual_hand_type = None
                self.logger.info("Current hand: NOTHING (river)")

            # EXPLOIT: Fold to board trips unless we have full house
            # This is a dangerous board where opponent likely has us beat
            if self.board_has_trips(community_cards):
                has_fh = self.has_full_house(my_cards, community_cards)
                if not has_fh:
                    self.logger.info("BOARD TRIPS DETECTED: Folding without full house")
                    return self.try_check_fold(valid_actions, "Board has trips and we don't have full house")
                else:
                    self.logger.info("BOARD TRIPS DETECTED: Continuing with full house")

            # EXPLOIT: Fold to tight opponent's overbet unless we have ultra-premium hands
            # Tight opponent = fold to river bet >= 70% AND fold to non-river bet >= 70%
            # Overbet = amount to call > original pot size before their bet
            if amount_to_call > 0:
                initial_pot = pot_size - amount_to_call
                is_overbet = amount_to_call > initial_pot
                fold_non_river = self.get_fold_to_non_river_bet()
                fold_river = self.get_fold_to_river_bet()
                is_tight_opponent = fold_river >= 0.70 and fold_non_river >= 0.70

                if is_tight_opponent and is_overbet:
                    self.logger.info(f"TIGHT OPPONENT OVERBET DETECTED: Opponent fold rates (non_river = {fold_non_river:.1%}, river={fold_river:.1%}), Overbet size={amount_to_call} vs initial pot={initial_pot}")

                    # Check if we have exception hands: SF, FH, or high equity flush
                    has_sf = self.has_straight_flush(my_cards, community_cards)
                    has_fh = self.has_full_house(my_cards, community_cards)

                    # Check for high equity flush
                    my_discarded = [c for c in observation.get("my_discarded_cards", []) if c != -1]
                    opp_discarded = [c for c in observation.get("opp_discarded_cards", []) if c != -1]
                    has_completed_flush = self.has_completed_flush(my_cards, community_cards)

                    has_high_equity_flush = False
                    if has_completed_flush:
                        # For completed flush, evaluate it as if it were a "5-card flush draw"
                        # by checking our highest flush card against all known cards
                        for suit in range(3):
                            suit_cards_all = [c for c in (my_cards + community_cards) if self.card_suit(c) == suit]
                            if len(suit_cards_all) >= 5:
                                # This is our flush suit
                                my_suit_cards = [c for c in my_cards if self.card_suit(c) == suit]
                                if my_suit_cards:
                                    our_highest_rank = max([self.card_rank(c) for c in my_suit_cards])

                                    # Check known higher ranks
                                    all_known_cards = my_cards + community_cards + my_discarded + opp_discarded
                                    all_known_suit_cards = [c for c in all_known_cards if self.card_suit(c) == suit]
                                    known_ranks = set(self.card_rank(c) for c in all_known_suit_cards)
                                    unavailable_higher_ranks = [r for r in known_ranks if r > our_highest_rank]

                                    # High equity flush: ≤3 higher ranks unavailable
                                    has_high_equity_flush = len(unavailable_higher_ranks) <= 3
                                    self.logger.info(f"Completed flush equity check: our_rank={our_highest_rank}, unavailable_higher={unavailable_higher_ranks}, is_high_equity={has_high_equity_flush}")
                                break

                    if has_sf:
                        self.logger.info("KEEPING: Have straight flush vs tight opponent overbet")
                    elif has_fh:
                        self.logger.info("KEEPING: Have full house vs tight opponent overbet")
                    elif has_high_equity_flush:
                        self.logger.info("KEEPING: Have high equity flush vs tight opponent overbet")
                    else:
                        # Fold everything else
                        self.logger.info("FOLDING: Tight opponent overbet - don't have SF/FH/high-equity-flush")
                        return self.action_types.FOLD.value, -1, -1, -1

            # RIVER-SPECIFIC LOGIC (street 3)
            if street == 3:
                import random

                # Get board texture and opponent aggression count
                board_texture = self.get_board_texture(community_cards)
                opp_agg_count = self.get_aggression_street_count()
                self.logger.info(f"RIVER: Opponent was aggressor on {opp_agg_count} streets")

                # STRATEGY 1: Opponent was aggressor on 2+ streets (DEFENSIVE)
                if opp_agg_count >= 2:
                    self.logger.info("RIVER Strategy: DEFENSIVE (opponent aggressed 2+ streets)")

                    if actual_hand_type == 'made_hand':
                        # Identify hand and decide whether to bet
                        has_sf, has_fh, has_flush, has_straight, has_trips, hand_name = self.identify_made_hand(my_cards, community_cards)
                        should_bet_pot = self.should_bet_river_defensive(has_sf, has_fh, has_flush, has_straight, has_trips, board_texture)

                        if should_bet_pot:
                            return self.execute_pot_bet_with_opp_check(
                                f"RIVER DEFENSIVE: {hand_name}",
                                pot_size, amount_to_call, valid_actions, observation, pot_size, street
                            )
                        else:
                            # Don't bet, just check/fold
                            return self.try_check_fold(valid_actions, f"RIVER DEFENSIVE: Checking/folding with {hand_name}")

                    # Always give up with missed draws or nothing when opponent was very aggressive
                    else:
                        return self.try_check_fold(valid_actions, "RIVER DEFENSIVE: Giving up with missed draw/nothing")

                # STRATEGY 2: Opponent was aggressor on 1 or fewer streets (AGGRESSIVE)
                else:
                    self.logger.info("RIVER Strategy: AGGRESSIVE (opponent aggressed ≤1 street)")

                    if actual_hand_type == 'made_hand':
                        # Identify hand and decide bet size
                        has_sf, has_fh, has_flush, has_straight, has_trips, hand_name = self.identify_made_hand(my_cards, community_cards)
                        should_bet_half_pot = self.should_bet_river_aggressive(has_sf, has_fh, has_flush, has_straight, has_trips, board_texture)

                        if should_bet_half_pot:
                            return self.execute_pot_bet_with_opp_check(
                                f"RIVER AGGRESSIVE: {hand_name}",
                                pot_size, amount_to_call, valid_actions, observation, pot_size // 2, street
                            )
                        else:
                            # Don't bet, just check/call
                            return self.try_check_fold(valid_actions, f"RIVER AGGRESSIVE: Checking/calling with {hand_name}")

                    # Bluff 50% with missed draws, give up 50%
                    elif actual_hand_type == 'draw':
                        bluff = random.random() < 0.5  # 50% chance to bluff

                        if bluff:
                            return self.execute_pot_bet_with_opp_check(
                                "RIVER AGGRESSIVE: Bluff",
                                pot_size, amount_to_call, valid_actions, observation, pot_size, street
                            )
                        else:
                            return self.try_check_fold(valid_actions, "RIVER AGGRESSIVE: Giving up with missed draw")

                    # Give up with nothing
                    else:
                        return self.try_check_fold(valid_actions, "RIVER AGGRESSIVE: Giving up with nothing")

            # NON-RIVER POSTFLOP LOGIC (streets 1, 2)
            # Check pot odds: if getting 4:1 or better (call <= 25% of pot), call with two pair or better
            if street in [1, 2] and amount_to_call > 0 and amount_to_call <= 0.25 * pot_size:
                if self.has_two_pair_or_better(my_cards, community_cards):
                    self.logger.info(f">>> Good pot odds: {amount_to_call}/{pot_size} ({amount_to_call/pot_size:.0%}) with 2pair+")
                    if valid_actions[self.action_types.CALL.value]:
                        self.logger.info(f">>> ACTION: CALL (pot odds)")
                        return self.action_types.CALL.value, -1, -1, -1
                    elif valid_actions[self.action_types.CHECK.value]:
                        self.logger.info(f">>> ACTION: CHECK (pot odds)")
                        return self.action_types.CHECK.value, -1, -1, -1

            # Against very aggressive non-river opponents, call with two pair or better
            if street in [1, 2] and non_river_bet > 0.80 and amount_to_call > 0:
                if self.has_two_pair_or_better(my_cards, community_cards):
                    self.logger.info(f">>> Aggressive opponent ({non_river_bet:.0%} bet freq) with 2pair+")
                    if valid_actions[self.action_types.CALL.value]:
                        self.logger.info(f">>> ACTION: CALL (vs aggressive)")
                        return self.action_types.CALL.value, -1, -1, -1
                    elif valid_actions[self.action_types.CHECK.value]:
                        self.logger.info(f">>> ACTION: CHECK (vs aggressive)")
                        return self.action_types.CHECK.value, -1, -1, -1

            if street in [1, 2] and actual_hand_type == 'made_hand':
                # Get board texture and identify hand
                board_texture = self.get_board_texture(community_cards)
                has_sf, has_fh, has_flush, has_straight, has_trips, hand_name = self.identify_made_hand(my_cards, community_cards)

                # Decide if we should bet based on board texture
                should_bet_pot = self.should_bet_non_river(has_sf, has_fh, has_flush, has_straight, has_trips, board_texture)

                if should_bet_pot:
                    return self.execute_pot_bet_with_opp_check(
                        f"Made hand ({hand_name})",
                        pot_size, amount_to_call, valid_actions, observation, pot_size, street
                    )
                else:
                    # Scary board - check/give up
                    return self.try_check_fold(valid_actions, f"{hand_name} on scary board - checking/giving up")

            # If we have a DRAW (flush draw or straight draw), call with good pot odds
            if street in [1, 2] and actual_hand_type == 'draw':
                # Identify which draw we have for logging
                my_discarded = [c for c in observation.get("my_discarded_cards", []) if c != -1]
                opp_discarded = [c for c in observation.get("opp_discarded_cards", []) if c != -1]

                has_fd, is_high_equity, fd_suit, our_rank, unavail_ranks = self.evaluate_flush_draw_equity(
                    my_cards, community_cards, my_discarded, opp_discarded
                )

                if has_fd:
                    equity_type = "HIGH EQUITY" if is_high_equity else "LOW EQUITY"
                    draw_name = f"{equity_type} flush draw"
                elif self.has_open_ended_straight_draw(my_cards, community_cards):
                    draw_name = "straight draw"
                else:
                    draw_name = "draw"

                # Check first if we can, otherwise call with good pot odds
                if valid_actions[self.action_types.CHECK.value]:
                    self.logger.info(f">>> ACTION: CHECK (with {draw_name})")
                    return self.action_types.CHECK.value, -1, -1, -1
                elif amount_to_call < 0.5 * pot_size and valid_actions[self.action_types.CALL.value]:
                    self.logger.info(f">>> ACTION: CALL {amount_to_call} with {draw_name} (good odds)")
                    return self.action_types.CALL.value, -1, -1, -1
                else:
                    self.logger.info(f">>> ACTION: FOLD {draw_name} (bad odds: {amount_to_call}/{pot_size})")
                    return self.try_check_fold(valid_actions, f"Folding {draw_name} (bad pot odds)")

            # No made hand or draw - check and give up
            return self.try_check_fold(valid_actions, "No made hand or draw - checking/folding")

        # Fallback: try to take the safest action
        return self.try_check_fold(valid_actions, "Fallback reached - checking/folding")

