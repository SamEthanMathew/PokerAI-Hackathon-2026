import bisect
import os
import pickle
from itertools import combinations

import numpy as np

from agents.agent import Agent
from gym_env import PokerEnv

# Unified action space (must match train_mccfr_gpu.py)
FOLD, CHECK, CALL, RAISE_3X, RAISE_25, RAISE_70, RAISE_130 = range(7)
NUM_ABSTRACT_ACTIONS = 7
MAX_BET = 100


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType

        base = os.path.dirname(__file__)
        with open(os.path.join(base, "hand_ranks.pkl"), "rb") as f:
            self.hand_ranks = pickle.load(f)
        with open(os.path.join(base, "buckets.pkl"), "rb") as f:
            data = pickle.load(f)
            self.preflop_buckets = data["preflop_buckets"]
            self.postflop_thresholds = data["postflop_thresholds"]
            self.num_buckets = data["num_buckets"]
        with open(os.path.join(base, "strategy.pkl"), "rb") as f:
            data = pickle.load(f)
            self.strategy = data["strategy"]
        discard_path = os.path.join(base, "discard_equity.pkl")
        if os.path.exists(discard_path):
            with open(discard_path, "rb") as f:
                self.discard_table = pickle.load(f)
        else:
            self.discard_table = None

        # EHS lookup tables (exact forward-looking equity)
        ehs_path = os.path.join(base, "ehs_tables.pkl")
        if os.path.exists(ehs_path):
            with open(ehs_path, "rb") as f:
                ehs_data = pickle.load(f)
            self.ehs_flop = ehs_data.get("flop")
            self.ehs_turn = ehs_data.get("turn")
        else:
            self.ehs_flop = None
            self.ehs_turn = None

        self._reset_hand_state()

    def __name__(self):
        return "PlayerAgent"

    def _reset_hand_state(self):
        self.prev_street = -1
        self.street_history = ()
        self.is_sb = None
        # Fold-when-ahead tracking (persists across hands)
        if not hasattr(self, 'bankroll'):
            self.bankroll = 0
            self.total_hands = 1000

    # ── Hand evaluation ───────────────────────────────────────────

    def best_hand_rank(self, hole_cards, community_cards):
        all_cards = hole_cards + community_cards
        if len(all_cards) == 5:
            return self.hand_ranks[tuple(sorted(all_cards))]
        best = float('inf')
        for combo in combinations(all_cards, 5):
            rank = self.hand_ranks[tuple(sorted(combo))]
            if rank < best:
                best = rank
        return best

    def _discard_weight(self, opp_keep, opp_discards, flop):
        """How likely is it that opponent kept opp_keep given they discarded opp_discards?

        Reconstructs opponent's 5-card hand and scores all 10 possible keeps.
        Returns a weight: 1.0 if best keep, decaying for worse keeps.
        Also considers draw potential (opponent may keep draws over made hands).
        """
        opp_hand5 = list(opp_keep) + list(opp_discards)
        keep_scores = []
        for i, j in combinations(range(5), 2):
            keep = (opp_hand5[i], opp_hand5[j])
            hand5 = tuple(sorted(keep + tuple(flop)))
            rank = self.hand_ranks.get(hand5, 999999)
            draw = self._draw_bonus(keep, flop)
            score = -rank + draw * 1000
            keep_scores.append((frozenset(keep), score))

        keep_scores.sort(key=lambda x: x[1], reverse=True)
        target = frozenset(opp_keep)

        for rank_pos, (keep_set, _) in enumerate(keep_scores):
            if keep_set == target:
                if rank_pos == 0:
                    return 1.0
                elif rank_pos == 1:
                    return 0.5
                elif rank_pos == 2:
                    return 0.2
                return 0.05
        return 0.05

    def _opp_rank_to_bucket(self, opp_rank):
        """Approximate opponent bucket from their hand rank.

        Maps raw hand rank (0=best, 1331=worst) to a bucket index.
        Uses a simple linear mapping — fast enough for runtime.
        """
        MAX_RANK = 1332
        strength = 1.0 - min(opp_rank, MAX_RANK - 1) / MAX_RANK
        return min(int(strength * self.num_buckets), self.num_buckets - 1)

    def _opp_action_weight(self, opp_bucket, street, opp_actions):
        """Probability opponent with given bucket would take observed actions.

        Walks through the opponent's action sequence, multiplying the MCCFR
        strategy probabilities at each step. Returns a weight in (0, 1].
        Hands the opponent would never play this way get very low weight.
        """
        if not opp_actions:
            return 1.0

        weight = 1.0
        history = ()
        for action in opp_actions:
            key = (opp_bucket, street, history)
            if key in self.strategy:
                strat = self.strategy[key]
                if action < len(strat):
                    prob = float(strat[action])
                    weight *= max(prob, 0.01)  # floor to avoid zeroing out
                else:
                    weight *= 0.05
            else:
                weight *= 0.1  # unknown node — slight discount
            history = history + (action,)

        return weight

    def compute_equity(self, hole_cards, community_cards, dead_cards,
                        opp_discards=None, use_range_adjust=False,
                        street=None, opp_actions=None):
        """Weighted equity vs opponent range.

        Combines two weighting sources:
        1. Discard consistency: how reasonable is opponent's keep choice
        2. Range adjustment: MCCFR strategy weights for observed actions
           (opponent hands that would play differently get downweighted)
        """
        all_dead = set(hole_cards) | set(community_cards) | set(dead_cards)
        remaining = [c for c in range(27) if c not in all_dead]
        my_rank = self.best_hand_rank(hole_cards, community_cards)

        flop = community_cards[:3] if len(community_cards) >= 3 else None
        use_discard_filter = (opp_discards is not None
                              and len(opp_discards) == 3
                              and flop is not None)

        weighted_wins = 0.0
        weighted_ties = 0.0
        total_weight = 0.0

        for opp in combinations(remaining, 2):
            w = 1.0
            if use_discard_filter:
                w = self._discard_weight(opp, opp_discards, flop)

            opp_rank = self.best_hand_rank(list(opp), community_cards)

            # Range adjustment: weight by how likely opponent plays this way
            if use_range_adjust and street is not None and opp_actions:
                opp_bucket = self._opp_rank_to_bucket(opp_rank)
                action_w = self._opp_action_weight(opp_bucket, street, opp_actions)
                w *= action_w

            if my_rank < opp_rank:
                weighted_wins += w
            elif my_rank == opp_rank:
                weighted_ties += w
            total_weight += w

        if total_weight <= 0:
            return 0.5
        return (weighted_wins + 0.5 * weighted_ties) / total_weight

    # ── Discard ───────────────────────────────────────────────────

    def _canonicalize_discard(self, keep, flop):
        """Suit-canonicalize (keep2, flop3) to match discard table keys."""
        suit_map = {}
        next_suit = 0
        for c in sorted(flop) + sorted(keep):
            s = c // 9
            if s not in suit_map:
                suit_map[s] = next_suit
                next_suit += 1
        canon_keep = tuple(sorted(suit_map[c // 9] * 9 + c % 9 for c in keep))
        canon_flop = tuple(sorted(suit_map[c // 9] * 9 + c % 9 for c in flop))
        return (canon_keep, canon_flop)

    def _flush_nuttedness(self, keep_suits, keep_ranks, community_cards):
        """How nutted is our flush draw? Returns 0.0-1.0."""
        comm_suits = [c // 9 for c in community_cards]
        comm_ranks = [c % 9 for c in community_cards]
        best_nut = 0.0
        for s in set(keep_suits):
            my_ranks_in_suit = [keep_ranks[i] for i in range(2) if keep_suits[i] == s]
            if not my_ranks_in_suit:
                continue
            max_mine = max(my_ranks_in_suit)
            on_board = set(comm_ranks[i] for i in range(len(community_cards))
                          if comm_suits[i] == s)
            higher = sum(1 for r in range(max_mine + 1, 9)
                         if r not in on_board and r not in set(my_ranks_in_suit))
            if higher == 0:
                best_nut = max(best_nut, 1.0)
            elif higher == 1:
                best_nut = max(best_nut, 0.6)
            elif higher == 2:
                best_nut = max(best_nut, 0.25)
            else:
                best_nut = max(best_nut, 0.05)
        return best_nut

    def _draw_bonus(self, keep, community_cards):
        """Bonus for draw potential scaled by nuttedness.

        Nut flush draw ~0.13, low flush ~0.01.
        Nut-end straight ~0.10, idiot-end ~0.01.
        Returns value in [0, 0.30] range to blend with equity.
        """
        k0, k1 = keep
        all_cards = list(keep) + list(community_cards)
        suits = [c // 9 for c in all_cards]
        ranks = [c % 9 for c in all_cards]
        keep_suits = [k0 // 9, k1 // 9]
        keep_ranks = [k0 % 9, k1 % 9]
        comm_ranks = [c % 9 for c in community_cards]

        bonus = 0.0
        flush_detected = False

        # ── Flush draw scaled by nuttedness ──
        suit_counts = [0, 0, 0]
        for s in suits:
            suit_counts[s] += 1
        for s in set(keep_suits):
            n_suited = suit_counts[s]
            n_mine = keep_suits.count(s)
            if n_suited >= 4 and n_mine >= 1:
                nut = self._flush_nuttedness(keep_suits, keep_ranks, community_cards)
                base = 0.14 if n_mine == 2 else 0.10
                bonus += base * nut
                flush_detected = True
            elif n_suited >= 3 and n_mine == 2:
                nut = self._flush_nuttedness(keep_suits, keep_ranks, community_cards)
                bonus += 0.05 * nut

        # ── Straight draw scaled by nuttedness ──
        rank_set = set(ranks)
        keep_ranks_ext = set(keep_ranks)
        if 8 in rank_set:
            rank_set.add(-1)
        if 8 in keep_ranks_ext:
            keep_ranks_ext.add(-1)

        comm_rank_set = set(comm_ranks)
        if 8 in comm_rank_set:
            comm_rank_set.add(-1)

        best_str_bonus = 0.0
        for base in range(-1, 8):
            window = set(range(base, base + 5))
            overlap = len(rank_set & window)
            keep_in = len(keep_ranks_ext & window)
            if keep_in < 1 or overlap < 3:
                continue
            # Nuttedness: are we at the top end?
            top_of_window = base + 4
            max_keep = max(r for r in keep_ranks_ext if r in window)
            position = top_of_window - max_keep
            if position == 0:
                nut_factor = 1.0
            elif position == 1:
                nut_factor = 0.5
            else:
                nut_factor = 0.08  # idiot end
            # Discount if higher straights possible
            for hbase in range(base + 1, 8):
                hwindow = set(range(hbase, hbase + 5))
                if len(comm_rank_set & hwindow) >= 3:
                    nut_factor *= 0.5
                    break
            if overlap >= 4:
                best_str_bonus = max(best_str_bonus, 0.10 * nut_factor)
            elif overlap >= 3 and keep_in >= 2:
                best_str_bonus = max(best_str_bonus, 0.04 * nut_factor)
        bonus += best_str_bonus

        # ── Suited connectors (straight flush = absolute nuts) ──
        if keep_suits[0] == keep_suits[1]:
            gap = abs(keep_ranks[1] - keep_ranks[0])
            if 8 in keep_ranks and 0 in keep_ranks:
                gap = 1
            if gap <= 2:
                bonus += 0.04
                if max(keep_ranks) >= 6:
                    bonus += 0.02
                if flush_detected:
                    bonus += 0.03  # combo draw

        return min(bonus, 0.30)

    def _forward_equity(self, hole_cards, community_cards, dead_cards,
                         opp_discards=None, use_range_adjust=False,
                         street=None, opp_actions=None, num_runouts=5):
        """Expected Hand Strength: average equity across random future runouts.

        Naturally values draws by simulating completions to river.
        No hardcoded heuristics — a flush draw gets high EHS because it
        wins in ~35% of runouts where it completes.
        """
        nc = len(community_cards)
        cards_needed = 5 - nc
        if cards_needed <= 0:
            return self.compute_equity(
                hole_cards, community_cards, dead_cards,
                opp_discards=opp_discards,
                use_range_adjust=use_range_adjust,
                street=street, opp_actions=opp_actions)

        all_dead = set(hole_cards) | set(community_cards) | set(dead_cards)
        remaining = [c for c in range(27) if c not in all_dead]

        if len(remaining) < cards_needed:
            return self.compute_equity(
                hole_cards, community_cards, dead_cards,
                opp_discards=opp_discards,
                use_range_adjust=use_range_adjust,
                street=street, opp_actions=opp_actions)

        total_eq = 0.0
        for _ in range(num_runouts):
            idx = np.random.choice(len(remaining), size=cards_needed, replace=False)
            runout = [remaining[i] for i in idx]
            full_comm = list(community_cards) + runout
            full_dead = dead_cards | set(runout)
            total_eq += self.compute_equity(
                hole_cards, full_comm, full_dead,
                opp_discards=opp_discards,
                use_range_adjust=use_range_adjust,
                street=3, opp_actions=opp_actions)

        return total_eq / num_runouts

    def _river_equity(self, keep, community_cards, dead, num_samples=50):
        """Forward-looking equity: average equity across sampled river runouts.

        Instead of evaluating on the current board (where draws have no value),
        sample random turn+river cards and compute equity on the full 5-card board.
        This properly values draws by their probability of improving.
        """
        remaining = [c for c in range(27) if c not in dead and c not in set(keep)
                     and c not in set(community_cards)]
        nc = len(community_cards)
        cards_needed = 5 - nc  # how many more community cards to deal

        if cards_needed <= 0:
            # Already at river, just compute equity directly
            return self.compute_equity(keep, community_cards, dead)

        rng = np.random.RandomState()
        total_eq = 0.0
        for _ in range(num_samples):
            # Sample remaining community cards
            runout_idx = rng.choice(len(remaining), size=cards_needed, replace=False)
            runout = [remaining[i] for i in runout_idx]
            full_board = list(community_cards) + runout
            full_dead = dead | set(runout)
            eq = self.compute_equity(keep, full_board, full_dead)
            total_eq += eq
        return total_eq / num_samples

    def choose_discard(self, observation):
        hole_cards = [int(c) for c in observation['my_cards'] if int(c) != -1]
        community_cards = [int(c) for c in observation['community_cards'] if int(c) != -1]
        flop = community_cards[:3]

        best_score = -1.0
        best_keep = (0, 1)
        for i, j in combinations(range(len(hole_cards)), 2):
            keep = (hole_cards[i], hole_cards[j])
            if self.discard_table is not None:
                canon = self._canonicalize_discard(keep, flop)
                equity = self.discard_table.get(canon, 0.5)
            else:
                # Fallback: flop equity + draw bonus
                opp_discards = [int(c) for c in observation['opp_discarded_cards'] if int(c) != -1]
                discards = [hole_cards[k] for k in range(len(hole_cards)) if k != i and k != j]
                dead = set(discards) | set(opp_discards)
                equity = self.compute_equity(list(keep), community_cards, dead)
            score = equity
            if score > best_score:
                best_score = score
                best_keep = (i, j)
        return best_keep

    # ── Bucket computation ────────────────────────────────────────

    def _extract_opp_actions(self):
        """Extract opponent's actions from street history for range adjustment.

        In the street_history, actions alternate: if we act first, odd indices
        are opponent's; if opponent acts first, even indices are opponent's.
        Before we've added our own action, the last action is always opponent's.
        """
        if not self.street_history:
            return ()
        # street_history at this point contains opponent actions that preceded
        # our current decision — these are what we condition on
        return self.street_history

    def _get_bucket(self, observation):
        street = int(observation['street'])
        if street == 0:
            cards = tuple(sorted(int(c) for c in observation['my_cards'] if int(c) != -1))
            return self.preflop_buckets.get(cards, self.num_buckets // 2)
        else:
            hole = [int(c) for c in observation['my_cards'] if int(c) != -1]
            community = [int(c) for c in observation['community_cards'] if int(c) != -1]
            opp_discards = [int(c) for c in observation['opp_discarded_cards'] if int(c) != -1]
            my_discards = [int(c) for c in observation['my_discarded_cards'] if int(c) != -1]
            dead = set(opp_discards) | set(my_discards)

            # Use EHS lookup tables when available (exact, matches training)
            street_name = {1: "flop", 2: "turn", 3: "river"}[street]
            h = tuple(sorted(hole))
            comm_sorted = tuple(sorted(community))

            if street == 1 and self.ehs_flop is not None:
                flop_key = (h[0], h[1]) + comm_sorted
                equity = self.ehs_flop.get(flop_key)
                if equity is not None:
                    return bisect.bisect_right(
                        self.postflop_thresholds[street_name], equity)

            if street == 2 and self.ehs_turn is not None:
                turn_key = (h[0], h[1]) + comm_sorted
                equity = self.ehs_turn.get(turn_key)
                if equity is not None:
                    return bisect.bisect_right(
                        self.postflop_thresholds[street_name], equity)

            # Fallback: compute equity (river, or if EHS table missing)
            opp_actions = self._extract_opp_actions()
            use_range = len(opp_actions) > 0

            if street < 3:
                equity = self._forward_equity(
                    hole, community, dead,
                    opp_discards=opp_discards,
                    use_range_adjust=use_range,
                    street=street,
                    opp_actions=opp_actions,
                    num_runouts=5)
            else:
                equity = self.compute_equity(
                    hole, community, dead,
                    opp_discards=opp_discards,
                    use_range_adjust=use_range,
                    street=street,
                    opp_actions=opp_actions)

            return bisect.bisect_right(self.postflop_thresholds[street_name], equity)

    # ── Action history tracking ───────────────────────────────────

    def _detect_position(self, observation):
        if self.is_sb is None:
            if int(observation['street']) == 0:
                opp_action = str(observation.get('opp_last_action', 'None'))
                if opp_action == 'None':
                    self.is_sb = True
                else:
                    self.is_sb = False
            else:
                self.is_sb = True

    def _is_first_to_act(self, observation):
        street = int(observation['street'])
        if street == 0:
            return self.is_sb
        return not self.is_sb

    def _classify_opp_raise(self, observation):
        """Classify opponent's raise into abstract action based on street and sizing.

        Preflop: always RAISE_3X (single preflop raise size).
        Postflop: find closest matching abstract raise by computing what each
        abstract raise WOULD have produced, and pick the nearest one.
        """
        street = int(observation['street'])
        my_bet = int(observation['my_bet'])
        opp_bet = int(observation['opp_bet'])

        if street == 0:
            return RAISE_3X

        raise_amt = opp_bet - my_bet

        # Compute what each abstract raise would produce at this pot/bet state
        # Use pot before opponent's raise ≈ 2 * my_bet (both were equal before raise)
        pot_before = max(2 * my_bet, 1)
        min_r = int(observation.get('min_raise', 2))

        candidates = []
        for a, frac in [(RAISE_25, 0.25), (RAISE_70, 0.70), (RAISE_130, 1.30)]:
            expected_amt = max(int(frac * pot_before), min_r)
            candidates.append((a, abs(raise_amt - expected_amt)))

        # Pick closest match
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _update_history(self, observation):
        street = int(observation['street'])
        opp_action = str(observation.get('opp_last_action', 'None'))

        if street != self.prev_street:
            self.street_history = ()
            if not self._is_first_to_act(observation) and opp_action not in ('None', 'DISCARD'):
                if opp_action == 'FOLD':
                    self.street_history = (FOLD,)
                elif opp_action == 'CHECK':
                    self.street_history = (CHECK,)
                elif opp_action == 'CALL':
                    self.street_history = (CALL,)
                elif opp_action == 'RAISE':
                    self.street_history = (self._classify_opp_raise(observation),)
        else:
            if opp_action not in ('None', 'DISCARD'):
                if opp_action == 'FOLD':
                    self.street_history += (FOLD,)
                elif opp_action == 'CHECK':
                    self.street_history += (CHECK,)
                elif opp_action == 'CALL':
                    self.street_history += (CALL,)
                elif opp_action == 'RAISE':
                    self.street_history += (self._classify_opp_raise(observation),)


    # ── MDF gate for off-tree bet sizes ──────────────────────────

    def _continuing_range_percentile(self, bucket, street, history):
        """Compute our percentile within the continuing range at this node.

        Looks at all buckets that reach this (street, history) in the
        strategy table and computes what fraction of that range we beat.
        Only counts buckets that would have played the same way we did
        (i.e., have a strategy entry for this exact action sequence).
        """
        # Collect (bucket_id, defend_weight) for all buckets at this node
        # defend_weight = probability that this bucket reaches here and
        # would continue (call or raise, not fold)
        continuing = []
        for b in range(self.num_buckets):
            key = (b, street, history)
            if key in self.strategy:
                # This bucket has a strategy at this node = it reaches here
                continuing.append(b)

        if len(continuing) <= 1:
            return 0.5  # can't determine range position

        # Our position among the buckets that reach this node
        below = sum(1 for b in continuing if b < bucket)
        return below / len(continuing)

    def _mdf_gate(self, strategy, observation, bucket):
        """MDF-based defense for oversized bets outside trained abstractions.

        When opponent bets far larger than any trained raise size, the MCCFR
        strategy is unreliable (never saw this size). Apply MDF to our
        *continuing range* — only hands that would have played identically
        to reach this decision point.

          MDF = pot_before / (pot_before + bet_size)
          Bottom (1-MDF) of continuing range folds, top MDF defends.
        """
        my_bet = int(observation['my_bet'])
        opp_bet = int(observation['opp_bet'])
        street = int(observation['street'])

        if opp_bet <= my_bet:
            return strategy  # not facing a bet

        # What's the largest trained raise for this street?
        if street == 0:
            max_trained_total = 3.0 * max(my_bet, 2)
        else:
            pot_before = max(2 * my_bet, 1)
            max_trained_total = my_bet + 1.3 * pot_before

        # Only intervene for bets significantly above trained range
        if opp_bet <= max_trained_total * 1.5:
            return strategy  # trust MCCFR

        # MDF: fraction of range that must defend
        bet_size = opp_bet - my_bet
        pot_before = max(2 * my_bet, 1)
        mdf = pot_before / (pot_before + bet_size)

        # Our percentile within the continuing range (not full range)
        percentile = self._continuing_range_percentile(
            bucket, street, self.street_history)

        # Fold threshold: bottom (1-MDF) of continuing range should fold
        fold_cutoff = 1.0 - mdf

        if percentile >= fold_cutoff:
            # We're in the defending portion — remove fold, use MCCFR
            strategy[FOLD] = 0.0
            total = strategy.sum()
            if total > 0:
                strategy /= total
            else:
                strategy[CALL] = 1.0
        else:
            # We're in the folding portion
            # Smooth transition near the boundary
            depth_in_fold = (fold_cutoff - percentile) / max(fold_cutoff, 0.01)
            fold_prob = 0.5 + 0.5 * min(depth_in_fold, 1.0)
            strategy[FOLD] = fold_prob
            remain = 1.0 - fold_prob
            non_fold = strategy.copy()
            non_fold[FOLD] = 0.0
            nf_total = non_fold.sum()
            if nf_total > 0:
                for a in range(1, NUM_ABSTRACT_ACTIONS):
                    strategy[a] = remain * (non_fold[a] / nf_total)
            else:
                strategy[CALL] = remain

        return strategy

    # ── Strategy lookup ───────────────────────────────────────────

    def _get_abstract_valid(self, valid, street, observation, bucket=None):
        """Build abstract valid action mask, deduplicating raise sizes.

        After 2 raises in the street, only the top 2% (true nuts) can
        initiate another raise. All other hands are limited to call/fold.
        This prevents us from bloating pots with marginal hands in
        under-trained deep nodes, while still allowing full response
        to opponent aggression via trained MCCFR strategies.
        """
        av = np.zeros(NUM_ABSTRACT_ACTIONS, dtype=np.float64)
        av[FOLD] = 1.0
        if valid[self.action_types.CHECK.value]:
            av[CHECK] = 1.0
        if valid[self.action_types.CALL.value]:
            av[CALL] = 1.0
        if valid[self.action_types.RAISE.value]:
            num_raises = sum(1 for a in self.street_history
                             if a in (RAISE_3X, RAISE_25, RAISE_70, RAISE_130))
            is_nut = (bucket is not None and
                      bucket >= int(self.num_buckets * 0.98))  # top 2% only

            # After 2 raises in the street, only nuts can raise again
            if num_raises >= 2 and not is_nut:
                return av  # call/fold only

            pot = int(observation['my_bet']) + int(observation['opp_bet'])
            opp_bet = int(observation['opp_bet'])
            min_r = int(observation['min_raise'])
            max_r = int(observation['max_raise'])

            raise_list = [RAISE_3X] if street == 0 else [RAISE_25, RAISE_70, RAISE_130]
            seen_amounts = set()
            for a in raise_list:
                if a == RAISE_3X:
                    target = int(3.0 * opp_bet)
                    amt = max(target - opp_bet, min_r)
                elif a == RAISE_25:
                    amt = max(int(0.25 * pot), min_r)
                elif a == RAISE_70:
                    amt = max(int(0.70 * pot), min_r)
                else:  # RAISE_130
                    amt = max(int(1.30 * pot), min_r)
                amt = max(min(amt, max_r), min(min_r, max_r))
                if amt not in seen_amounts:
                    av[a] = 1.0
                    seen_amounts.add(amt)
        return av

    def _lookup_strategy(self, info_key, abstract_valid):
        if info_key in self.strategy:
            strat = self.strategy[info_key].copy()
        else:
            # No trained strategy — use a conservative default
            # that gets more fold-heavy for deeper action sequences
            _, _, history = info_key
            num_raises = sum(1 for a in history
                             if a in (RAISE_3X, RAISE_25, RAISE_70, RAISE_130))
            strat = np.zeros(NUM_ABSTRACT_ACTIONS, dtype=np.float64)
            if num_raises == 0:
                # No raises yet: check/call favored
                strat[FOLD] = 0.2
                strat[CHECK] = 0.4
                strat[CALL] = 0.3
                strat[RAISE_25] = 0.05
                strat[RAISE_70] = 0.03
                strat[RAISE_130] = 0.02
            elif num_raises == 1:
                # Facing first raise: mostly fold/call
                strat[FOLD] = 0.5
                strat[CALL] = 0.35
                strat[RAISE_25] = 0.08
                strat[RAISE_70] = 0.05
                strat[RAISE_130] = 0.02
            else:
                # 2+ raises (re-raise war): heavily fold
                strat[FOLD] = 0.7
                strat[CALL] = 0.25
                strat[RAISE_25] = 0.03
                strat[RAISE_70] = 0.01
                strat[RAISE_130] = 0.01

        strat *= abstract_valid
        total = strat.sum()
        if total > 0:
            return strat / total
        return abstract_valid / abstract_valid.sum()

    def _tighten_preflop_raises(self, strategy, abstract_valid, bucket):
        """Tighten preflop raising based on action context.

        Opening (no raise yet): top ~40% can raise, rest call/check/fold.
        Facing a raise (3-bet spot): only top ~15% re-raise.
        Facing a re-raise (4-bet+): only top ~8% continue raising.

        Hands below the threshold have raise probability redirected to
        call (if available) or check. Fold decisions stay with MCCFR.
        """
        raise_prob = strategy[RAISE_3X] if abstract_valid[RAISE_3X] else 0.0
        if raise_prob <= 0:
            return strategy

        # Count how many raises are in the history (opponent + ours)
        num_raises = sum(1 for a in self.street_history if a == RAISE_3X)

        if num_raises == 0:
            # Opening raise: top 40%
            threshold = int(self.num_buckets * 0.60)
        elif num_raises == 1:
            # Facing a raise (3-bet spot): top 15%
            threshold = int(self.num_buckets * 0.85)
        else:
            # Facing re-raise (4-bet+): top 8%
            threshold = int(self.num_buckets * 0.92)

        if bucket >= threshold:
            return strategy  # strong enough to raise

        # Redirect raise probability to call/check
        strategy[RAISE_3X] = 0.0
        if abstract_valid[CALL]:
            strategy[CALL] += raise_prob
        elif abstract_valid[CHECK]:
            strategy[CHECK] += raise_prob

        total = strategy.sum()
        if total > 0:
            strategy /= total
        return strategy

    def _enforce_mdf(self, strategy, observation, bucket):
        """Runtime MDF enforcement: cap fold probability for strong hands.

        If we're facing a bet and our bucket is in the top MDF% of the range,
        we must defend (fold probability capped at 1 - MDF).

        On the river, trust the MCCFR strategy more — only enforce MDF for
        clearly strong hands (top 30% of range) since there are no more
        cards to come and the trained strategy already knows correct frequencies.
        """
        street = int(observation['street'])
        mdf = self._compute_mdf(observation)
        if mdf <= 0:
            return strategy

        max_fold = 1.0 - mdf
        # Bucket threshold: hands at or above this must defend
        defend_threshold = int(self.num_buckets * (1.0 - mdf))

        # On the river, only enforce for clearly strong hands
        # The MCCFR strategy has learned river fold/call correctly
        if street == 3:
            defend_threshold = max(defend_threshold, int(self.num_buckets * 0.70))

        if bucket >= defend_threshold and strategy[FOLD] > max_fold:
            excess = strategy[FOLD] - max_fold
            strategy[FOLD] = max_fold

            # Redistribute to non-fold actions proportionally
            non_fold = strategy.copy()
            non_fold[FOLD] = 0.0
            nf_total = non_fold.sum()
            if nf_total > 0:
                for a in range(1, NUM_ABSTRACT_ACTIONS):
                    strategy[a] += excess * (non_fold[a] / nf_total)
            else:
                strategy[CALL] += excess

            # Renormalize
            total = strategy.sum()
            if total > 0:
                strategy /= total

        return strategy

    # ── Abstract → concrete action ────────────────────────────────

    def _abstract_to_concrete(self, abstract_action, observation):
        AT = self.action_types
        if abstract_action == FOLD:
            return AT.FOLD.value, 0, 0, 0
        elif abstract_action == CHECK:
            return AT.CHECK.value, 0, 0, 0
        elif abstract_action == CALL:
            return AT.CALL.value, 0, 0, 0
        else:
            pot = int(observation['my_bet']) + int(observation['opp_bet'])
            opp_bet = int(observation['opp_bet'])
            min_r = int(observation['min_raise'])
            max_r = int(observation['max_raise'])
            if abstract_action == RAISE_3X:
                target = int(3.0 * opp_bet)
                amt = max(target - opp_bet, min_r)
            elif abstract_action == RAISE_25:
                amt = max(int(0.25 * pot), min_r)
            elif abstract_action == RAISE_70:
                amt = max(int(0.70 * pot), min_r)
            else:  # RAISE_130
                amt = max(int(1.30 * pot), min_r)
            amt = max(min(amt, max_r), min(min_r, max_r))
            return AT.RAISE.value, int(amt), 0, 0

    # ── Main action logic ─────────────────────────────────────────

    def act(self, observation, reward, terminated, truncated, info):
        valid = observation["valid_actions"]
        AT = self.action_types

        self._detect_position(observation)

        if valid[AT.DISCARD.value]:
            keep_i, keep_j = self.choose_discard(observation)
            return AT.DISCARD.value, 0, keep_i, keep_j

        # Fold-when-ahead: if our lead covers worst-case losses, don't risk chips
        if info and 'hand_number' in info:
            hand_num = info['hand_number']
            remaining = self.total_hands - hand_num - 1
            my_bet = int(observation['my_bet'])
            if self.bankroll - my_bet > remaining * 1.5 + 5:
                # Auto-fold mode: check if free, fold if facing a bet
                if valid[AT.CHECK.value]:
                    return AT.CHECK.value, 0, 0, 0
                elif valid[AT.FOLD.value]:
                    return AT.FOLD.value, 0, 0, 0

        self._update_history(observation)

        street = int(observation['street'])
        bucket = self._get_bucket(observation)
        info_key = (bucket, street, self.street_history)

        abstract_valid = self._get_abstract_valid(valid, street, observation,
                                                    bucket=bucket)
        strategy = self._lookup_strategy(info_key, abstract_valid)

        # MDF gate: fold weak hands against oversized off-tree bets
        strategy = self._mdf_gate(strategy, observation, bucket)

        # Tighten preflop raises based on context
        if street == 0:
            strategy = self._tighten_preflop_raises(strategy, abstract_valid, bucket)

        abstract_action = int(np.random.choice(NUM_ABSTRACT_ACTIONS, p=strategy))

        if abstract_action == FOLD and abstract_valid[CHECK]:
            abstract_action = CHECK

        self.street_history += (abstract_action,)
        self.prev_street = street

        return self._abstract_to_concrete(abstract_action, observation)

    def observe(self, observation, reward, terminated, truncated, info):
        if terminated:
            self.bankroll += reward
            self._reset_hand_state()
