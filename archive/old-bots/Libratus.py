"""
Libratus-Lite: A loose-aggressive, table-driven, mixed-strategy poker bot.
Runtime component -- lightweight policy lookup with MC discard evaluation.
"""
import random
from collections import Counter
from itertools import combinations

from agents.agent import Agent
from gym_env import PokerEnv

# ── Inline lookup tables (auto-generated) ──

# Equity by keep archetype (higher = stronger)
KEEP_EQUITY = {'premium_pair': 0.7275, 'medium_pair': 0.635, 'low_pair': 0.498, 'suited_connector': 0.465, 'suited_semi': 0.5885, 'suited_gapper': 0.5035, 'offsuit_connector': 0.461, 'offsuit_other': 0.501}
# Opponent posterior: P(keep_bucket | discard_bucket, flop_bucket)
POSTERIOR = {'suited_cluster|medium': {'suited_connector': 0.18677213799165018, 'offsuit_other': 0.18193803559657218, 'low_pair': 0.11052515930564712, 'suited_semi': 0.21907273126785323, 'offsuit_connector': 0.0911887497253351, 'suited_gapper': 0.08679411118435508, 'medium_pair': 0.03317952098439903, 'premium_pair': 0.09052955394418809}, 'suited_cluster|wet': {'suited_semi': 0.20992676973148902, 'suited_gapper': 0.10089503661513426, 'low_pair': 0.0951993490642799, 'offsuit_connector': 0.12530512611879577, 'suited_connector': 0.18633034987794955, 'offsuit_other': 0.1871440195280716, 'medium_pair': 0.03254678600488202, 'premium_pair': 0.06265256305939788}, 'high_junk|wet': {'suited_semi': 0.2, 'premium_pair': 0.075, 'medium_pair': 0.05, 'offsuit_connector': 0.175, 'offsuit_other': 0.1, 'low_pair': 0.125, 'suited_connector': 0.2, 'suited_gapper': 0.075}, 'discarded_pair|medium': {'offsuit_connector': 0.0996290408055114, 'suited_semi': 0.21038685744568097, 'suited_connector': 0.18071012188659247, 'offsuit_other': 0.21356650768415475, 'suited_gapper': 0.09114997350291468, 'premium_pair': 0.09273979862215156, 'low_pair': 0.09009009009009009, 'medium_pair': 0.02172760996290408}, 'suited_cluster|dry': {'suited_gapper': 0.08571428571428572, 'suited_connector': 0.16857142857142857, 'offsuit_other': 0.21142857142857144, 'suited_semi': 0.17714285714285713, 'offsuit_connector': 0.09142857142857143, 'premium_pair': 0.11142857142857143, 'low_pair': 0.11428571428571428, 'medium_pair': 0.04}, 'discarded_pair|dry': {'premium_pair': 0.09395973154362416, 'low_pair': 0.10067114093959731, 'suited_semi': 0.20134228187919462, 'offsuit_other': 0.26174496644295303, 'suited_connector': 0.14093959731543623, 'suited_gapper': 0.09395973154362416, 'offsuit_connector': 0.06040268456375839, 'medium_pair': 0.04697986577181208}, 'discarded_pair|wet': {'offsuit_other': 0.22494432071269488, 'suited_connector': 0.16258351893095768, 'suited_gapper': 0.12472160356347439, 'offsuit_connector': 0.1403118040089087, 'premium_pair': 0.0645879732739421, 'low_pair': 0.08240534521158129, 'suited_semi': 0.18485523385300667, 'medium_pair': 0.015590200445434299}, 'connected_cluster|dry': {'offsuit_other': 0.2857142857142857, 'premium_pair': 0.21428571428571427, 'suited_connector': 0.07142857142857142, 'low_pair': 0.07142857142857142, 'suited_semi': 0.21428571428571427, 'offsuit_connector': 0.07142857142857142, 'suited_gapper': 0.07142857142857142}, 'low_junk|wet': {'low_pair': 0.0425531914893617, 'suited_connector': 0.2978723404255319, 'suited_semi': 0.2765957446808511, 'suited_gapper': 0.1276595744680851, 'medium_pair': 0.02127659574468085, 'offsuit_other': 0.14893617021276595, 'offsuit_connector': 0.02127659574468085, 'premium_pair': 0.06382978723404255}, 'low_junk|medium': {'suited_semi': 0.18846153846153846, 'offsuit_other': 0.19230769230769232, 'suited_connector': 0.17692307692307693, 'premium_pair': 0.1576923076923077, 'offsuit_connector': 0.1, 'low_pair': 0.08846153846153847, 'suited_gapper': 0.05, 'medium_pair': 0.046153846153846156}, 'mixed_discard|medium': {'suited_semi': 0.23042505592841164, 'offsuit_other': 0.19239373601789708, 'suited_connector': 0.17225950782997762, 'premium_pair': 0.10514541387024609, 'low_pair': 0.12527964205816555, 'offsuit_connector': 0.09619686800894854, 'suited_gapper': 0.06040268456375839, 'medium_pair': 0.017897091722595078}, 'high_junk|medium': {'offsuit_other': 0.13821138211382114, 'suited_gapper': 0.04878048780487805, 'low_pair': 0.13414634146341464, 'suited_semi': 0.23577235772357724, 'offsuit_connector': 0.11382113821138211, 'suited_connector': 0.2073170731707317, 'premium_pair': 0.06910569105691057, 'medium_pair': 0.052845528455284556}, 'connected_cluster|medium': {'low_pair': 0.08552631578947369, 'suited_connector': 0.24342105263157895, 'suited_semi': 0.19736842105263158, 'offsuit_other': 0.16447368421052633, 'premium_pair': 0.05263157894736842, 'offsuit_connector': 0.16447368421052633, 'suited_gapper': 0.05263157894736842, 'medium_pair': 0.039473684210526314}, 'low_junk|dry': {'suited_gapper': 0.08333333333333333, 'premium_pair': 0.25, 'suited_connector': 0.16666666666666666, 'offsuit_connector': 0.041666666666666664, 'suited_semi': 0.25, 'medium_pair': 0.08333333333333333, 'offsuit_other': 0.08333333333333333, 'low_pair': 0.041666666666666664}, 'high_junk|dry': {'low_pair': 0.35, 'suited_connector': 0.2, 'offsuit_other': 0.15, 'suited_semi': 0.2, 'suited_gapper': 0.1}, 'connected_cluster|wet': {'suited_connector': 0.21739130434782608, 'offsuit_other': 0.13043478260869565, 'premium_pair': 0.13043478260869565, 'low_pair': 0.17391304347826086, 'suited_semi': 0.17391304347826086, 'offsuit_connector': 0.043478260869565216, 'suited_gapper': 0.13043478260869565}, 'mixed_discard|wet': {'offsuit_other': 0.08695652173913043, 'suited_semi': 0.2028985507246377, 'premium_pair': 0.13043478260869565, 'low_pair': 0.15942028985507245, 'suited_connector': 0.2318840579710145, 'suited_gapper': 0.08695652173913043, 'offsuit_connector': 0.10144927536231885}, 'mixed_discard|dry': {'suited_gapper': 0.11627906976744186, 'offsuit_connector': 0.09302325581395349, 'suited_semi': 0.13953488372093023, 'offsuit_other': 0.23255813953488372, 'medium_pair': 0.023255813953488372, 'suited_connector': 0.16279069767441862, 'low_pair': 0.18604651162790697, 'premium_pair': 0.046511627906976744}}
# Head-to-head matchup win rates
MATCHUPS = {'premium_pair|premium_pair': 0.5, 'premium_pair|medium_pair': 0.7765, 'medium_pair|premium_pair': 0.2235, 'premium_pair|low_pair': 0.752, 'low_pair|premium_pair': 0.248, 'premium_pair|suited_connector': 0.6453, 'suited_connector|premium_pair': 0.3547, 'premium_pair|suited_semi': 0.6684, 'suited_semi|premium_pair': 0.3316, 'premium_pair|suited_gapper': 0.6918, 'suited_gapper|premium_pair': 0.3082, 'premium_pair|offsuit_connector': 0.6851, 'offsuit_connector|premium_pair': 0.3149, 'premium_pair|offsuit_other': 0.7254, 'offsuit_other|premium_pair': 0.2746, 'medium_pair|medium_pair': 0.5, 'medium_pair|low_pair': 0.768, 'low_pair|medium_pair': 0.232, 'medium_pair|suited_connector': 0.6082, 'suited_connector|medium_pair': 0.3918, 'medium_pair|suited_semi': 0.5983, 'suited_semi|medium_pair': 0.4017, 'medium_pair|suited_gapper': 0.6515, 'suited_gapper|medium_pair': 0.3485, 'medium_pair|offsuit_connector': 0.6874, 'offsuit_connector|medium_pair': 0.3126, 'medium_pair|offsuit_other': 0.6827, 'offsuit_other|medium_pair': 0.3173, 'low_pair|low_pair': 0.5, 'low_pair|suited_connector': 0.5303, 'suited_connector|low_pair': 0.4697, 'low_pair|suited_semi': 0.5269, 'suited_semi|low_pair': 0.4731, 'low_pair|suited_gapper': 0.4947, 'suited_gapper|low_pair': 0.5053, 'low_pair|offsuit_connector': 0.4875, 'offsuit_connector|low_pair': 0.5125, 'low_pair|offsuit_other': 0.547, 'offsuit_other|low_pair': 0.453, 'suited_connector|suited_connector': 0.5, 'suited_connector|suited_semi': 0.4905, 'suited_semi|suited_connector': 0.5095, 'suited_connector|suited_gapper': 0.5399, 'suited_gapper|suited_connector': 0.4601, 'suited_connector|offsuit_connector': 0.5191, 'offsuit_connector|suited_connector': 0.4809, 'suited_connector|offsuit_other': 0.5244, 'offsuit_other|suited_connector': 0.4756, 'suited_semi|suited_semi': 0.5, 'suited_semi|suited_gapper': 0.5355, 'suited_gapper|suited_semi': 0.4645, 'suited_semi|offsuit_connector': 0.5139, 'offsuit_connector|suited_semi': 0.4861, 'suited_semi|offsuit_other': 0.5312, 'offsuit_other|suited_semi': 0.4688, 'suited_gapper|suited_gapper': 0.5, 'suited_gapper|offsuit_connector': 0.4845, 'offsuit_connector|suited_gapper': 0.5155, 'suited_gapper|offsuit_other': 0.5035, 'offsuit_other|suited_gapper': 0.4965, 'offsuit_connector|offsuit_connector': 0.5, 'offsuit_connector|offsuit_other': 0.4917, 'offsuit_other|offsuit_connector': 0.5083, 'offsuit_other|offsuit_other': 0.5}
# Betting policy: (street, position, strength, board, to_call) -> action probs
POLICY = {"(0, 'sb', 'monster', 'any', 'none')": {'fold': 0.0, 'check_call': 0.05, 'small_bet': 0.35, 'medium_bet': 0.35, 'large_bet': 0.15, 'jam': 0.1}, "(0, 'sb', 'strong', 'any', 'none')": {'fold': 0.0, 'check_call': 0.05, 'small_bet': 0.35, 'medium_bet': 0.35, 'large_bet': 0.15, 'jam': 0.1}, "(0, 'sb', 'good', 'any', 'none')": {'fold': 0.0, 'check_call': 0.2, 'small_bet': 0.45, 'medium_bet': 0.25, 'large_bet': 0.1, 'jam': 0.0}, "(0, 'sb', 'marginal', 'any', 'none')": {'fold': 0.05, 'check_call': 0.55, 'small_bet': 0.3, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'sb', 'weak', 'any', 'none')": {'fold': 0.15, 'check_call': 0.5, 'small_bet': 0.25, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'sb', 'monster', 'any', 'small')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'sb', 'strong', 'any', 'small')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'sb', 'good', 'any', 'small')": {'fold': 0.1, 'check_call': 0.5, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(0, 'sb', 'marginal', 'any', 'small')": {'fold': 0.3, 'check_call': 0.5, 'small_bet': 0.15, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'sb', 'weak', 'any', 'small')": {'fold': 0.6, 'check_call': 0.25, 'small_bet': 0.1, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'sb', 'monster', 'any', 'medium')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'sb', 'strong', 'any', 'medium')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'sb', 'good', 'any', 'medium')": {'fold': 0.1, 'check_call': 0.5, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(0, 'sb', 'marginal', 'any', 'medium')": {'fold': 0.3, 'check_call': 0.5, 'small_bet': 0.15, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'sb', 'weak', 'any', 'medium')": {'fold': 0.6, 'check_call': 0.25, 'small_bet': 0.1, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'sb', 'monster', 'any', 'large')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'sb', 'strong', 'any', 'large')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'sb', 'good', 'any', 'large')": {'fold': 0.1, 'check_call': 0.5, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(0, 'sb', 'marginal', 'any', 'large')": {'fold': 0.3, 'check_call': 0.5, 'small_bet': 0.15, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'sb', 'weak', 'any', 'large')": {'fold': 0.8, 'check_call': 0.15, 'small_bet': 0.05, 'medium_bet': 0.0, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'bb', 'monster', 'any', 'none')": {'fold': 0.0, 'check_call': 0.05, 'small_bet': 0.35, 'medium_bet': 0.35, 'large_bet': 0.15, 'jam': 0.1}, "(0, 'bb', 'strong', 'any', 'none')": {'fold': 0.0, 'check_call': 0.05, 'small_bet': 0.35, 'medium_bet': 0.35, 'large_bet': 0.15, 'jam': 0.1}, "(0, 'bb', 'good', 'any', 'none')": {'fold': 0.0, 'check_call': 0.2, 'small_bet': 0.45, 'medium_bet': 0.25, 'large_bet': 0.1, 'jam': 0.0}, "(0, 'bb', 'marginal', 'any', 'none')": {'fold': 0.05, 'check_call': 0.55, 'small_bet': 0.3, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'bb', 'weak', 'any', 'none')": {'fold': 0.15, 'check_call': 0.5, 'small_bet': 0.25, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'bb', 'monster', 'any', 'small')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'bb', 'strong', 'any', 'small')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'bb', 'good', 'any', 'small')": {'fold': 0.1, 'check_call': 0.5, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(0, 'bb', 'marginal', 'any', 'small')": {'fold': 0.3, 'check_call': 0.5, 'small_bet': 0.15, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'bb', 'weak', 'any', 'small')": {'fold': 0.6, 'check_call': 0.25, 'small_bet': 0.1, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'bb', 'monster', 'any', 'medium')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'bb', 'strong', 'any', 'medium')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'bb', 'good', 'any', 'medium')": {'fold': 0.1, 'check_call': 0.5, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(0, 'bb', 'marginal', 'any', 'medium')": {'fold': 0.3, 'check_call': 0.5, 'small_bet': 0.15, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'bb', 'weak', 'any', 'medium')": {'fold': 0.6, 'check_call': 0.25, 'small_bet': 0.1, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'bb', 'monster', 'any', 'large')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'bb', 'strong', 'any', 'large')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.2, 'medium_bet': 0.35, 'large_bet': 0.25, 'jam': 0.1}, "(0, 'bb', 'good', 'any', 'large')": {'fold': 0.1, 'check_call': 0.5, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(0, 'bb', 'marginal', 'any', 'large')": {'fold': 0.3, 'check_call': 0.5, 'small_bet': 0.15, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(0, 'bb', 'weak', 'any', 'large')": {'fold': 0.8, 'check_call': 0.15, 'small_bet': 0.05, 'medium_bet': 0.0, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'monster', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(1, 'sb', 'strong', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(1, 'sb', 'good', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(1, 'sb', 'marginal', 'wet', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'wet', 'none')": {'fold': 0.118, 'check_call': 0.471, 'small_bet': 0.235, 'medium_bet': 0.118, 'large_bet': 0.059, 'jam': 0.0}, "(1, 'sb', 'monster', 'wet', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'sb', 'strong', 'wet', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'sb', 'good', 'wet', 'small')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(1, 'sb', 'marginal', 'wet', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'wet', 'small')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'monster', 'wet', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'sb', 'strong', 'wet', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'sb', 'good', 'wet', 'medium')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(1, 'sb', 'marginal', 'wet', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'wet', 'medium')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'monster', 'wet', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'sb', 'strong', 'wet', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'sb', 'good', 'wet', 'large')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(1, 'sb', 'marginal', 'wet', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'wet', 'large')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'monster', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(1, 'sb', 'strong', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(1, 'sb', 'good', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(1, 'sb', 'marginal', 'medium', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'medium', 'none')": {'fold': 0.118, 'check_call': 0.471, 'small_bet': 0.235, 'medium_bet': 0.118, 'large_bet': 0.059, 'jam': 0.0}, "(1, 'sb', 'monster', 'medium', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'sb', 'strong', 'medium', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'sb', 'good', 'medium', 'small')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'sb', 'marginal', 'medium', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'medium', 'small')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'monster', 'medium', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'sb', 'strong', 'medium', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'sb', 'good', 'medium', 'medium')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'sb', 'marginal', 'medium', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'medium', 'medium')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'monster', 'medium', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'sb', 'strong', 'medium', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'sb', 'good', 'medium', 'large')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'sb', 'marginal', 'medium', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'medium', 'large')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'monster', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(1, 'sb', 'strong', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(1, 'sb', 'good', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(1, 'sb', 'marginal', 'dry', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'dry', 'none')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.316, 'medium_bet': 0.105, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'sb', 'monster', 'dry', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'sb', 'strong', 'dry', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'sb', 'good', 'dry', 'small')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'sb', 'marginal', 'dry', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'dry', 'small')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'monster', 'dry', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'sb', 'strong', 'dry', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'sb', 'good', 'dry', 'medium')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'sb', 'marginal', 'dry', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'dry', 'medium')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'monster', 'dry', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'sb', 'strong', 'dry', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'sb', 'good', 'dry', 'large')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'sb', 'marginal', 'dry', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'sb', 'weak', 'dry', 'large')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'monster', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(1, 'bb', 'strong', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(1, 'bb', 'good', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(1, 'bb', 'marginal', 'wet', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'wet', 'none')": {'fold': 0.118, 'check_call': 0.471, 'small_bet': 0.235, 'medium_bet': 0.118, 'large_bet': 0.059, 'jam': 0.0}, "(1, 'bb', 'monster', 'wet', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'bb', 'strong', 'wet', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'bb', 'good', 'wet', 'small')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(1, 'bb', 'marginal', 'wet', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'wet', 'small')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'monster', 'wet', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'bb', 'strong', 'wet', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'bb', 'good', 'wet', 'medium')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(1, 'bb', 'marginal', 'wet', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'wet', 'medium')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'monster', 'wet', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'bb', 'strong', 'wet', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'bb', 'good', 'wet', 'large')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(1, 'bb', 'marginal', 'wet', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'wet', 'large')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'monster', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(1, 'bb', 'strong', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(1, 'bb', 'good', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(1, 'bb', 'marginal', 'medium', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'medium', 'none')": {'fold': 0.118, 'check_call': 0.471, 'small_bet': 0.235, 'medium_bet': 0.118, 'large_bet': 0.059, 'jam': 0.0}, "(1, 'bb', 'monster', 'medium', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'bb', 'strong', 'medium', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'bb', 'good', 'medium', 'small')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'bb', 'marginal', 'medium', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'medium', 'small')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'monster', 'medium', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'bb', 'strong', 'medium', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'bb', 'good', 'medium', 'medium')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'bb', 'marginal', 'medium', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'medium', 'medium')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'monster', 'medium', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'bb', 'strong', 'medium', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'bb', 'good', 'medium', 'large')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'bb', 'marginal', 'medium', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'medium', 'large')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'monster', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(1, 'bb', 'strong', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(1, 'bb', 'good', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(1, 'bb', 'marginal', 'dry', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'dry', 'none')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.316, 'medium_bet': 0.105, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'bb', 'monster', 'dry', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'bb', 'strong', 'dry', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'bb', 'good', 'dry', 'small')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'bb', 'marginal', 'dry', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'dry', 'small')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'monster', 'dry', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'bb', 'strong', 'dry', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'bb', 'good', 'dry', 'medium')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'bb', 'marginal', 'dry', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'dry', 'medium')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'monster', 'dry', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(1, 'bb', 'strong', 'dry', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(1, 'bb', 'good', 'dry', 'large')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(1, 'bb', 'marginal', 'dry', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(1, 'bb', 'weak', 'dry', 'large')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'monster', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(2, 'sb', 'strong', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(2, 'sb', 'good', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(2, 'sb', 'marginal', 'wet', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'wet', 'none')": {'fold': 0.118, 'check_call': 0.471, 'small_bet': 0.235, 'medium_bet': 0.118, 'large_bet': 0.059, 'jam': 0.0}, "(2, 'sb', 'monster', 'wet', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'sb', 'strong', 'wet', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'sb', 'good', 'wet', 'small')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(2, 'sb', 'marginal', 'wet', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'wet', 'small')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'monster', 'wet', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'sb', 'strong', 'wet', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'sb', 'good', 'wet', 'medium')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(2, 'sb', 'marginal', 'wet', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'wet', 'medium')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'monster', 'wet', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'sb', 'strong', 'wet', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'sb', 'good', 'wet', 'large')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(2, 'sb', 'marginal', 'wet', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'wet', 'large')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'monster', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(2, 'sb', 'strong', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(2, 'sb', 'good', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(2, 'sb', 'marginal', 'medium', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'medium', 'none')": {'fold': 0.118, 'check_call': 0.471, 'small_bet': 0.235, 'medium_bet': 0.118, 'large_bet': 0.059, 'jam': 0.0}, "(2, 'sb', 'monster', 'medium', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'sb', 'strong', 'medium', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'sb', 'good', 'medium', 'small')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'sb', 'marginal', 'medium', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'medium', 'small')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'monster', 'medium', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'sb', 'strong', 'medium', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'sb', 'good', 'medium', 'medium')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'sb', 'marginal', 'medium', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'medium', 'medium')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'monster', 'medium', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'sb', 'strong', 'medium', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'sb', 'good', 'medium', 'large')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'sb', 'marginal', 'medium', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'medium', 'large')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'monster', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(2, 'sb', 'strong', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(2, 'sb', 'good', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(2, 'sb', 'marginal', 'dry', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'dry', 'none')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.316, 'medium_bet': 0.105, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'sb', 'monster', 'dry', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'sb', 'strong', 'dry', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'sb', 'good', 'dry', 'small')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'sb', 'marginal', 'dry', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'dry', 'small')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'monster', 'dry', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'sb', 'strong', 'dry', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'sb', 'good', 'dry', 'medium')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'sb', 'marginal', 'dry', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'dry', 'medium')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'monster', 'dry', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'sb', 'strong', 'dry', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'sb', 'good', 'dry', 'large')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'sb', 'marginal', 'dry', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'sb', 'weak', 'dry', 'large')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'monster', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(2, 'bb', 'strong', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(2, 'bb', 'good', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(2, 'bb', 'marginal', 'wet', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'wet', 'none')": {'fold': 0.118, 'check_call': 0.471, 'small_bet': 0.235, 'medium_bet': 0.118, 'large_bet': 0.059, 'jam': 0.0}, "(2, 'bb', 'monster', 'wet', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'bb', 'strong', 'wet', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'bb', 'good', 'wet', 'small')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(2, 'bb', 'marginal', 'wet', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'wet', 'small')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'monster', 'wet', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'bb', 'strong', 'wet', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'bb', 'good', 'wet', 'medium')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(2, 'bb', 'marginal', 'wet', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'wet', 'medium')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'monster', 'wet', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'bb', 'strong', 'wet', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'bb', 'good', 'wet', 'large')": {'fold': 0.15, 'check_call': 0.4, 'small_bet': 0.25, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(2, 'bb', 'marginal', 'wet', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'wet', 'large')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'monster', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(2, 'bb', 'strong', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(2, 'bb', 'good', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(2, 'bb', 'marginal', 'medium', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'medium', 'none')": {'fold': 0.118, 'check_call': 0.471, 'small_bet': 0.235, 'medium_bet': 0.118, 'large_bet': 0.059, 'jam': 0.0}, "(2, 'bb', 'monster', 'medium', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'bb', 'strong', 'medium', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'bb', 'good', 'medium', 'small')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'bb', 'marginal', 'medium', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'medium', 'small')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'monster', 'medium', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'bb', 'strong', 'medium', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'bb', 'good', 'medium', 'medium')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'bb', 'marginal', 'medium', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'medium', 'medium')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'monster', 'medium', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'bb', 'strong', 'medium', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'bb', 'good', 'medium', 'large')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'bb', 'marginal', 'medium', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'medium', 'large')": {'fold': 0.611, 'check_call': 0.222, 'small_bet': 0.111, 'medium_bet': 0.056, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'monster', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(2, 'bb', 'strong', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(2, 'bb', 'good', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(2, 'bb', 'marginal', 'dry', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'dry', 'none')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.316, 'medium_bet': 0.105, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'bb', 'monster', 'dry', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'bb', 'strong', 'dry', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'bb', 'good', 'dry', 'small')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'bb', 'marginal', 'dry', 'small')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'dry', 'small')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'monster', 'dry', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'bb', 'strong', 'dry', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'bb', 'good', 'dry', 'medium')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'bb', 'marginal', 'dry', 'medium')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'dry', 'medium')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'monster', 'dry', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(2, 'bb', 'strong', 'dry', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(2, 'bb', 'good', 'dry', 'large')": {'fold': 0.105, 'check_call': 0.421, 'small_bet': 0.263, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(2, 'bb', 'marginal', 'dry', 'large')": {'fold': 0.3, 'check_call': 0.45, 'small_bet': 0.15, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(2, 'bb', 'weak', 'dry', 'large')": {'fold': 0.55, 'check_call': 0.2, 'small_bet': 0.2, 'medium_bet': 0.05, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'monster', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(3, 'sb', 'strong', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(3, 'sb', 'good', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(3, 'sb', 'marginal', 'wet', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'wet', 'none')": {'fold': 0.278, 'check_call': 0.444, 'small_bet': 0.111, 'medium_bet': 0.111, 'large_bet': 0.056, 'jam': 0.0}, "(3, 'sb', 'monster', 'wet', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'sb', 'strong', 'wet', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'sb', 'good', 'wet', 'small')": {'fold': 0.25, 'check_call': 0.35, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(3, 'sb', 'marginal', 'wet', 'small')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'wet', 'small')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'monster', 'wet', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'sb', 'strong', 'wet', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'sb', 'good', 'wet', 'medium')": {'fold': 0.25, 'check_call': 0.35, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(3, 'sb', 'marginal', 'wet', 'medium')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'wet', 'medium')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'monster', 'wet', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'sb', 'strong', 'wet', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'sb', 'good', 'wet', 'large')": {'fold': 0.25, 'check_call': 0.35, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(3, 'sb', 'marginal', 'wet', 'large')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'wet', 'large')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'monster', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(3, 'sb', 'strong', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(3, 'sb', 'good', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(3, 'sb', 'marginal', 'medium', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'medium', 'none')": {'fold': 0.278, 'check_call': 0.444, 'small_bet': 0.111, 'medium_bet': 0.111, 'large_bet': 0.056, 'jam': 0.0}, "(3, 'sb', 'monster', 'medium', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'sb', 'strong', 'medium', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'sb', 'good', 'medium', 'small')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'sb', 'marginal', 'medium', 'small')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'medium', 'small')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'monster', 'medium', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'sb', 'strong', 'medium', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'sb', 'good', 'medium', 'medium')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'sb', 'marginal', 'medium', 'medium')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'medium', 'medium')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'monster', 'medium', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'sb', 'strong', 'medium', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'sb', 'good', 'medium', 'large')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'sb', 'marginal', 'medium', 'large')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'medium', 'large')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'monster', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(3, 'sb', 'strong', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(3, 'sb', 'good', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(3, 'sb', 'marginal', 'dry', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'dry', 'none')": {'fold': 0.25, 'check_call': 0.4, 'small_bet': 0.2, 'medium_bet': 0.1, 'large_bet': 0.05, 'jam': 0.0}, "(3, 'sb', 'monster', 'dry', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'sb', 'strong', 'dry', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'sb', 'good', 'dry', 'small')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'sb', 'marginal', 'dry', 'small')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'dry', 'small')": {'fold': 0.667, 'check_call': 0.19, 'small_bet': 0.095, 'medium_bet': 0.048, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'monster', 'dry', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'sb', 'strong', 'dry', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'sb', 'good', 'dry', 'medium')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'sb', 'marginal', 'dry', 'medium')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'dry', 'medium')": {'fold': 0.667, 'check_call': 0.19, 'small_bet': 0.095, 'medium_bet': 0.048, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'monster', 'dry', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'sb', 'strong', 'dry', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'sb', 'good', 'dry', 'large')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'sb', 'marginal', 'dry', 'large')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'sb', 'weak', 'dry', 'large')": {'fold': 0.667, 'check_call': 0.19, 'small_bet': 0.095, 'medium_bet': 0.048, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'monster', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(3, 'bb', 'strong', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(3, 'bb', 'good', 'wet', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(3, 'bb', 'marginal', 'wet', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'wet', 'none')": {'fold': 0.278, 'check_call': 0.444, 'small_bet': 0.111, 'medium_bet': 0.111, 'large_bet': 0.056, 'jam': 0.0}, "(3, 'bb', 'monster', 'wet', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'bb', 'strong', 'wet', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'bb', 'good', 'wet', 'small')": {'fold': 0.25, 'check_call': 0.35, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(3, 'bb', 'marginal', 'wet', 'small')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'wet', 'small')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'monster', 'wet', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'bb', 'strong', 'wet', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'bb', 'good', 'wet', 'medium')": {'fold': 0.25, 'check_call': 0.35, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(3, 'bb', 'marginal', 'wet', 'medium')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'wet', 'medium')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'monster', 'wet', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'bb', 'strong', 'wet', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'bb', 'good', 'wet', 'large')": {'fold': 0.25, 'check_call': 0.35, 'small_bet': 0.2, 'medium_bet': 0.15, 'large_bet': 0.05, 'jam': 0.0}, "(3, 'bb', 'marginal', 'wet', 'large')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'wet', 'large')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'monster', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(3, 'bb', 'strong', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(3, 'bb', 'good', 'medium', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(3, 'bb', 'marginal', 'medium', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'medium', 'none')": {'fold': 0.278, 'check_call': 0.444, 'small_bet': 0.111, 'medium_bet': 0.111, 'large_bet': 0.056, 'jam': 0.0}, "(3, 'bb', 'monster', 'medium', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'bb', 'strong', 'medium', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'bb', 'good', 'medium', 'small')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'bb', 'marginal', 'medium', 'small')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'medium', 'small')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'monster', 'medium', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'bb', 'strong', 'medium', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'bb', 'good', 'medium', 'medium')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'bb', 'marginal', 'medium', 'medium')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'medium', 'medium')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'monster', 'medium', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'bb', 'strong', 'medium', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'bb', 'good', 'medium', 'large')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'bb', 'marginal', 'medium', 'large')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'medium', 'large')": {'fold': 0.737, 'check_call': 0.211, 'small_bet': 0.0, 'medium_bet': 0.053, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'monster', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.1, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.35, 'jam': 0.15}, "(3, 'bb', 'strong', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.25, 'medium_bet': 0.35, 'large_bet': 0.2, 'jam': 0.05}, "(3, 'bb', 'good', 'dry', 'none')": {'fold': 0.0, 'check_call': 0.35, 'small_bet': 0.35, 'medium_bet': 0.2, 'large_bet': 0.1, 'jam': 0.0}, "(3, 'bb', 'marginal', 'dry', 'none')": {'fold': 0.05, 'check_call': 0.5, 'small_bet': 0.3, 'medium_bet': 0.15, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'dry', 'none')": {'fold': 0.25, 'check_call': 0.4, 'small_bet': 0.2, 'medium_bet': 0.1, 'large_bet': 0.05, 'jam': 0.0}, "(3, 'bb', 'monster', 'dry', 'small')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'bb', 'strong', 'dry', 'small')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'bb', 'good', 'dry', 'small')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'bb', 'marginal', 'dry', 'small')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'dry', 'small')": {'fold': 0.667, 'check_call': 0.19, 'small_bet': 0.095, 'medium_bet': 0.048, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'monster', 'dry', 'medium')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'bb', 'strong', 'dry', 'medium')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'bb', 'good', 'dry', 'medium')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'bb', 'marginal', 'dry', 'medium')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'dry', 'medium')": {'fold': 0.667, 'check_call': 0.19, 'small_bet': 0.095, 'medium_bet': 0.048, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'monster', 'dry', 'large')": {'fold': 0.0, 'check_call': 0.15, 'small_bet': 0.1, 'medium_bet': 0.3, 'large_bet': 0.3, 'jam': 0.15}, "(3, 'bb', 'strong', 'dry', 'large')": {'fold': 0.03, 'check_call': 0.25, 'small_bet': 0.2, 'medium_bet': 0.3, 'large_bet': 0.17, 'jam': 0.05}, "(3, 'bb', 'good', 'dry', 'large')": {'fold': 0.211, 'check_call': 0.368, 'small_bet': 0.211, 'medium_bet': 0.158, 'large_bet': 0.053, 'jam': 0.0}, "(3, 'bb', 'marginal', 'dry', 'large')": {'fold': 0.45, 'check_call': 0.35, 'small_bet': 0.1, 'medium_bet': 0.1, 'large_bet': 0.0, 'jam': 0.0}, "(3, 'bb', 'weak', 'dry', 'large')": {'fold': 0.667, 'check_call': 0.19, 'small_bet': 0.095, 'medium_bet': 0.048, 'large_bet': 0.0, 'jam': 0.0}}

# ── Constants ────────────────────────────────────────────────────────────────

RANKS = "23456789A"
SUITS = "dhs"
NUM_RANKS = 9
DECK_SIZE = 27
RANK_A = 8
RANK_9 = 7
RANK_8 = 6

FOLD = PokerEnv.ActionType.FOLD.value
RAISE = PokerEnv.ActionType.RAISE.value
CHECK = PokerEnv.ActionType.CHECK.value
CALL = PokerEnv.ActionType.CALL.value
DISCARD = PokerEnv.ActionType.DISCARD.value

_int_to_card = PokerEnv.int_to_card

# ── Card helpers ─────────────────────────────────────────────────────────────

def _rank(c):
    return c % NUM_RANKS

def _suit(c):
    return c // NUM_RANKS

def _same_suit(c1, c2):
    return _suit(c1) == _suit(c2)

def _rank_gap(c1, c2):
    return abs(_rank(c1) - _rank(c2))

def _effective_gap(c1, c2):
    g = _rank_gap(c1, c2)
    return g if g <= 4 else NUM_RANKS - g

def _is_connected(c1, c2):
    return _effective_gap(c1, c2) == 1

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ── Bucketing functions (inlined for runtime speed) ──────────────────────────

def _bucket_keep(keep2):
    r1, r2 = _rank(keep2[0]), _rank(keep2[1])
    s1, s2 = _suit(keep2[0]), _suit(keep2[1])
    is_pair = r1 == r2
    suited = s1 == s2
    gap = abs(r1 - r2)
    eg = gap if gap <= 4 else NUM_RANKS - gap

    if is_pair:
        if r1 in (RANK_A, RANK_9):
            return "premium_pair"
        if r1 >= RANK_8:
            return "medium_pair"
        return "low_pair"
    if suited:
        if eg <= 1:
            return "suited_connector"
        if eg <= 3:
            return "suited_semi"
        return "suited_gapper"
    if eg <= 1:
        return "offsuit_connector"
    return "offsuit_other"


def _bucket_flop_simple(community):
    if len(community) < 3:
        return "dry"
    suits = [_suit(c) for c in community]
    ranks = [_rank(c) for c in community]
    sc = Counter(suits)
    rc = Counter(ranks)
    max_sc = sc.most_common(1)[0][1]
    is_paired = rc.most_common(1)[0][1] >= 2

    sorted_r = sorted(set(ranks))
    conn = 0
    for i in range(len(sorted_r) - 1):
        if sorted_r[i + 1] - sorted_r[i] == 1:
            conn += 1
    if RANK_A in sorted_r and 0 in sorted_r:
        conn += 1

    score = 0
    if max_sc >= 3:
        score += 3
    elif max_sc >= 2:
        score += 1
    score += conn
    if is_paired:
        score += 1
    if score >= 3:
        return "wet"
    if score >= 1:
        return "medium"
    return "dry"


def _bucket_strength(equity):
    if equity > 0.80:
        return "monster"
    if equity > 0.65:
        return "strong"
    if equity > 0.50:
        return "good"
    if equity > 0.35:
        return "marginal"
    return "weak"


def _bucket_to_call(to_call, pot_size):
    if to_call <= 0:
        return "none"
    if pot_size <= 0:
        return "large"
    ratio = to_call / pot_size
    if ratio <= 0.15:
        return "small"
    if ratio <= 0.40:
        return "medium"
    return "large"


def _bucket_opp_discard(opp_discards):
    if len(opp_discards) < 3:
        return "unknown"
    ranks = [_rank(c) for c in opp_discards]
    suits = [_suit(c) for c in opp_discards]
    sc = Counter(suits)
    rc = Counter(ranks)
    has_pair = rc.most_common(1)[0][1] >= 2
    max_sc = sc.most_common(1)[0][1]
    has_ace = RANK_A in ranks

    sorted_r = sorted(ranks)
    conn = 0
    for i in range(len(sorted_r) - 1):
        if sorted_r[i + 1] - sorted_r[i] == 1:
            conn += 1
    if RANK_A in sorted_r and 0 in sorted_r:
        conn += 1

    if has_pair:
        return "discarded_pair"
    if max_sc >= 2:
        return "suited_cluster"
    if conn >= 2:
        return "connected_cluster"
    if has_ace:
        return "high_junk"
    if max(ranks) <= 5:
        return "low_junk"
    return "mixed_discard"


# ── Keep scoring (runtime version, lightweight MC) ───────────────────────────

def _structural_bonus(keep2):
    r1, r2 = _rank(keep2[0]), _rank(keep2[1])
    if r1 == r2:
        if r1 in (RANK_A, RANK_9):
            return 0.10
        if r1 >= RANK_8:
            return 0.05
        return 0.03
    eg = _effective_gap(keep2[0], keep2[1])
    if _same_suit(keep2[0], keep2[1]):
        if eg <= 1:
            return 0.12
        if eg <= 3:
            return 0.08
        return 0.06
    if eg <= 1:
        return 0.04
    return 0.0


def _board_interaction(keep2, community):
    if not community:
        return 0.0
    bonus = 0.0
    k_suits = [_suit(c) for c in keep2]
    k_ranks = [_rank(c) for c in keep2]
    b_suits = [_suit(c) for c in community]
    b_ranks = [_rank(c) for c in community]

    for s in set(k_suits):
        bm = sum(1 for bs in b_suits if bs == s)
        km = sum(1 for ks in k_suits if ks == s)
        if bm >= 2 and km >= 1:
            bonus += 0.08
            if km >= 2:
                bonus += 0.04
            break

    all_r = sorted(set(k_ranks + b_ranks))
    mc = 1
    best = 1
    for i in range(1, len(all_r)):
        if all_r[i] - all_r[i - 1] == 1:
            mc += 1
            best = max(best, mc)
        else:
            mc = 1
    if RANK_A in all_r and 0 in all_r:
        best = max(best, 2)
    if best >= 4:
        bonus += 0.06

    for kr in k_ranks:
        if kr in b_ranks:
            bonus += 0.04
            break
    return bonus


def _inference_bonus(keep2, opp_discards, community):
    if not opp_discards or len(opp_discards) < 3:
        return 0.0
    bonus = 0.0
    opp_b = _bucket_opp_discard(opp_discards)
    opp_d_suits = set(_suit(c) for c in opp_discards)
    k_suits = [_suit(c) for c in keep2]
    b_suits = [_suit(c) for c in community] if community else []

    if opp_b == "low_junk":
        if max(_rank(c) for c in keep2) <= 5:
            bonus -= 0.04

    if opp_b == "suited_cluster":
        threat_suits = set(range(3)) - opp_d_suits
        blocking = sum(1 for ks in k_suits if ks in threat_suits)
        bonus += 0.02 * blocking
        for ts in threat_suits:
            if sum(1 for bs in b_suits if bs == ts) >= 2:
                bonus += 0.03 * sum(1 for ks in k_suits if ks == ts)
                break

    if opp_b == "discarded_pair":
        bonus += 0.02
    return bonus


def _board_paired_penalty(my_cards, community):
    """Return equity penalty when board is paired and we don't hold trips+."""
    if len(community) < 3:
        return 0.0
    b_ranks = [_rank(c) for c in community]
    rc = Counter(b_ranks)
    board_pair_rank = None
    for r, cnt in rc.items():
        if cnt >= 2:
            board_pair_rank = r
            break
    if board_pair_rank is None:
        return 0.0
    my_ranks = [_rank(c) for c in my_cards]
    if my_ranks.count(board_pair_rank) >= 1:
        return 0.0
    if my_ranks[0] == my_ranks[1] and my_ranks[0] in (RANK_A, RANK_9):
        return 0.0
    return 0.08


# ── PlayerAgent ──────────────────────────────────────────────────────────────

class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self._env = PokerEnv()
        self.evaluator = self._env.evaluator

        self.hand_number = 0
        self._last_street = -1
        self.opp_stats = {
            "fold_to_raise": [0, 0],  # [folds, total]
            "aggression": [0, 0],     # [raises, opportunities]
            "river_calldown": [0, 0],
        }
        self._prev_action = None
        self._prev_opp_bet = 0
        self._preflop_equity_cache = (None, None)  # (hand_number, equity)

    def __name__(self):
        return "Libratus"

    # ── Seeded RNG ──────────────────────────────────────────────────────

    def _seed_rng(self, my_cards, street):
        seed_val = hash((self.hand_number, street, tuple(sorted(my_cards))))
        self._rng = random.Random(seed_val)

    # ── MC equity ───────────────────────────────────────────────────────

    def _mc_equity(self, my2, community, dead, num_sims=200):
        known = set(my2) | set(community) | dead
        remaining = [c for c in range(DECK_SIZE) if c not in known]
        board_needed = 5 - len(community)
        sample_needed = 2 + board_needed

        if sample_needed > len(remaining):
            return 0.5

        wins = 0.0
        total = 0
        for _ in range(num_sims):
            sample = self._rng.sample(remaining, sample_needed)
            opp = sample[:2]
            full_board = list(community) + sample[2:]
            my_hand = [_int_to_card(c) for c in my2]
            opp_hand = [_int_to_card(c) for c in opp]
            board = [_int_to_card(c) for c in full_board]
            mr = self.evaluator.evaluate(my_hand, board)
            orank = self.evaluator.evaluate(opp_hand, board)
            if mr < orank:
                wins += 1.0
            elif mr == orank:
                wins += 0.5
            total += 1
        return wins / total if total > 0 else 0.5

    # ── Discard: choose best keep ───────────────────────────────────────

    def _choose_keep(self, my_cards, community, opp_discards):
        best_score = -999.0
        best_ij = (0, 1)
        candidates = []

        for i, j in combinations(range(len(my_cards)), 2):
            keep = [my_cards[i], my_cards[j]]
            toss = [my_cards[k] for k in range(len(my_cards)) if k not in (i, j)]
            dead = set(toss)
            if opp_discards:
                dead |= set(opp_discards)

            eq = self._mc_equity(keep, community, dead, num_sims=60)
            struct = _structural_bonus(keep)
            board = _board_interaction(keep, community)
            infer = _inference_bonus(keep, opp_discards, community)

            score = 3.0 * eq + 1.5 * struct + 1.0 * board + 0.5 * infer
            candidates.append((i, j, score, eq))
            if score > best_score:
                best_score = score
                best_ij = (i, j)

        # Near-tie randomization
        ties = [(i, j) for i, j, sc, _ in candidates if best_score - sc < 0.06]
        if len(ties) > 1:
            best_ij = self._rng.choice(ties)

        return best_ij

    # ── Betting: policy table lookup + mixed strategy ───────────────────

    def _choose_bet(self, street, my_cards, community, opp_discards, my_discards,
                    valid, min_raise, max_raise, my_bet, opp_bet, pot_size, blind_pos):
        to_call = max(0, opp_bet - my_bet)
        position = "sb" if blind_pos == 0 else "bb"

        dead = set()
        if my_discards:
            dead |= set(my_discards)
        if opp_discards:
            dead |= set(opp_discards)

        # Compute equity
        if len(my_cards) == 2 and len(community) >= 3:
            equity = self._mc_equity(my_cards, community, dead, num_sims=80)
        elif len(my_cards) == 5 and street == 0:
            cached_hn, cached_eq = self._preflop_equity_cache
            if cached_hn == self.hand_number:
                equity = cached_eq
            else:
                best_eq = 0.0
                for i, j in combinations(range(5), 2):
                    keep = [my_cards[i], my_cards[j]]
                    toss = [my_cards[k] for k in range(5) if k not in (i, j)]
                    eq = self._mc_equity(keep, [], set(toss) | dead, num_sims=10)
                    best_eq = max(best_eq, eq)
                equity = best_eq
                self._preflop_equity_cache = (self.hand_number, equity)
        else:
            equity = 0.45

        # Board-paired penalty
        if len(community) >= 3 and len(my_cards) == 2:
            equity -= _board_paired_penalty(my_cards, community)

        # Adjust equity based on opponent discard inference (all 6 buckets)
        opp_b = "unknown"
        if opp_discards and len(opp_discards) >= 3:
            opp_b = _bucket_opp_discard(opp_discards)
            if opp_b == "low_junk":
                equity -= 0.04
            elif opp_b == "suited_cluster":
                equity -= 0.03
            elif opp_b == "connected_cluster":
                equity -= 0.02
            elif opp_b == "high_junk":
                equity += 0.02
            elif opp_b == "discarded_pair":
                equity += 0.03

        # Range-vs-range equity from POSTERIOR + MATCHUPS tables
        if opp_b != "unknown" and len(my_cards) == 2 and street >= 1:
            board_b_for_post = _bucket_flop_simple(community)
            post_key = f"{opp_b}|{board_b_for_post}"
            opp_dist = POSTERIOR.get(post_key, {})
            my_kb = _bucket_keep(my_cards)
            if opp_dist:
                matchup_eq = 0.0
                weight_sum = 0.0
                for opp_kb, prob in opp_dist.items():
                    mk = MATCHUPS.get(f"{my_kb}|{opp_kb}", 0.5)
                    matchup_eq += prob * mk
                    weight_sum += prob
                if weight_sum > 0:
                    matchup_eq /= weight_sum
                    equity = 0.7 * equity + 0.3 * matchup_eq

        strength = _bucket_strength(equity)
        board_b = _bucket_flop_simple(community) if street >= 1 else "any"
        tc_b = _bucket_to_call(to_call, pot_size)

        # Look up policy
        key = str((street, position, strength, board_b, tc_b))
        policy = POLICY.get(key)
        if not policy:
            # Fallback: try with "medium" board and "none" to_call
            key2 = str((street, position, strength, "medium", "none"))
            policy = POLICY.get(key2, {
                "fold": 0.20, "check_call": 0.40, "small_bet": 0.25,
                "medium_bet": 0.10, "large_bet": 0.05, "jam": 0.0
            })

        # Opponent-model adjustments
        policy = dict(policy)
        opp_fold_rate = self._opp_fold_rate()
        opp_agg = self._opp_aggression()
        if opp_fold_rate > 0.5:
            policy["small_bet"] = policy.get("small_bet", 0) + 0.05
            policy["fold"] = max(0, policy.get("fold", 0) - 0.05)
        elif opp_fold_rate < 0.2:
            policy["small_bet"] = max(0, policy.get("small_bet", 0) - 0.04)
            policy["check_call"] = policy.get("check_call", 0) + 0.04
        if opp_agg > 0.55 and strength in ("marginal", "weak"):
            policy["check_call"] = max(0, policy.get("check_call", 0) - 0.05)
            policy["fold"] = policy.get("fold", 0) + 0.05

        # Sample action from mixed policy
        action = self._sample_action(policy)

        # Convert to game action
        return self._action_to_tuple(
            action, valid, min_raise, max_raise, pot_size, to_call, equity
        )

    def _sample_action(self, policy):
        r = self._rng.random()
        cumul = 0.0
        for act, prob in policy.items():
            cumul += prob
            if r < cumul:
                return act
        return "check_call"

    def _action_to_tuple(self, action, valid, min_raise, max_raise, pot_size, to_call, equity):
        if action == "fold":
            if valid[FOLD]:
                return (FOLD, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        if action == "check_call":
            if to_call > 0 and valid[CALL]:
                return (CALL, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            return (CHECK, 0, 0, 0)

        if action == "jam":
            if valid[RAISE] and max_raise > 0:
                return (RAISE, max_raise, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        # Bet sizing
        pot_ref = max(pot_size, 1)
        if action == "small_bet":
            frac = self._rng.uniform(0.30, 0.45)
        elif action == "medium_bet":
            frac = self._rng.uniform(0.55, 0.75)
        elif action == "large_bet":
            frac = self._rng.uniform(0.85, 1.10)
        else:
            frac = 0.50

        raw_amount = int(pot_ref * frac)
        amount = _clamp(raw_amount, min_raise, max_raise)

        if valid[RAISE] and max_raise >= min_raise:
            return (RAISE, amount, 0, 0)
        if valid[CALL]:
            return (CALL, 0, 0, 0)
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    # ── Opponent model ──────────────────────────────────────────────────

    def _opp_fold_rate(self):
        folds, total = self.opp_stats["fold_to_raise"]
        if total < 10:
            return 0.35
        return folds / total

    def _opp_aggression(self):
        raises, opps = self.opp_stats["aggression"]
        if opps < 10:
            return 0.35
        return raises / opps

    def _update_opp_stats(self, observation, reward, terminated):
        opp_bet = observation["opp_bet"]
        street = observation["street"]

        if self._prev_action == RAISE:
            self.opp_stats["fold_to_raise"][1] += 1
            if terminated and reward > 0:
                self.opp_stats["fold_to_raise"][0] += 1

        if opp_bet > self._prev_opp_bet and opp_bet > 2:
            self.opp_stats["aggression"][0] += 1
            self.opp_stats["aggression"][1] += 1
        elif street >= 1:
            self.opp_stats["aggression"][1] += 1

        self._prev_opp_bet = opp_bet

    # ── Main act function ───────────────────────────────────────────────

    def act(self, observation, reward, terminated, truncated, info):
        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards = [c for c in observation["my_discarded_cards"] if c != -1]
        valid = observation["valid_actions"]
        street = observation["street"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        pot_size = observation.get("pot_size", my_bet + opp_bet)
        blind_pos = observation.get("blind_position", 0)

        # Track hand number via street transitions
        if street == 0 and self._last_street != 0:
            self.hand_number += 1
        self._last_street = street

        self._update_opp_stats(observation, reward, terminated)
        self._seed_rng(my_cards, street)

        # ── Discard phase ───────────────────────────────────────────
        if valid[DISCARD]:
            i, j = self._choose_keep(my_cards, community, opp_discards)
            self._prev_action = DISCARD
            return (DISCARD, 0, i, j)

        # ── Betting phase ───────────────────────────────────────────
        result = self._choose_bet(
            street, my_cards, community, opp_discards, my_discards,
            valid, min_raise, max_raise, my_bet, opp_bet, pot_size, blind_pos
        )
        self._prev_action = result[0]
        return result
