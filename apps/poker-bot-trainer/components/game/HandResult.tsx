'use client';

import { motion } from 'framer-motion';
import type { HandResult as HandResultType, EngineState } from '@/lib/engine/types';
import Card from './Card';

interface HandResultProps {
  result: HandResultType;
  engineState: EngineState;
  humanIndex: 0 | 1;
  onDismiss: () => void;
}

export default function HandResult({ result, engineState, humanIndex, onDismiss }: HandResultProps) {
  const oppIndex = (1 - humanIndex) as 0 | 1;
  const myCards = engineState.players[humanIndex].cards;
  const oppCards = engineState.players[oppIndex].cards;
  const community = engineState.all_community;

  return (
    <motion.div
      className="fixed inset-0 flex items-center justify-center z-50 bg-black/70 backdrop-blur-sm"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      onClick={onDismiss}
    >
      <motion.div
        className="bg-gray-900 border border-gray-700 rounded-2xl p-6 max-w-sm w-full mx-4 shadow-2xl"
        initial={{ scale: 0.8, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        onClick={e => e.stopPropagation()}
      >
        {/* Result banner */}
        <div className={`text-center text-2xl font-bold mb-4 ${result.won ? 'text-green-400' : result.chips_won_lost === 0 ? 'text-gray-300' : 'text-red-400'}`}>
          {result.won ? '🏆 You Win!' : result.chips_won_lost === 0 ? '🤝 Tie' : '💀 You Lose'}
        </div>

        <div className={`text-center text-lg font-bold mb-4 ${result.chips_won_lost >= 0 ? 'text-green-300' : 'text-red-300'}`}>
          {result.chips_won_lost >= 0 ? '+' : ''}{result.chips_won_lost} chips
        </div>

        {/* Cards display */}
        {result.showdown && (
          <div className="space-y-3 mb-4">
            <div>
              <div className="text-xs text-gray-400 mb-1">Your hand — {result.my_final_hand}</div>
              <div className="flex gap-1">
                {myCards.map((c, i) => <Card key={i} card={c} size="sm" />)}
              </div>
            </div>

            <div>
              <div className="text-xs text-gray-400 mb-1">Board</div>
              <div className="flex gap-1">
                {community.map((c, i) => <Card key={i} card={c} size="sm" />)}
              </div>
            </div>

            <div>
              <div className="text-xs text-gray-400 mb-1">Opponent — {result.opp_final_hand}</div>
              <div className="flex gap-1">
                {oppCards.map((c, i) => <Card key={i} card={c} size="sm" />)}
              </div>
            </div>
          </div>
        )}

        <button
          onClick={onDismiss}
          className="w-full py-2 bg-yellow-400 text-gray-900 font-bold rounded-lg hover:bg-yellow-300 transition-colors"
        >
          Next Hand →
        </button>
      </motion.div>
    </motion.div>
  );
}
