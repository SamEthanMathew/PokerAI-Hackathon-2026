'use client';

import { motion, AnimatePresence } from 'framer-motion';
import type { RecordedHand } from '@/lib/engine/types';

interface HandHistoryProps {
  hands: RecordedHand[];
  currentHand: number;
}

export default function HandHistory({ hands, currentHand }: HandHistoryProps) {
  const reversed = [...hands].reverse();

  return (
    <div className="flex flex-col h-full">
      <div className="text-xs text-gray-400 font-medium mb-2">
        Hand {currentHand} / 1000
      </div>
      <div className="flex-1 overflow-y-auto space-y-1 pr-1">
        <AnimatePresence>
          {reversed.map((hand) => (
            <motion.div
              key={hand.hand_number}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-xs bg-gray-800/60 rounded px-2 py-1.5"
            >
              <div className="flex justify-between items-center">
                <span className="text-gray-400">#{hand.hand_number}</span>
                <span className="text-gray-500">{hand.position}</span>
                <span className={hand.result.won
                  ? 'text-green-400 font-bold'
                  : hand.result.chips_won_lost === 0
                  ? 'text-gray-400'
                  : 'text-red-400 font-bold'}>
                  {hand.result.won ? '+' : ''}{hand.result.chips_won_lost}
                </span>
              </div>
              {hand.showdown && hand.my_final_hand && (
                <div className="text-gray-500 mt-0.5">{hand.my_final_hand}</div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
        {hands.length === 0 && (
          <div className="text-gray-600 text-xs text-center mt-4">No hands yet</div>
        )}
      </div>
    </div>
  );
}
