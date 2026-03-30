'use client';

import { motion, AnimatePresence } from 'framer-motion';
import type { CardInt } from '@/lib/engine/types';
import Card from './Card';

interface OpponentAreaProps {
  cards: CardInt[];
  discarded: CardInt[];
  bet: number;
  lastAction: string | null;
  isThinking: boolean;
  isShowdown: boolean;
  position: string;
}

export default function OpponentArea({
  cards,
  discarded,
  bet,
  lastAction,
  isThinking,
  isShowdown,
  position,
}: OpponentAreaProps) {
  return (
    <div className="flex flex-col items-center gap-2">
      {/* Opponent label + bet */}
      <div className="flex items-center gap-2">
        <span className="text-gray-300 text-sm font-medium">Bot Opponent</span>
        <span className="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full">
          {position}
        </span>
        {bet > 0 && (
          <span className="text-xs bg-yellow-900/60 text-yellow-300 px-2 py-0.5 rounded-full">
            Bet: {bet}
          </span>
        )}
      </div>

      {/* Last action label */}
      <AnimatePresence>
        {isThinking && (
          <motion.div
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="text-xs text-gray-400 animate-pulse"
          >
            Thinking...
          </motion.div>
        )}
        {lastAction && !isThinking && (
          <motion.div
            key={lastAction}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="text-xs font-bold text-white bg-gray-700 px-3 py-1 rounded-full"
          >
            {lastAction}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex flex-col items-center gap-2">
        {/* Hole cards — face down unless showdown */}
        <div className="flex gap-2">
          {cards.length > 0 ? (
            cards.map((card, i) => (
              <Card
                key={i}
                card={isShowdown && card >= 0 ? card : 'back'}
                size="md"
                animate={isShowdown}
              />
            ))
          ) : (
            [0, 1].map(i => <Card key={i} card="back" size="md" />)
          )}
        </div>

        {/* Discarded cards (revealed) */}
        {discarded.length > 0 && (
          <div className="flex flex-col items-center gap-1">
            <span className="text-xs text-gray-500">Discarded</span>
            <div className="flex gap-1">
              {discarded.map((card, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                >
                  <Card card={card >= 0 ? card : 'back'} size="sm" />
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
