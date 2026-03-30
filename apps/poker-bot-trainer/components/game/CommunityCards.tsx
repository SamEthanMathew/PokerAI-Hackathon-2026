'use client';

import { motion, AnimatePresence } from 'framer-motion';
import type { CardInt } from '@/lib/engine/types';
import Card from './Card';

interface CommunityCardsProps {
  cards: CardInt[];
  potSize: number;
  street: number;
  handName?: string;
}

const STREET_NAMES = ['Pre-flop', 'Flop', 'Turn', 'River'];

export default function CommunityCards({ cards, potSize, street, handName }: CommunityCardsProps) {
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-400 bg-gray-800/60 px-2 py-0.5 rounded-full">
          {STREET_NAMES[street] ?? 'River'}
        </span>
        <span className="text-white font-bold text-lg">
          Pot: <span className="text-yellow-300">{potSize}</span>
        </span>
        {handName && (
          <span className="text-xs text-green-400 bg-gray-800/60 px-2 py-0.5 rounded-full">
            {handName}
          </span>
        )}
      </div>

      <div className="flex gap-2">
        {Array.from({ length: 5 }).map((_, i) => {
          const card = cards[i];
          return (
            <AnimatePresence key={i} mode="wait">
              {card !== undefined && card >= 0 ? (
                <motion.div
                  key={`card-${card}`}
                  initial={{ opacity: 0, rotateY: 90, scale: 0.8 }}
                  animate={{ opacity: 1, rotateY: 0, scale: 1 }}
                  transition={{ duration: 0.35, delay: i < 3 ? i * 0.08 : 0 }}
                >
                  <Card card={card} size="md" animate={false} />
                </motion.div>
              ) : (
                <motion.div
                  key={`empty-${i}`}
                  className="w-14 h-20 rounded-md border-2 border-dashed border-gray-600/50"
                />
              )}
            </AnimatePresence>
          );
        })}
      </div>
    </div>
  );
}
