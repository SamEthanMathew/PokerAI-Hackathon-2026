'use client';

import { motion } from 'framer-motion';
import type { CardInt } from '@/lib/engine/types';
import Card from './Card';

interface DiscardSelectorProps {
  cards: CardInt[];
  kept: (number | null)[];
  onToggleKeep: (cardIndex: number) => void;
  onConfirm: () => void;
  disabled?: boolean;
}

export default function DiscardSelector({
  cards,
  kept,
  onToggleKeep,
  onConfirm,
  disabled = false,
}: DiscardSelectorProps) {
  const keptSet = new Set(kept.filter(k => k !== null) as number[]);
  const numKept = keptSet.size;
  const canConfirm = numKept === 2 && !disabled;

  return (
    <motion.div
      className="flex flex-col items-center gap-4"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <p className="text-yellow-300 text-sm font-medium text-center px-4">
        Select 2 cards to keep. The other 3 will be discarded and revealed to your opponent.
      </p>

      <div className="flex gap-2">
        {cards.map((card, idx) => {
          const isKept = keptSet.has(idx);
          const isDimmed = numKept === 2 && !isKept;
          return (
            <div key={idx} className="flex flex-col items-center gap-1">
              <Card
                card={card}
                size="lg"
                selected={isKept}
                dimmed={isDimmed}
                onClick={disabled ? undefined : () => onToggleKeep(idx)}
                animate={false}
              />
              <span className={`text-xs ${isKept ? 'text-green-400 font-bold' : isDimmed ? 'text-gray-500' : 'text-gray-300'}`}>
                {isKept ? 'KEEP' : isDimmed ? 'DISCARD' : ''}
              </span>
            </div>
          );
        })}
      </div>

      <div className="flex items-center gap-3">
        <span className="text-gray-400 text-xs">
          {numKept}/2 selected
        </span>
        <motion.button
          onClick={canConfirm ? onConfirm : undefined}
          disabled={!canConfirm}
          className={`px-6 py-2 rounded-lg font-bold text-sm transition-all ${
            canConfirm
              ? 'bg-yellow-400 text-gray-900 hover:bg-yellow-300 cursor-pointer shadow-lg'
              : 'bg-gray-600 text-gray-400 cursor-not-allowed'
          }`}
          whileHover={canConfirm ? { scale: 1.05 } : {}}
          whileTap={canConfirm ? { scale: 0.97 } : {}}
        >
          Confirm Keep
        </motion.button>
      </div>
    </motion.div>
  );
}
