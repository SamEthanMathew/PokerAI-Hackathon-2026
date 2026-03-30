'use client';

import type { CardInt } from '@/lib/engine/types';
import Card from './Card';
import DiscardSelector from './DiscardSelector';

interface PlayerHandProps {
  cards: CardInt[];
  discarded: CardInt[];
  bet: number;
  position: string;
  isDiscardPhase: boolean;
  keptIndices: (number | null)[];
  onToggleKeep: (idx: number) => void;
  onConfirmDiscard: () => void;
  stack: number;
}

export default function PlayerHand({
  cards,
  discarded,
  bet,
  position,
  isDiscardPhase,
  keptIndices,
  onToggleKeep,
  onConfirmDiscard,
  stack,
}: PlayerHandProps) {
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="flex items-center gap-2">
        <span className="text-white text-sm font-medium">You</span>
        <span className="text-xs bg-blue-900/60 text-blue-300 px-2 py-0.5 rounded-full">
          {position}
        </span>
        <span className="text-xs text-gray-400">Stack: {stack}</span>
        {bet > 0 && (
          <span className="text-xs bg-yellow-900/60 text-yellow-300 px-2 py-0.5 rounded-full">
            Bet: {bet}
          </span>
        )}
      </div>

      {isDiscardPhase ? (
        <DiscardSelector
          cards={cards}
          kept={keptIndices}
          onToggleKeep={onToggleKeep}
          onConfirm={onConfirmDiscard}
        />
      ) : (
        <div className="flex flex-col items-center gap-2">
          <div className="flex gap-2">
            {cards.map((card, i) => (
              <Card key={i} card={card >= 0 ? card : 'back'} size="lg" />
            ))}
          </div>

          {discarded.length > 0 && (
            <div className="flex flex-col items-center gap-1">
              <span className="text-xs text-gray-500">Discarded</span>
              <div className="flex gap-1">
                {discarded.map((card, i) => (
                  <Card key={i} card={card >= 0 ? card : 'back'} size="sm" className="opacity-50" />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
