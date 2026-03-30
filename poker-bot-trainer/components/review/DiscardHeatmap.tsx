'use client';

import { useMemo } from 'react';
import type { GameState } from '@/lib/engine/types';
import { intToRank, RANKS } from '@/lib/engine/deck';

interface DiscardHeatmapProps {
  hands: GameState[];
}

export default function DiscardHeatmap({ hands }: DiscardHeatmapProps) {
  // Count how often each rank-pair is kept during discard
  const heatmap = useMemo(() => {
    const matrix: number[][] = Array.from({ length: 9 }, () => Array(9).fill(0));
    let total = 0;

    for (const h of hands) {
      if (h.action_taken.type !== 'discard' || !h.action_taken.kept_cards) continue;
      const [k1, k2] = h.action_taken.kept_cards;
      const cards = h.my_cards;
      if (!cards[k1] || !cards[k2]) continue;
      const r1 = intToRank(cards[k1]);
      const r2 = intToRank(cards[k2]);
      // Symmetric
      matrix[r1][r2]++;
      if (r1 !== r2) matrix[r2][r1]++;
      total++;
    }

    return { matrix, total };
  }, [hands]);

  const maxVal = Math.max(...heatmap.matrix.flat(), 1);
  const rankLabels = RANKS.split('');

  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-sm text-gray-300 font-medium">
        Kept Card Pair Heatmap (rank × rank, from {heatmap.total} discards)
      </h3>

      {heatmap.total === 0 ? (
        <div className="text-gray-500 text-sm">No discard decisions found in this session</div>
      ) : (
        <div className="overflow-auto">
          <div style={{ display: 'inline-block' }}>
            {/* Column headers */}
            <div className="flex">
              <div className="w-8 h-7" />
              {rankLabels.map(r => (
                <div key={r} className="w-9 h-7 flex items-center justify-center text-xs text-gray-400 font-mono">
                  {r}
                </div>
              ))}
            </div>

            {/* Rows */}
            {rankLabels.map((rowRank, ri) => (
              <div key={rowRank} className="flex">
                <div className="w-8 h-9 flex items-center justify-center text-xs text-gray-400 font-mono">
                  {rowRank}
                </div>
                {rankLabels.map((colRank, ci) => {
                  const val = heatmap.matrix[ri][ci];
                  const intensity = val / maxVal;
                  const bg = intensity > 0
                    ? `rgba(234, 179, 8, ${0.1 + intensity * 0.8})`
                    : '#1f2937';
                  return (
                    <div
                      key={colRank}
                      title={`${rowRank}-${colRank}: ${val} times`}
                      className="w-9 h-9 flex items-center justify-center text-xs border border-gray-800/50 cursor-help"
                      style={{ backgroundColor: bg }}
                    >
                      {val > 0 && (
                        <span style={{ color: intensity > 0.5 ? '#111' : '#eee', fontSize: 9 }}>
                          {val}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="text-xs text-gray-500">
        Darker yellow = more frequently kept. Hover for exact count. Diagonal = pairs.
      </p>
    </div>
  );
}
