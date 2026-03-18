'use client';

import { useState, useMemo } from 'react';
import type { GameState } from '@/lib/engine/types';

interface SessionTableProps {
  hands: GameState[];
}

const STREET_NAMES = ['Preflop', 'Flop', 'Turn', 'River'];
const ACTION_COLORS: Record<string, string> = {
  fold: 'text-red-400',
  raise: 'text-yellow-400',
  check: 'text-gray-300',
  call: 'text-blue-400',
  discard: 'text-purple-400',
};

export default function SessionTable({ hands }: SessionTableProps) {
  const [streetFilter, setStreetFilter] = useState<number | 'all'>('all');
  const [actionFilter, setActionFilter] = useState<string | 'all'>('all');
  const [outcomeFilter, setOutcomeFilter] = useState<'all' | 'won' | 'lost'>('all');

  const filtered = useMemo(() => {
    return hands.filter(h => {
      if (streetFilter !== 'all' && h.street !== streetFilter) return false;
      if (actionFilter !== 'all' && h.action_taken.type !== actionFilter) return false;
      if (outcomeFilter === 'won' && !h.hand_result?.won) return false;
      if (outcomeFilter === 'lost' && h.hand_result?.won) return false;
      return true;
    });
  }, [hands, streetFilter, actionFilter, outcomeFilter]);

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Filters */}
      <div className="flex gap-3 flex-wrap">
        <Select
          label="Street"
          value={String(streetFilter)}
          onChange={v => setStreetFilter(v === 'all' ? 'all' : Number(v) as 0|1|2|3)}
          options={[
            { label: 'All Streets', value: 'all' },
            { label: 'Preflop', value: '0' },
            { label: 'Flop', value: '1' },
            { label: 'Turn', value: '2' },
            { label: 'River', value: '3' },
          ]}
        />
        <Select
          label="Action"
          value={actionFilter}
          onChange={v => setActionFilter(v)}
          options={[
            { label: 'All Actions', value: 'all' },
            { label: 'Fold', value: 'fold' },
            { label: 'Check', value: 'check' },
            { label: 'Call', value: 'call' },
            { label: 'Raise', value: 'raise' },
            { label: 'Discard', value: 'discard' },
          ]}
        />
        <Select
          label="Outcome"
          value={outcomeFilter}
          onChange={v => setOutcomeFilter(v as 'all'|'won'|'lost')}
          options={[
            { label: 'All Outcomes', value: 'all' },
            { label: 'Won', value: 'won' },
            { label: 'Lost', value: 'lost' },
          ]}
        />
        <span className="text-xs text-gray-500 self-center">{filtered.length} decisions</span>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto rounded-xl border border-gray-700/50">
        <table className="w-full text-xs">
          <thead className="bg-gray-800 sticky top-0">
            <tr>
              {['Hand', 'Street', 'Pos', 'Action', 'Pot', 'My Bet', 'Opp Bet', 'Result'].map(h => (
                <th key={h} className="text-left text-gray-400 font-medium px-3 py-2">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((hand, i) => {
              const result = hand.hand_result;
              return (
                <tr
                  key={i}
                  className="border-t border-gray-800 hover:bg-gray-800/50 transition-colors"
                >
                  <td className="px-3 py-1.5 text-gray-400">#{hand.hand_number}</td>
                  <td className="px-3 py-1.5 text-gray-300">{STREET_NAMES[hand.street]}</td>
                  <td className="px-3 py-1.5 text-gray-400">{hand.position}</td>
                  <td className={`px-3 py-1.5 font-medium ${ACTION_COLORS[hand.action_taken.type] ?? 'text-gray-300'}`}>
                    {hand.action_taken.type.toUpperCase()}
                    {hand.action_taken.raise_amount ? ` ${hand.action_taken.raise_amount}` : ''}
                  </td>
                  <td className="px-3 py-1.5 text-gray-300">{hand.pot_size}</td>
                  <td className="px-3 py-1.5 text-gray-400">{hand.my_bet}</td>
                  <td className="px-3 py-1.5 text-gray-400">{hand.opp_bet}</td>
                  <td className={`px-3 py-1.5 font-bold ${
                    result?.won ? 'text-green-400' :
                    result?.chips_won_lost === 0 ? 'text-gray-400' :
                    'text-red-400'
                  }`}>
                    {result
                      ? `${result.chips_won_lost >= 0 ? '+' : ''}${result.chips_won_lost}`
                      : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div className="text-center text-gray-600 py-8">No decisions match the filters</div>
        )}
      </div>
    </div>
  );
}

function Select({ label, value, onChange, options }: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { label: string; value: string }[];
}) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="bg-gray-800 text-gray-300 text-xs rounded-lg px-2 py-1.5 border border-gray-700 focus:outline-none focus:border-yellow-400"
    >
      {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}
