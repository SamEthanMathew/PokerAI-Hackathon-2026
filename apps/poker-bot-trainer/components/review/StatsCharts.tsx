'use client';

import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Legend,
} from 'recharts';
import type { GameState, SessionStats } from '@/lib/engine/types';

interface StatsChartsProps {
  hands: GameState[];
  stats: SessionStats;
}

export default function StatsCharts({ hands, stats }: StatsChartsProps) {
  // Action distribution by street
  const actionByStreet = useMemo(() => {
    const streets = ['Preflop', 'Flop', 'Turn', 'River'];
    return streets.map((name, si) => {
      const streetHands = hands.filter(h => h.street === si);
      const total = streetHands.length || 1;
      return {
        name,
        Fold: Math.round(streetHands.filter(h => h.action_taken.type === 'fold').length / total * 100),
        Check: Math.round(streetHands.filter(h => h.action_taken.type === 'check').length / total * 100),
        Call: Math.round(streetHands.filter(h => h.action_taken.type === 'call').length / total * 100),
        Raise: Math.round(streetHands.filter(h => h.action_taken.type === 'raise').length / total * 100),
      };
    });
  }, [hands]);

  // Cumulative net chips over hands
  const chipProgression = useMemo(() => {
    const handResults = new Map<number, number>();
    for (const h of hands) {
      if (h.hand_result) {
        handResults.set(h.hand_number, h.hand_result.chips_won_lost);
      }
    }
    let cumulative = 0;
    const points: { hand: number; chips: number }[] = [];
    const sortedHands = [...handResults.keys()].sort((a, b) => a - b);
    for (const num of sortedHands) {
      cumulative += handResults.get(num) ?? 0;
      if (num % 10 === 0 || sortedHands.length <= 50) {
        points.push({ hand: num, chips: cumulative });
      }
    }
    return points;
  }, [hands]);

  const statCards = [
    { label: 'Hands Won', value: stats.hands_won },
    { label: 'Hands Lost', value: stats.hands_lost },
    { label: 'VPIP', value: `${Math.round(stats.vpip * 100)}%` },
    { label: 'PFR', value: `${Math.round(stats.pfr * 100)}%` },
    { label: 'Agg Factor', value: stats.aggression_factor.toFixed(2) },
    { label: 'Showdown %', value: `${Math.round(stats.went_to_showdown * 100)}%` },
  ];

  return (
    <div className="flex flex-col gap-6">
      {/* Stat cards */}
      <div className="grid grid-cols-6 gap-3">
        {statCards.map(({ label, value }) => (
          <div key={label} className="bg-gray-800/60 rounded-lg p-3 text-center">
            <div className="text-gray-400 text-xs mb-1">{label}</div>
            <div className="text-white font-bold">{value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Action distribution by street */}
        <div className="bg-gray-800/50 rounded-xl p-4">
          <h3 className="text-sm text-gray-300 mb-3 font-medium">Action Distribution by Street</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={actionByStreet}>
              <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} unit="%" />
              <Tooltip
                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="Fold" fill="#ef4444" stackId="a" />
              <Bar dataKey="Check" fill="#6b7280" stackId="a" />
              <Bar dataKey="Call" fill="#3b82f6" stackId="a" />
              <Bar dataKey="Raise" fill="#eab308" stackId="a" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Chip progression */}
        <div className="bg-gray-800/50 rounded-xl p-4">
          <h3 className="text-sm text-gray-300 mb-3 font-medium">Cumulative Chips</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chipProgression}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="hand" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Line
                type="monotone"
                dataKey="chips"
                stroke="#eab308"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
