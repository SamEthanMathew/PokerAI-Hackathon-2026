'use client';

import type { SessionStats } from '@/lib/engine/types';

interface StatsPanelProps {
  stats: SessionStats;
}

export default function StatsPanel({ stats }: StatsPanelProps) {
  const pct = (n: number) => `${Math.round(n * 100)}%`;
  const winRate = stats.total_hands_played > 0
    ? stats.hands_won / stats.total_hands_played
    : 0;

  const rows = [
    { label: 'Hands', value: stats.total_hands_played },
    { label: 'Win Rate', value: pct(winRate) },
    { label: 'Net', value: stats.net_chips >= 0 ? `+${stats.net_chips}` : stats.net_chips },
    { label: 'VPIP', value: pct(stats.vpip) },
    { label: 'PFR', value: pct(stats.pfr) },
    { label: 'AF', value: stats.aggression_factor.toFixed(1) },
  ];

  return (
    <div className="flex flex-col gap-1">
      <div className="text-xs text-gray-400 font-medium mb-1">Session Stats</div>
      {rows.map(({ label, value }) => (
        <div key={label} className="flex justify-between text-xs">
          <span className="text-gray-500">{label}</span>
          <span className={`font-mono ${
            label === 'Net'
              ? stats.net_chips >= 0 ? 'text-green-400' : 'text-red-400'
              : 'text-gray-300'
          }`}>
            {value}
          </span>
        </div>
      ))}
    </div>
  );
}
