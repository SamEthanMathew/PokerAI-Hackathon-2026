'use client';

import { useState, useCallback } from 'react';
import type { GameState, SessionStats } from '@/lib/engine/types';
import type { Session } from '@/lib/data/export';
import SessionTable from '@/components/review/SessionTable';
import StatsCharts from '@/components/review/StatsCharts';
import DiscardHeatmap from '@/components/review/DiscardHeatmap';
import Link from 'next/link';

export default function ReviewPage() {
  const [session, setSession] = useState<Session | null>(null);
  const [activeTab, setActiveTab] = useState<'table' | 'charts' | 'discard'>('table');

  const handleFileLoad = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target?.result as string) as Session;
        setSession(data);
      } catch {
        alert('Invalid session file');
      }
    };
    reader.readAsText(file);
  }, []);

  const tabs = [
    { id: 'table' as const, label: 'Decision Table' },
    { id: 'charts' as const, label: 'Stats Charts' },
    { id: 'discard' as const, label: 'Discard Heatmap' },
  ];

  return (
    <div className="min-h-screen bg-[#0f1a0f] p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Review Session Data</h1>
        <Link href="/" className="text-sm text-gray-400 hover:text-gray-300">← Back</Link>
      </div>

      {!session ? (
        <div className="flex flex-col items-center justify-center flex-1 gap-6 min-h-[60vh]">
          <p className="text-gray-400">Load a session JSON file exported from gameplay</p>
          <label className="px-6 py-3 bg-yellow-400 text-gray-900 font-bold rounded-lg cursor-pointer hover:bg-yellow-300 transition-colors">
            Load Session File
            <input type="file" accept=".json" onChange={handleFileLoad} className="hidden" />
          </label>
        </div>
      ) : (
        <div className="flex flex-col gap-4 flex-1">
          {/* Session summary */}
          <div className="bg-gray-800/50 rounded-xl p-4 flex gap-6 text-sm">
            <div>
              <span className="text-gray-400">Hands: </span>
              <span className="text-white font-bold">{session.summary.total_hands_played}</span>
            </div>
            <div>
              <span className="text-gray-400">Win Rate: </span>
              <span className="text-white font-bold">
                {session.summary.total_hands_played > 0
                  ? Math.round(session.summary.hands_won / session.summary.total_hands_played * 100)
                  : 0}%
              </span>
            </div>
            <div>
              <span className="text-gray-400">Net: </span>
              <span className={`font-bold ${session.summary.net_chips >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {session.summary.net_chips >= 0 ? '+' : ''}{session.summary.net_chips}
              </span>
            </div>
            <div>
              <span className="text-gray-400">VPIP: </span>
              <span className="text-white font-bold">{Math.round(session.summary.vpip * 100)}%</span>
            </div>
            <div>
              <span className="text-gray-400">Decision Points: </span>
              <span className="text-white font-bold">{session.hands.length}</span>
            </div>
            <button
              onClick={() => setSession(null)}
              className="ml-auto text-xs text-gray-500 hover:text-gray-300"
            >
              Load Different File
            </button>
          </div>

          {/* Tab navigation */}
          <div className="flex gap-2">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-yellow-400 text-gray-900'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 min-h-0">
            {activeTab === 'table' && <SessionTable hands={session.hands} />}
            {activeTab === 'charts' && <StatsCharts hands={session.hands} stats={session.summary} />}
            {activeTab === 'discard' && <DiscardHeatmap hands={session.hands} />}
          </div>
        </div>
      )}
    </div>
  );
}
