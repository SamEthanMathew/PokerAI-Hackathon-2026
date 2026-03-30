'use client';

import GameBoard from '@/components/game/GameBoard';

export default function PlayPage() {
  return (
    <div className="h-screen overflow-hidden bg-green-900 p-3" style={{
      background: 'radial-gradient(ellipse at center, #1a4a1a 0%, #0f2a0f 60%, #080f08 100%)',
    }}>
      <GameBoard />
    </div>
  );
}
