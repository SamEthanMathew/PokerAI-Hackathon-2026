'use client';

import { useEffect } from 'react';
import { useGameStore } from '@/lib/store/gameStore';
import OpponentArea from './OpponentArea';
import CommunityCards from './CommunityCards';
import PlayerHand from './PlayerHand';
import ActionBar from './ActionBar';
import HandHistory from './HandHistory';
import StatsPanel from './StatsPanel';
import HandResult from './HandResult';
import { evaluateBest } from '@/lib/engine/evaluator';

export default function GameBoard() {
  const {
    phase,
    engineState,
    humanIndex,
    selectedKeep,
    raiseAmount,
    handHistory,
    sessionStats,
    handNumber,
    lastBotAction,
    lastHandResult,
    isShowdown,
    botMode,
    botLabel,
    startNewHand,
    submitAction,
    toggleKeep,
    confirmDiscard,
    setRaiseAmount,
    exportCurrentSession,
    dismissHandResult,
    resetSession,
    checkPythonServer,
  } = useGameStore();

  // Check if Python bot server is running on mount
  useEffect(() => {
    checkPythonServer();
  }, [checkPythonServer]);

  // ── Idle / start screen ────────────────────────────────────────────────────
  if (phase === 'idle') {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-6">
        <h1 className="text-3xl font-bold text-white">Poker Bot Trainer</h1>
        <p className="text-gray-400 text-center max-w-md">
          Play 1000 hands of 27-card Hold'em against the bot. Every decision you make is recorded to train your personal poker AI.
        </p>
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${botMode === 'python' ? 'bg-green-900/60 text-green-300 border border-green-700' : 'bg-gray-800 text-gray-400 border border-gray-700'}`}>
          <span className={`w-2 h-2 rounded-full ${botMode === 'python' ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
          {botMode === 'python' ? `Opponent: ${botLabel}` : 'Opponent: Heuristic Bot'}
        </div>
        <button
          onClick={startNewHand}
          className="px-8 py-3 bg-yellow-400 text-gray-900 font-bold rounded-xl text-lg hover:bg-yellow-300 transition-all shadow-lg"
        >
          Start Playing
        </button>
      </div>
    );
  }

  if (phase === 'session_over') {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-6">
        <h2 className="text-2xl font-bold text-white">Session Complete!</h2>
        <p className="text-green-400 text-xl font-bold">Net: {sessionStats.net_chips >= 0 ? '+' : ''}{sessionStats.net_chips} chips</p>
        <div className="flex gap-3">
          <button onClick={exportCurrentSession} className="px-6 py-2 bg-blue-600 text-white rounded-lg font-bold hover:bg-blue-500">
            Export Data
          </button>
          <button onClick={resetSession} className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-500">
            New Session
          </button>
        </div>
      </div>
    );
  }

  if (!engineState) return null;

  const oppIndex = (1 - humanIndex) as 0 | 1;
  const myState = engineState.players[humanIndex];
  const oppState = engineState.players[oppIndex];
  const myPosition = humanIndex === engineState.small_blind_player ? 'SB' : 'BB';
  const oppPosition = oppIndex === engineState.small_blind_player ? 'SB' : 'BB';

  const isMyDiscard = phase === 'discard_my_turn';
  const isBotDiscard = phase === 'discard_bot_turn';
  const isMyAction = phase === 'player_action';
  const isBotThinking = phase === 'bot_thinking' || isBotDiscard;
  const isActionDisabled = !isMyAction && !isMyDiscard;

  // Hand name for community cards display
  let handName: string | undefined;
  if (engineState.community_cards.length >= 3 && myState.cards.length === 2) {
    const best = evaluateBest(myState.cards, engineState.community_cards);
    if (best.rank > 1) handName = best.name;
  }

  return (
    <div className="flex h-full gap-4">
      {/* Main game area */}
      <div className="flex-1 flex flex-col justify-between py-4 px-2 min-h-0">

        {/* Opponent area */}
        <div className="flex justify-center">
          <OpponentArea
            cards={oppState.cards}
            discarded={oppState.discarded}
            bet={oppState.bet}
            lastAction={lastBotAction}
            isThinking={isBotThinking}
            isShowdown={isShowdown}
            position={oppPosition}
          />
        </div>

        {/* Community cards + pot */}
        <div className="flex justify-center">
          <CommunityCards
            cards={engineState.community_cards}
            potSize={myState.bet + oppState.bet}
            street={engineState.street}
            handName={handName}
          />
        </div>

        {/* Player hand */}
        <div className="flex justify-center">
          <PlayerHand
            cards={myState.cards}
            discarded={myState.discarded}
            bet={myState.bet}
            position={myPosition}
            isDiscardPhase={isMyDiscard}
            keptIndices={selectedKeep}
            onToggleKeep={toggleKeep}
            onConfirmDiscard={confirmDiscard}
            stack={100 - myState.bet}
          />
        </div>

        {/* Action bar */}
        {!isMyDiscard && (
          <div className="flex justify-center">
            <ActionBar
              validActions={engineState.valid_actions}
              minRaise={engineState.min_raise}
              maxRaise={engineState.max_raise}
              potSize={myState.bet + oppState.bet}
              myBet={myState.bet}
              oppBet={oppState.bet}
              raiseAmount={raiseAmount}
              onAction={submitAction}
              onRaiseAmountChange={setRaiseAmount}
              disabled={isActionDisabled}
            />
          </div>
        )}

        {/* Bot discard notice */}
        {isBotDiscard && (
          <div className="text-center text-gray-400 text-sm animate-pulse">
            Opponent is selecting cards to discard...
          </div>
        )}
      </div>

      {/* Right sidebar */}
      <div className="w-48 flex flex-col gap-4 py-4 pr-2 min-h-0">
        <div className="flex justify-between items-center">
          <span className="text-xs text-gray-500">Hand {handNumber}/1000</span>
          <button
            onClick={exportCurrentSession}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            Export
          </button>
        </div>
        <div className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs ${botMode === 'python' ? 'bg-green-900/40 text-green-400' : 'bg-gray-800/60 text-gray-500'}`}>
          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${botMode === 'python' ? 'bg-green-400' : 'bg-gray-500'}`} />
          <span className="truncate">{botMode === 'python' ? botLabel : 'Heuristic'}</span>
        </div>
        <div className="bg-gray-800/50 rounded-xl p-3">
          <StatsPanel stats={sessionStats} />
        </div>
        <div className="flex-1 bg-gray-800/50 rounded-xl p-3 min-h-0 overflow-hidden">
          <HandHistory hands={handHistory} currentHand={handNumber} />
        </div>
      </div>

      {/* Hand result overlay */}
      {phase === 'hand_result' && lastHandResult && (
        <HandResult
          result={lastHandResult}
          engineState={engineState}
          humanIndex={humanIndex}
          onDismiss={dismissHandResult}
        />
      )}
    </div>
  );
}
