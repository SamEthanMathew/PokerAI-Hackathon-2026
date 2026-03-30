'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import type { ValidActions, Action } from '@/lib/engine/types';
import RaiseSlider from './RaiseSlider';

interface ActionBarProps {
  validActions: ValidActions;
  minRaise: number;
  maxRaise: number;
  potSize: number;
  myBet: number;
  oppBet: number;
  raiseAmount: number;
  onAction: (action: Action) => void;
  onRaiseAmountChange: (v: number) => void;
  disabled: boolean;
}

export default function ActionBar({
  validActions,
  minRaise,
  maxRaise,
  potSize,
  myBet,
  oppBet,
  raiseAmount,
  onAction,
  onRaiseAmountChange,
  disabled,
}: ActionBarProps) {
  const [showRaise, setShowRaise] = useState(false);
  const toCall = oppBet - myBet;

  const callLabel = toCall > 0 ? `Call ${toCall}` : 'Call';
  const raiseLabel = toCall > 0 ? `Raise to ${oppBet + raiseAmount}` : `Bet ${raiseAmount}`;

  function handleFold() { onAction({ type: 'FOLD' }); }
  function handleCheck() { onAction({ type: 'CHECK' }); }
  function handleCall() { onAction({ type: 'CALL' }); }
  function handleRaise() {
    onAction({ type: 'RAISE', raise_amount: raiseAmount });
    setShowRaise(false);
  }

  if (disabled) {
    return (
      <div className="flex gap-3 justify-center items-center h-16">
        <div className="text-gray-500 text-sm animate-pulse">Waiting...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-3">
      {showRaise && validActions.raise && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800 rounded-xl p-3 border border-gray-600"
        >
          <RaiseSlider
            min={minRaise}
            max={maxRaise}
            value={raiseAmount}
            onChange={onRaiseAmountChange}
            pot={potSize}
          />
        </motion.div>
      )}

      <div className="flex gap-3">
        {/* Fold */}
        <ActionButton
          onClick={handleFold}
          color="red"
          label="Fold"
          enabled={validActions.fold}
        />

        {/* Check */}
        {validActions.check && (
          <ActionButton onClick={handleCheck} color="gray" label="Check" enabled />
        )}

        {/* Call */}
        {validActions.call && (
          <ActionButton onClick={handleCall} color="blue" label={callLabel} enabled />
        )}

        {/* Raise toggle / confirm */}
        {validActions.raise && (
          showRaise ? (
            <ActionButton onClick={handleRaise} color="yellow" label={raiseLabel} enabled />
          ) : (
            <ActionButton onClick={() => setShowRaise(true)} color="yellow" label="Raise" enabled />
          )
        )}

        {showRaise && (
          <ActionButton
            onClick={() => setShowRaise(false)}
            color="gray"
            label="✕"
            enabled
          />
        )}
      </div>
    </div>
  );
}

interface ActionButtonProps {
  onClick: () => void;
  color: 'red' | 'blue' | 'yellow' | 'gray' | 'green';
  label: string;
  enabled: boolean;
}

function ActionButton({ onClick, color, label, enabled }: ActionButtonProps) {
  const colors = {
    red: 'bg-red-600 hover:bg-red-500 text-white',
    blue: 'bg-blue-600 hover:bg-blue-500 text-white',
    yellow: 'bg-yellow-400 hover:bg-yellow-300 text-gray-900',
    gray: 'bg-gray-600 hover:bg-gray-500 text-gray-200',
    green: 'bg-green-600 hover:bg-green-500 text-white',
  };

  return (
    <motion.button
      onClick={enabled ? onClick : undefined}
      disabled={!enabled}
      className={`px-5 py-2.5 rounded-lg font-bold text-sm min-w-[80px] transition-colors shadow-md ${
        enabled ? colors[color] : 'bg-gray-700 text-gray-500 cursor-not-allowed'
      }`}
      whileHover={enabled ? { scale: 1.04 } : {}}
      whileTap={enabled ? { scale: 0.96 } : {}}
    >
      {label}
    </motion.button>
  );
}
