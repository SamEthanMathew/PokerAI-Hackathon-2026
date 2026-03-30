import { NextRequest, NextResponse } from 'next/server';
import type { EngineState } from '@/lib/engine/types';
import { getBotDecision } from '@/lib/bot/heuristicBot';

const PYTHON_SERVER = 'http://127.0.0.1:8765';

// ============================================================================
// POST /api/bot-action
// Proxies to Python bot server (genesisV2 or clone).
// Falls back to heuristic bot if server is not reachable.
// ============================================================================

interface BotActionBody {
  engineState: EngineState;
  botIndex: 0 | 1;
  humanIndex: 0 | 1;
  hand_number: number;
  last_human_action: string;
}

export async function POST(req: NextRequest) {
  const body: BotActionBody = await req.json();
  const { engineState, botIndex, humanIndex, hand_number, last_human_action } = body;

  // Build a gym_env-style observation dict from the TypeScript EngineState
  const myState = engineState.players[botIndex];
  const oppState = engineState.players[humanIndex];

  // Pad cards to fixed lengths (gym_env pads with -1)
  const padTo = (arr: number[], len: number) =>
    [...arr, ...Array(Math.max(0, len - arr.length)).fill(-1)];

  const observation = {
    street: engineState.street,
    acting_agent: engineState.acting_agent,
    my_cards: padTo(myState.cards, 5),
    community_cards: padTo(engineState.community_cards, 5),
    my_bet: myState.bet,
    my_discarded_cards: padTo(myState.discarded, 3),
    opp_bet: oppState.bet,
    opp_discarded_cards: padTo(oppState.discarded, 3),
    min_raise: engineState.min_raise,
    max_raise: engineState.max_raise,
    valid_actions: [
      engineState.valid_actions.fold ? 1 : 0,
      engineState.valid_actions.raise ? 1 : 0,
      engineState.valid_actions.check ? 1 : 0,
      engineState.valid_actions.call ? 1 : 0,
      engineState.valid_actions.discard ? 1 : 0,
    ],
    pot_size: myState.bet + oppState.bet,
    blind_position: botIndex === engineState.small_blind_player ? 0 : 1,
    time_used: 0,
    time_left: 1000,
  };

  // Try Python server
  try {
    const resp = await fetch(`${PYTHON_SERVER}/act`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        observation,
        hand_number,
        last_human_action,
        reward: 0,
        terminated: false,
      }),
      signal: AbortSignal.timeout(3500),
    });

    if (resp.ok) {
      const data = await resp.json();
      return NextResponse.json({ ...data, source: 'python' });
    }
  } catch {
    // Python server not running — fall through to heuristic
  }

  // Heuristic fallback
  const { action } = getBotDecision(engineState, botIndex);
  const tuple = actionToTuple(action);
  return NextResponse.json({
    action: tuple,
    action_label: action.type,
    source: 'heuristic',
  });
}


function actionToTuple(action: { type: string; raise_amount?: number; keep_card_1?: number; keep_card_2?: number }): [number, number, number, number] {
  switch (action.type) {
    case 'FOLD':    return [0, 0, 0, 1];
    case 'RAISE':   return [1, action.raise_amount ?? 0, 0, 1];
    case 'CHECK':   return [2, 0, 0, 1];
    case 'CALL':    return [3, 0, 0, 1];
    case 'DISCARD': return [4, 0, action.keep_card_1 ?? 0, action.keep_card_2 ?? 1];
    default:        return [0, 0, 0, 1];
  }
}
