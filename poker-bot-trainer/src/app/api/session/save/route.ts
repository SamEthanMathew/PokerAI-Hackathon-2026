import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// ============================================================================
// POST /api/session/save
// Appends completed hand records to training/data/accumulated_session.json.
// Called automatically after each hand ends.
//
// Body: { records: GameState[], humanWeight?: number }
// ============================================================================

// process.cwd() = poker-bot-trainer/ when running `next dev`
const ACCUMULATED = path.join(process.cwd(), 'training', 'data', 'accumulated_session.json');

function loadAccumulated(): object[] {
  try {
    if (fs.existsSync(ACCUMULATED)) {
      return JSON.parse(fs.readFileSync(ACCUMULATED, 'utf-8'));
    }
  } catch {
    // corrupt file — start fresh
  }
  return [];
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const records: Record<string, unknown>[] = body.records ?? [];
    const humanWeight: number = body.humanWeight ?? 3;

    if (!records.length) {
      return NextResponse.json({ ok: true, added: 0 });
    }

    // Find hand outcome from whichever record has hand_result
    const outcomeRecord = records.find(r => r.hand_result) as Record<string, unknown> | undefined;
    const handResult = outcomeRecord?.hand_result as { won?: boolean; chips_won_lost?: number } | undefined;
    const humanWon: boolean | null = handResult != null ? (handResult.won ?? false) : null;
    const chipsAbs = Math.abs((handResult?.chips_won_lost ?? 0));

    // Find final community cards (for post-hoc optimal discard labels)
    // Look for the last record with 5 community cards (full board)
    const fullBoardRecord = [...records].reverse().find(r => {
      const comm = r.community_cards as number[] | undefined;
      return comm && comm.filter((c: number) => c >= 0).length >= 5;
    });
    const finalCommunity = fullBoardRecord
      ? (fullBoardRecord.community_cards as number[]).filter((c: number) => c >= 0)
      : null;

    // Tag each record with outcome context and weight
    const tagged = records.map(r => {
      const player = (r.player as string) ?? 'human';

      // Outcome weight: reward actions that led to wins, penalise those that led to losses
      // Scale slightly by chips magnitude (big wins/losses carry more signal)
      const magnitude = Math.min(chipsAbs / 20, 2.0); // 0-2 scale
      let outcome_weight = 1.0;
      if (humanWon === null) {
        outcome_weight = 1.0;
      } else if (player === 'human') {
        outcome_weight = humanWon ? 1.5 + 0.5 * magnitude : 0.25;
      } else {
        // bot record: bot won when human lost
        outcome_weight = humanWon ? 0.4 : 1.5 + 0.5 * magnitude;
      }

      const tagged: Record<string, unknown> = {
        ...r,
        outcome: humanWon === null ? null : (humanWon ? 'human_won' : 'bot_won'),
        outcome_weight,
      };

      // Attach final community to discard records so training can compute optimal label
      const action = r.action_taken as { type?: string } | undefined;
      if (action?.type === 'discard' && finalCommunity) {
        tagged.final_community_cards = finalCommunity;
      }

      return tagged;
    });

    // Replicate based on player importance (human actions still replicated more to ensure coverage)
    const weighted: object[] = [];
    for (const r of tagged) {
      const player = (r.player as string) ?? 'human';
      const reps = player === 'human' ? humanWeight : 1;
      for (let i = 0; i < reps; i++) weighted.push(r);
    }

    // Ensure data dir exists
    const dir = path.dirname(ACCUMULATED);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    const existing = loadAccumulated();
    const merged = [...existing, ...weighted];
    fs.writeFileSync(ACCUMULATED, JSON.stringify(merged));

    return NextResponse.json({ ok: true, added: weighted.length, total: merged.length });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

export async function GET() {
  const existing = loadAccumulated();
  return NextResponse.json({ total: existing.length });
}
