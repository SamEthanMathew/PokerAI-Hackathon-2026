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
    const records: object[] = body.records ?? [];
    const humanWeight: number = body.humanWeight ?? 3;

    if (!records.length) {
      return NextResponse.json({ ok: true, added: 0 });
    }

    // Weight human records
    const weighted: object[] = [];
    for (const r of records) {
      const player = (r as { player?: string }).player ?? 'human';
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
