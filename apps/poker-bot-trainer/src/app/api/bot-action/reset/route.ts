import { NextResponse } from 'next/server';

const PYTHON_SERVER = 'http://127.0.0.1:8765';

export async function POST() {
  try {
    const resp = await fetch(`${PYTHON_SERVER}/reset`, {
      method: 'POST',
      signal: AbortSignal.timeout(2000),
    });
    if (resp.ok) {
      return NextResponse.json({ ok: true });
    }
  } catch {
    // Server not running — no-op
  }
  return NextResponse.json({ ok: true });
}
