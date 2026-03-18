import { NextResponse } from 'next/server';

const PYTHON_SERVER = 'http://127.0.0.1:8765';

export async function GET() {
  try {
    const resp = await fetch(`${PYTHON_SERVER}/health`, {
      signal: AbortSignal.timeout(1500),
    });
    if (resp.ok) {
      const data = await resp.json();
      return NextResponse.json(data);
    }
  } catch {
    // Python server not running
  }
  return NextResponse.json({ ok: false }, { status: 503 });
}
