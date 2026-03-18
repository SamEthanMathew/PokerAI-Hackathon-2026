import Link from 'next/link';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-8 p-8 bg-[#0f1a0f]">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-white mb-3">Poker Bot Trainer</h1>
        <p className="text-gray-400 max-w-lg">
          Play 27-card Hold'em (3 suits, 5 hole cards, mandatory discard) against a bot.
          Your decisions are logged to train a personal AI via behavioral cloning.
        </p>
      </div>

      <div className="flex gap-4">
        <Link
          href="/play"
          className="px-8 py-4 bg-yellow-400 text-gray-900 font-bold rounded-xl text-lg hover:bg-yellow-300 transition-all shadow-lg"
        >
          Play
        </Link>
        <Link
          href="/review"
          className="px-8 py-4 bg-gray-700 text-white font-bold rounded-xl text-lg hover:bg-gray-600 transition-all"
        >
          Review Data
        </Link>
      </div>

      <div className="grid grid-cols-3 gap-6 max-w-2xl text-center">
        {[
          { step: '1', title: 'Play', desc: '1000 hands of 27-card Hold\'em' },
          { step: '2', title: 'Review', desc: 'Analyze your decisions and patterns' },
          { step: '3', title: 'Train', desc: 'Export data → train your bot' },
        ].map(({ step, title, desc }) => (
          <div key={step} className="bg-gray-800/50 rounded-xl p-4">
            <div className="text-yellow-400 font-bold text-2xl mb-1">{step}</div>
            <div className="text-white font-semibold mb-1">{title}</div>
            <div className="text-gray-400 text-sm">{desc}</div>
          </div>
        ))}
      </div>

      <div className="text-gray-600 text-xs text-center max-w-md">
        27-card deck · 3 suits (♦♥♠) · Ranks 2-9,A · 5 hole cards → discard 3 on flop
      </div>
    </main>
  );
}
