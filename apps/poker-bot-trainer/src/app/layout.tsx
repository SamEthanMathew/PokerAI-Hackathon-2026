import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Poker Bot Trainer',
  description: "Play 27-card Hold'em, collect training data, train your own bot",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0f1a0f] text-white">
        {children}
      </body>
    </html>
  );
}
