'use client';

import { motion } from 'framer-motion';
import type { CardInt } from '@/lib/engine/types';
import { cardToSvgPath } from '@/lib/engine/deck';

interface CardProps {
  card: CardInt | 'back' | 'empty';
  size?: 'xs' | 'sm' | 'md' | 'lg';
  selected?: boolean;
  dimmed?: boolean;
  onClick?: () => void;
  animate?: boolean;
  className?: string;
}

const SIZE_CLASSES = {
  xs: 'w-8 h-12',
  sm: 'w-10 h-14',
  md: 'w-14 h-20',
  lg: 'w-18 h-24',
};

const SIZE_PX = {
  xs: { w: 32, h: 48 },
  sm: { w: 40, h: 56 },
  md: { w: 56, h: 80 },
  lg: { w: 72, h: 96 },
};

export default function Card({
  card,
  size = 'md',
  selected = false,
  dimmed = false,
  onClick,
  animate = false,
  className = '',
}: CardProps) {
  const svgPath = cardToSvgPath(card as CardInt);
  const { w, h } = SIZE_PX[size];

  const ring = selected ? 'ring-4 ring-green-400 ring-offset-1' : '';
  const opacity = dimmed ? 'opacity-30' : 'opacity-100';
  const cursor = onClick ? 'cursor-pointer' : '';
  const shadow = selected ? 'shadow-lg shadow-green-400/50' : 'shadow-md';

  return (
    <motion.div
      className={`relative rounded-md overflow-hidden ${ring} ${opacity} ${cursor} ${shadow} transition-all duration-150 ${className}`}
      style={{ width: w, height: h }}
      onClick={onClick}
      initial={animate ? { y: -30, opacity: 0 } : false}
      animate={{ y: 0, opacity: dimmed ? 0.3 : 1 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      whileHover={onClick ? { scale: 1.05, y: -3 } : {}}
      whileTap={onClick ? { scale: 0.97 } : {}}
    >
      <img
        src={svgPath}
        alt={card === 'back' ? 'Card back' : card === 'empty' ? 'Empty' : `Card ${card}`}
        width={w}
        height={h}
        className="w-full h-full object-contain"
        draggable={false}
      />
      {selected && (
        <div className="absolute top-0.5 right-0.5 w-4 h-4 bg-green-400 rounded-full flex items-center justify-center">
          <span className="text-white text-xs font-bold">✓</span>
        </div>
      )}
    </motion.div>
  );
}
