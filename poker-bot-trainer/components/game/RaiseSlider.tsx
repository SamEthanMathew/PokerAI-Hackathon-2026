'use client';

import * as Slider from '@radix-ui/react-slider';

interface RaiseSliderProps {
  min: number;
  max: number;
  value: number;
  onChange: (v: number) => void;
  pot: number;
}

export default function RaiseSlider({ min, max, value, onChange, pot }: RaiseSliderProps) {
  const halfPot = Math.max(min, Math.min(Math.floor(pot / 2), max));
  const fullPot = Math.max(min, Math.min(pot, max));

  const quickButtons = [
    { label: 'Min', val: min },
    { label: '½ Pot', val: halfPot },
    { label: 'Pot', val: fullPot },
    { label: 'All-In', val: max },
  ];

  return (
    <div className="flex flex-col gap-2 w-full max-w-xs">
      <div className="flex justify-between text-xs text-gray-400">
        <span>Min: {min}</span>
        <span className="text-yellow-300 font-bold text-sm">{value}</span>
        <span>Max: {max}</span>
      </div>

      <Slider.Root
        className="relative flex items-center select-none touch-none w-full h-5"
        value={[value]}
        min={min}
        max={max}
        step={1}
        onValueChange={([v]) => onChange(v)}
      >
        <Slider.Track className="bg-gray-600 relative grow rounded-full h-1.5">
          <Slider.Range className="absolute bg-yellow-400 rounded-full h-full" />
        </Slider.Track>
        <Slider.Thumb
          className="block w-4 h-4 bg-yellow-400 rounded-full shadow-md hover:bg-yellow-300 focus:outline-none focus:ring-2 focus:ring-yellow-400 cursor-pointer"
          aria-label="Raise amount"
        />
      </Slider.Root>

      <div className="flex gap-1">
        {quickButtons.map(({ label, val }) => (
          <button
            key={label}
            onClick={() => onChange(val)}
            className={`flex-1 text-xs py-1 rounded transition-colors ${
              value === val
                ? 'bg-yellow-400 text-gray-900 font-bold'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
