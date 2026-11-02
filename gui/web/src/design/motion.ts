import { useEffect, useState } from 'react';

export const motion = {
  instant: 80, fast: 160, base: 240, slow: 420, scene: 700,
  snappy: { stiffness: 500, damping: 40 },
  gentle: { stiffness: 170, damping: 26 },
  bouncy: { stiffness: 300, damping: 18 },
} as const;

export function useReducedMotion() {
  const [enabled, setEnabled] = useState(() => localStorage.getItem('studio-reduced-motion') === 'true');
  useEffect(() => {
    document.documentElement.dataset.reducedMotion = String(enabled);
    localStorage.setItem('studio-reduced-motion', String(enabled));
  }, [enabled]);
  return [enabled, setEnabled] as const;
}
