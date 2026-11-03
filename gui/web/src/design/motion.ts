import { useEffect, useState } from 'react';

export interface SpringConfig {
  stiffness: number;
  damping: number;
  mass?: number;
}

export const motion = {
  instant: 80,
  fast: 160,
  base: 240,
  slow: 420,
  scene: 700,
  snappy: { stiffness: 500, damping: 40, mass: 1 },
  gentle: { stiffness: 170, damping: 26, mass: 1 },
  bouncy: { stiffness: 300, damping: 18, mass: 1 },
  settle: { stiffness: 220, damping: 30, mass: 1 },
} as const;

export function getSpring(preset: keyof typeof motion, reduced: boolean): SpringConfig | { duration: number } {
  if (reduced) {
    return { duration: 0 };
  }
  const val = motion[preset];
  if (typeof val === 'object' && val !== null && 'stiffness' in val) {
    return val as SpringConfig;
  }
  return { duration: typeof val === 'number' ? val : 240 };
}

export function useReducedMotion() {
  const [enabled, setEnabled] = useState(() => {
    if (typeof window === 'undefined') return false;
    try {
      return localStorage.getItem('studio-reduced-motion') === 'true' || window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch {
      // ignore storage access error
      return false;
    }
  });

  useEffect(() => {
    if (typeof document === 'undefined') return;
    document.documentElement.dataset.reducedMotion = String(enabled);
    try {
      localStorage.setItem('studio-reduced-motion', String(enabled));
    } catch {
      // ignore storage write error
    }
  }, [enabled]);

  return [enabled, setEnabled] as const;
}
