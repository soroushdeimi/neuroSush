import { useEffect, useState } from 'react';

export type Theme = 'light' | 'dark' | 'system';

export function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined' || !window.matchMedia) return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function getStoredTheme(): Theme {
  if (typeof window === 'undefined' || !window.localStorage) return 'system';
  try {
    const t = localStorage.getItem('studio-theme') as Theme;
    if (t === 'light' || t === 'dark' || t === 'system') return t;
  } catch {
    // ignore storage read error
  }
  return 'system';
}

export function applyTheme(theme: Theme) {
  if (typeof document === 'undefined') return;
  const resolved = theme === 'system' ? getSystemTheme() : theme;
  document.documentElement.setAttribute('data-theme', resolved);
  if (theme === 'system') {
    localStorage.removeItem('studio-theme');
  } else {
    localStorage.setItem('studio-theme', theme);
  }
}

export function initReducedMotion() {
  if (typeof window === 'undefined' || !window.matchMedia) return;
  const media = window.matchMedia('(prefers-reduced-motion: reduce)');
  const update = (matches: boolean) => {
    try {
      const stored = localStorage.getItem('studio-reduced-motion');
      const reduced = stored !== null ? stored === 'true' : matches;
      document.documentElement.dataset.reducedMotion = String(reduced);
    } catch {
      document.documentElement.dataset.reducedMotion = String(matches);
    }
  };
  update(media.matches);
  media.addEventListener('change', (e) => update(e.matches));
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(getStoredTheme);

  useEffect(() => {
    applyTheme(theme);
    initReducedMotion();
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const listener = () => {
      if (getStoredTheme() === 'system') {
        applyTheme('system');
      }
    };
    media.addEventListener('change', listener);
    return () => media.removeEventListener('change', listener);
  }, [theme]);

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme);
    applyTheme(newTheme);
  };

  return [theme, setTheme] as const;
}
