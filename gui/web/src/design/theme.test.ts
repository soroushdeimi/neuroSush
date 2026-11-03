import { describe, it, expect, beforeEach, vi } from 'vitest';
import { applyTheme, initReducedMotion } from './theme';

describe('theme helper', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.documentElement.removeAttribute('data-reduced-motion');
  });

  it('applies light theme and sets data-theme attribute', () => {
    applyTheme('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    expect(localStorage.getItem('studio-theme')).toBe('light');
  });

  it('applies dark theme and sets data-theme attribute', () => {
    applyTheme('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(localStorage.getItem('studio-theme')).toBe('dark');
  });

  it('handles system theme by resolving to prefers-color-scheme', () => {
    vi.stubGlobal('matchMedia', (query: string) => ({
      matches: query.includes('dark'),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }));

    applyTheme('system');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(localStorage.getItem('studio-theme')).toBeNull();

    vi.unstubAllGlobals();
  });

  it('respects prefers-reduced-motion', () => {
    vi.stubGlobal('matchMedia', (query: string) => ({
      matches: query.includes('reduce'),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }));

    initReducedMotion();
    expect(document.documentElement.dataset.reducedMotion).toBe('true');

    vi.unstubAllGlobals();
  });
});
