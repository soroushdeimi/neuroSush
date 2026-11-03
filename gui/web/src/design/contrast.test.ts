// @vitest-environment node
import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';

function parseHex(hex: string): [number, number, number] {
  let clean = hex.replace('#', '').trim();
  if (clean.length === 3) {
    clean = clean.split('').map(c => c + c).join('');
  }
  if (clean.length !== 6) {
    throw new Error(`Invalid hex color: ${hex}`);
  }
  const r = parseInt(clean.substring(0, 2), 16);
  const g = parseInt(clean.substring(2, 4), 16);
  const b = parseInt(clean.substring(4, 6), 16);
  return [r, g, b];
}

function getLuminance(r: number, g: number, b: number): number {
  const a = [r, g, b].map(v => {
    const channel = v / 255;
    return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.05, 2.4);
  });
  return 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2];
}

function getContrast(color1: string, color2: string): number {
  const [r1, g1, b1] = parseHex(color1);
  const [r2, g2, b2] = parseHex(color2);
  const l1 = getLuminance(r1, g1, b1);
  const l2 = getLuminance(r2, g2, b2);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

describe('Design Token WCAG Contrast', () => {
  const cssPath = path.resolve(__dirname, 'tokens.css');
  const tokensCss = fs.readFileSync(cssPath, 'utf-8');

  it('asserts theme sections exist and are non-empty', () => {
    expect(tokensCss).toContain('/* LIGHT_THEME */');
    expect(tokensCss).toContain('/* DARK_THEME */');
  });

  function extractVars(block: string): Record<string, string> {
    const vars: Record<string, string> = {};
    const regex = /(--[\w-]+)\s*:\s*([^;]+);/g;
    let match;
    while ((match = regex.exec(block)) !== null) {
      vars[match[1].trim()] = match[2].trim();
    }
    return vars;
  }

  const lightSection = tokensCss.split('/* DARK_THEME */')[0] || '';
  const darkSection = tokensCss.split('/* DARK_THEME */')[1] || '';

  expect(lightSection.trim().length).toBeGreaterThan(0);
  expect(darkSection.trim().length).toBeGreaterThan(0);

  const lightVars = extractVars(lightSection);
  const darkVars = extractVars(darkSection);

  const uiTokens = ['--accent', '--coral', '--record'];
  const textTokens = ['--fg', '--muted', '--error', '--exc', '--inh', '--input', '--plastic', '--warn', '--ok'];

  it('passes WCAG AA contrast (4.5:1 text, 3:1 UI) for Paper (light) theme with exact count', () => {
    const bg = lightVars['--bg'];
    expect(bg).toBeDefined();
    expect(bg.startsWith('#')).toBe(true);

    let checkedCount = 0;

    for (const token of textTokens) {
      const color = lightVars[token];
      expect(color, `Token ${token} must exist in light theme`).toBeDefined();
      expect(color.startsWith('#'), `Token ${token} value ${color} must be a hex color`).toBe(true);
      const contrast = getContrast(color, bg);
      expect(contrast).toBeGreaterThanOrEqual(4.5);
      checkedCount++;
    }

    for (const token of uiTokens) {
      const color = lightVars[token];
      expect(color, `Token ${token} must exist in light theme`).toBeDefined();
      expect(color.startsWith('#'), `Token ${token} value ${color} must be a hex color`).toBe(true);
      const contrast = getContrast(color, bg);
      expect(contrast).toBeGreaterThanOrEqual(3.0);
      checkedCount++;
    }

    expect(checkedCount).toBe(textTokens.length + uiTokens.length);
  });

  it('passes WCAG AA contrast (4.5:1 text, 3:1 UI) for Orbit (dark) theme with exact count', () => {
    const bg = darkVars['--bg'];
    expect(bg).toBeDefined();
    expect(bg.startsWith('#')).toBe(true);

    let checkedCount = 0;
    const allTokens = [...textTokens, ...uiTokens];

    for (const token of allTokens) {
      const color = darkVars[token];
      expect(color, `Token ${token} must exist in dark theme`).toBeDefined();
      expect(color.startsWith('#'), `Token ${token} value ${color} must be a hex color`).toBe(true);
      const contrast = getContrast(color, bg);
      expect(contrast).toBeGreaterThanOrEqual(4.5);
      checkedCount++;
    }

    expect(checkedCount).toBe(allTokens.length);
  });

  it('correctly flags low-contrast and known contrast pairs', () => {
    const lowContrast = getContrast('#777777', '#888888');
    expect(lowContrast).toBeLessThan(4.5);

    const whiteBlackContrast = getContrast('#000000', '#ffffff');
    expect(whiteBlackContrast).toBeCloseTo(21, 0);
  });
});
