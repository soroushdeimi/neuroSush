import { describe, it, expect } from 'vitest';
import { motion, getSpring } from './motion';

describe('motion system and springs', () => {
  it('has required spring presets with stiffness, damping and mass', () => {
    expect(motion.snappy).toHaveProperty('stiffness', 500);
    expect(motion.gentle).toHaveProperty('stiffness', 170);
    expect(motion.bouncy).toHaveProperty('stiffness', 300);
    expect(motion.settle).toHaveProperty('stiffness', 220);
  });

  it('returns spring config when reduced motion is false', () => {
    const config = getSpring('snappy', false);
    expect(config).toEqual({ stiffness: 500, damping: 40, mass: 1 });
  });

  it('returns zero duration when reduced motion is true', () => {
    const config = getSpring('snappy', true);
    expect(config).toEqual({ duration: 0 });
  });
});
