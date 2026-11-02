import { describe, it, expect } from 'vitest';
import { decodeFrame, spikeBit } from './frames';

function hexToBuffer(hex: string): ArrayBuffer {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
  }
  return bytes.buffer;
}

describe('frames codec', () => {
  it('decodes SPIKES golden frame and tests spikeBit', () => {
    const hex = '0100000001000000640000000200000001020800';
    const frame = decodeFrame(hexToBuffer(hex));

    expect(frame.type).toBe(1);
    expect(frame.flags).toBe(0);
    expect(frame.stream).toBe(0);
    expect(frame.seq).toBe(1);
    expect(frame.step0).toBe(100);
    expect(frame.nsteps).toBe(2);
    expect(frame.isDecimated).toBe(false);
    expect(frame.isLast).toBe(false);

    if (frame.type === 1) {
      expect(spikeBit(frame, 100, 0)).toBe(true);
      expect(spikeBit(frame, 100, 9)).toBe(true);
      expect(spikeBit(frame, 100, 1)).toBe(false);
      expect(spikeBit(frame, 100, 3)).toBe(false);

      expect(spikeBit(frame, 101, 3)).toBe(true);
      expect(spikeBit(frame, 101, 0)).toBe(false);
      expect(spikeBit(frame, 101, 9)).toBe(false);

      expect(spikeBit(frame, 99, 0)).toBe(false);
      expect(spikeBit(frame, 102, 0)).toBe(false);
    }
  });

  it('decodes VSAMPLE golden frame', () => {
    const hex = '02000100020000006400000002000000020000000000000005000000000082c2000081c2000071c20000f041';
    const frame = decodeFrame(hexToBuffer(hex));

    expect(frame.type).toBe(2);
    expect(frame.stream).toBe(1);
    expect(frame.seq).toBe(2);
    expect(frame.step0).toBe(100);
    expect(frame.nsteps).toBe(2);

    if (frame.type === 2) {
      expect(frame.count).toBe(2);
      expect(frame.indices).toEqual([0, 5]);
      expect(frame.values).toEqual([
        [-65.0, -64.5],
        [-60.25, 30.0],
      ]);
    }
  });

  it('decodes SCALAR golden frame', () => {
    const hex = '03000000030000000000000003000000000000000000003f00004441';
    const frame = decodeFrame(hexToBuffer(hex));

    expect(frame.type).toBe(3);
    expect(frame.stream).toBe(0);
    expect(frame.seq).toBe(3);
    expect(frame.step0).toBe(0);
    expect(frame.nsteps).toBe(3);

    if (frame.type === 3) {
      expect(frame.values).toEqual([0, 0.5, 12.25]);
    }
  });

  it('decodes WEIGHTS golden frame', () => {
    const hex = '0401000004000000f4010000010000000200030004000400000000000000803e0000003f0000403f0000803f000080bf';
    const frame = decodeFrame(hexToBuffer(hex));

    expect(frame.type).toBe(4);
    expect(frame.flags).toBe(1);
    expect(frame.isDecimated).toBe(true);
    expect(frame.stream).toBe(0);
    expect(frame.seq).toBe(4);
    expect(frame.step0).toBe(500);
    expect(frame.nsteps).toBe(1);

    if (frame.type === 4) {
      expect(frame.rows).toBe(2);
      expect(frame.cols).toBe(3);
      expect(frame.rowStride).toBe(4);
      expect(frame.colStride).toBe(4);
      expect(frame.values).toEqual([0, 0.25, 0.5, 0.75, 1.0, -1.0]);
    }
  });
});
