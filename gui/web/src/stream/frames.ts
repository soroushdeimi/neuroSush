export interface BaseFrame {
  type: number;
  flags: number;
  stream: number;
  seq: number;
  step0: number;
  nsteps: number;
  isDecimated: boolean;
  isLast: boolean;
}

export interface SpikesFrame extends BaseFrame {
  type: 1;
  payload: Uint8Array;
}

export interface VSampleFrame extends BaseFrame {
  type: 2;
  count: number;
  indices: number[];
  values: number[][];
}

export interface ScalarFrame extends BaseFrame {
  type: 3;
  values: number[];
}

export interface WeightsFrame extends BaseFrame {
  type: 4;
  rows: number;
  cols: number;
  rowStride: number;
  colStride: number;
  values: number[];
}

export type Frame = SpikesFrame | VSampleFrame | ScalarFrame | WeightsFrame;

export function decodeFrame(buffer: ArrayBuffer): Frame {
  const view = new DataView(buffer);
  if (buffer.byteLength < 16) {
    throw new Error('Invalid frame: buffer too short for header');
  }

  const type = view.getUint8(0);
  const flags = view.getUint8(1);
  const stream = view.getUint16(2, true);
  const seq = view.getUint32(4, true);
  const step0 = view.getUint32(8, true);
  const nsteps = view.getUint16(12, true);

  const base: BaseFrame = {
    type,
    flags,
    stream,
    seq,
    step0,
    nsteps,
    isDecimated: (flags & 1) !== 0,
    isLast: (flags & 2) !== 0,
  };

  let offset = 16;

  switch (type) {
    case 1: {
      const payload = new Uint8Array(buffer, offset);
      return {
        ...base,
        type: 1,
        payload,
      };
    }
    case 2: {
      if (offset + 4 > buffer.byteLength) {
        throw new Error('Invalid VSAMPLE frame: payload too short');
      }
      const count = view.getUint16(offset, true);
      offset += 4;

      if (offset + count * 4 > buffer.byteLength) {
        throw new Error('Invalid VSAMPLE frame: indices truncated');
      }
      const indices: number[] = [];
      for (let i = 0; i < count; i++) {
        indices.push(view.getUint32(offset, true));
        offset += 4;
      }

      const values: number[][] = [];
      for (let s = 0; s < nsteps; s++) {
        const stepValues: number[] = [];
        for (let c = 0; c < count; c++) {
          if (offset + 4 > buffer.byteLength) {
            throw new Error('Invalid VSAMPLE frame: values truncated');
          }
          stepValues.push(view.getFloat32(offset, true));
          offset += 4;
        }
        values.push(stepValues);
      }

      return {
        ...base,
        type: 2,
        count,
        indices,
        values,
      };
    }
    case 3: {
      const values: number[] = [];
      for (let s = 0; s < nsteps; s++) {
        if (offset + 4 > buffer.byteLength) {
          throw new Error('Invalid SCALAR frame: values truncated');
        }
        values.push(view.getFloat32(offset, true));
        offset += 4;
      }
      return {
        ...base,
        type: 3,
        values,
      };
    }
    case 4: {
      if (offset + 8 > buffer.byteLength) {
        throw new Error('Invalid WEIGHTS frame: header truncated');
      }
      const rows = view.getUint16(offset, true);
      const cols = view.getUint16(offset + 2, true);
      const rowStride = view.getUint16(offset + 4, true);
      const colStride = view.getUint16(offset + 6, true);
      offset += 8;

      const totalElements = rows * cols;
      if (offset + totalElements * 4 > buffer.byteLength) {
        throw new Error('Invalid WEIGHTS frame: values truncated');
      }
      const values: number[] = [];
      for (let i = 0; i < totalElements; i++) {
        values.push(view.getFloat32(offset, true));
        offset += 4;
      }

      return {
        ...base,
        type: 4,
        rows,
        cols,
        rowStride,
        colStride,
        values,
      };
    }
    default:
      throw new Error(`Unknown frame type: ${type}`);
  }
}

export function spikeBit(frame: Frame, step: number, neuron: number): boolean {
  if (frame.type !== 1) {
    return false;
  }
  const relStep = step - frame.step0;
  if (relStep < 0 || relStep >= frame.nsteps) {
    return false;
  }
  const bytesPerStep = Math.ceil(frame.payload.byteLength / frame.nsteps);
  const byteIndex = Math.floor(neuron / 8);
  const bitIndex = neuron % 8;
  const offset = relStep * bytesPerStep + byteIndex;
  if (offset >= frame.payload.byteLength) {
    return false;
  }
  const byte = frame.payload[offset];
  return (byte & (1 << bitIndex)) !== 0;
}
