import { expect, test, vi, beforeEach } from 'vitest';
import { useRunStore } from './run';
import * as apiClient from '../api/client';

class MockWebSocket {
  public static instances: MockWebSocket[] = [];
  public url: string;
  public readyState: number = 0;
  public binaryType: BinaryType = 'blob';
  public sentData: string[] = [];

  public onopen: (() => void) | null = null;
  public onclose: ((event: { code: number; reason: string; wasClean: boolean }) => void) | null = null;
  public onerror: ((error: unknown) => void) | null = null;
  public onmessage: ((event: { data: unknown }) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  public send(data: string) {
    this.sentData.push(data);
  }

  public close(code = 1000, reason = 'Normal closure') {
    this.readyState = 3; // CLOSED
    this.onclose?.({ code, reason, wasClean: code === 1000 });
  }

  public simulateOpen() {
    this.readyState = 1; // OPEN
    this.onopen?.();
  }

  public simulateTextMessage(msg: object | string) {
    const raw = typeof msg === 'string' ? msg : JSON.stringify(msg);
    this.onmessage?.({ data: raw });
  }

  public simulateBinaryMessage(buffer: ArrayBuffer) {
    this.onmessage?.({ data: buffer });
  }
}

vi.mock('../api/client', () => ({
  post: vi.fn(),
}));

beforeEach(() => {
  MockWebSocket.instances = [];
  useRunStore.getState().reset();
  vi.clearAllMocks();
});

test('startRun initializes run and processes stream events correctly', async () => {
  const mockRunInfo = {
    id: 'r_1234567890abcdef',
    project_id: 'p_123',
    status: 'queued',
    step: 0,
    steps: 100,
    created_at: new Date().toISOString(),
    started_at: null,
    finished_at: null,
    error: null,
    provenance: {},
  };

  vi.mocked(apiClient.post).mockResolvedValueOnce(mockRunInfo);

  const spec = { version: 2 };
  const startPromise = useRunStore.getState().startRun(spec, 100, {}, {
    WebSocketImpl: MockWebSocket as unknown as typeof WebSocket,
  });

  await vi.waitFor(() => {
    expect(MockWebSocket.instances.length).toBe(1);
  });

  const mockWs = MockWebSocket.instances[0];
  mockWs.simulateOpen();

  await startPromise;

  expect(useRunStore.getState().runId).toBe('r_1234567890abcdef');
  expect(useRunStore.getState().status).toBe('queued');
  expect(useRunStore.getState().isConnected).toBe(true);

  // (f) Late-joiner sequence hello then started (no earlier frames) fills groups/scalars
  mockWs.simulateTextMessage({ type: 'hello', seq: 1, run: { ...mockRunInfo, status: 'building' } });
  expect(useRunStore.getState().status).toBe('building');

  mockWs.simulateTextMessage({
    type: 'started',
    seq: 2,
    groups: [{ stream: 0, name: 'exc', size: 100, inhibitory: false }],
    scalars: [{ id: 'rate:exc', name: 'Rate', unit: '1/step' }],
  });
  expect(useRunStore.getState().groups.length).toBe(1);
  expect(useRunStore.getState().scalars['rate:exc']).toBeDefined();

  mockWs.simulateTextMessage({ type: 'progress', seq: 3, step: 50, steps: 100, steps_per_s: 250.0 });
  expect(useRunStore.getState().step).toBe(50);
  expect(useRunStore.getState().stepsPerS).toBe(250.0);

  mockWs.simulateTextMessage({ type: 'log', seq: 4, level: 'info', message: 'Simulation running' });
  expect(useRunStore.getState().logs.length).toBe(1);
  expect(useRunStore.getState().logs[0].message).toBe('Simulation running');

  mockWs.simulateTextMessage({ type: 'issue', seq: 5, issue: { path: '/net', message: 'Warning', fix: 'Check' } });
  expect(useRunStore.getState().issues.length).toBe(1);

  // (h) dropped event increments dropped counter by frames
  mockWs.simulateTextMessage({ type: 'dropped', seq: 6, frames: 3, from_seq: 4 });
  expect(useRunStore.getState().droppedCount).toBe(3);

  mockWs.simulateTextMessage({ type: 'finished', seq: 7, status: 'completed', step: 100, error: null, artifacts: ['checkpoint.pt'] });
  expect(useRunStore.getState().status).toBe('completed');
  expect(useRunStore.getState().artifacts).toContain('checkpoint.pt');
});

test('pause, resume, and cancel actions invoke correct endpoints', async () => {
  const mockRunInfo = {
    id: 'r_1234567890abcdef',
    project_id: 'p_123',
    status: 'running',
    step: 10,
    steps: 100,
    created_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
    finished_at: null,
    error: null,
    provenance: {},
  };

  useRunStore.setState({ runId: 'r_1234567890abcdef', runInfo: mockRunInfo });

  vi.mocked(apiClient.post).mockResolvedValueOnce({ ...mockRunInfo, status: 'paused' });
  await useRunStore.getState().pauseRun();
  expect(apiClient.post).toHaveBeenCalledWith('/api/runs/r_1234567890abcdef/pause');
  expect(useRunStore.getState().status).toBe('paused');

  vi.mocked(apiClient.post).mockResolvedValueOnce({ ...mockRunInfo, status: 'running' });
  await useRunStore.getState().resumeRun();
  expect(apiClient.post).toHaveBeenCalledWith('/api/runs/r_1234567890abcdef/resume');
  expect(useRunStore.getState().status).toBe('running');

  vi.mocked(apiClient.post).mockResolvedValueOnce({ ...mockRunInfo, status: 'cancelled' });
  await useRunStore.getState().cancelRun();
  expect(apiClient.post).toHaveBeenCalledWith('/api/runs/r_1234567890abcdef/cancel');
  expect(useRunStore.getState().status).toBe('cancelled');
});

test('(a) unexpected close (code 1006) sets connectionError and status error', async () => {
  const mockRunInfo = {
    id: 'r_1234567890abcdef',
    project_id: 'p_123',
    status: 'running',
    step: 10,
    steps: 100,
    created_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
    finished_at: null,
    error: null,
    provenance: {},
  };

  vi.mocked(apiClient.post).mockResolvedValueOnce(mockRunInfo);

  await useRunStore.getState().startRun({ version: 2 }, 100, {}, {
    WebSocketImpl: MockWebSocket as unknown as typeof WebSocket,
  });

  const mockWs = MockWebSocket.instances[0];
  mockWs.simulateOpen();

  mockWs.close(1006, 'Abnormal closure');

  expect(useRunStore.getState().status).toBe('failed');
  expect(useRunStore.getState().connectionError).toContain('1006');
});

test('(b) close 1000 after finished is not an error', async () => {
  const mockRunInfo = {
    id: 'r_1234567890abcdef',
    project_id: 'p_123',
    status: 'running',
    step: 10,
    steps: 100,
    created_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
    finished_at: null,
    error: null,
    provenance: {},
  };

  vi.mocked(apiClient.post).mockResolvedValueOnce(mockRunInfo);

  await useRunStore.getState().startRun({ version: 2 }, 100, {}, {
    WebSocketImpl: MockWebSocket as unknown as typeof WebSocket,
  });

  const mockWs = MockWebSocket.instances[0];
  mockWs.simulateOpen();

  mockWs.simulateTextMessage({ type: 'finished', seq: 1, status: 'completed', step: 100, error: null, artifacts: [] });
  expect(useRunStore.getState().status).toBe('completed');

  mockWs.close(1000, 'Normal closure');
  expect(useRunStore.getState().status).toBe('completed');
  expect(useRunStore.getState().connectionError).toBeNull();
});

test('(c) close 1000 before finished is an error', async () => {
  const mockRunInfo = {
    id: 'r_1234567890abcdef',
    project_id: 'p_123',
    status: 'running',
    step: 10,
    steps: 100,
    created_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
    finished_at: null,
    error: null,
    provenance: {},
  };

  vi.mocked(apiClient.post).mockResolvedValueOnce(mockRunInfo);

  await useRunStore.getState().startRun({ version: 2 }, 100, {}, {
    WebSocketImpl: MockWebSocket as unknown as typeof WebSocket,
  });

  const mockWs = MockWebSocket.instances[0];
  mockWs.simulateOpen();

  mockWs.close(1000, 'Premature close');
  expect(useRunStore.getState().status).toBe('failed');
  expect(useRunStore.getState().connectionError).toContain('closed before finished');
});

test('(d) spikeBuffer never exceeds maxSpikeBufferSize after pushing size+5 frames', async () => {
  const mockRunInfo = {
    id: 'r_1234567890abcdef',
    project_id: 'p_123',
    status: 'running',
    step: 0,
    steps: 100,
    created_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
    finished_at: null,
    error: null,
    provenance: {},
  };

  vi.mocked(apiClient.post).mockResolvedValueOnce(mockRunInfo);

  await useRunStore.getState().startRun({ version: 2 }, 100, {}, {
    WebSocketImpl: MockWebSocket as unknown as typeof WebSocket,
  });

  const mockWs = MockWebSocket.instances[0];
  mockWs.simulateOpen();

  useRunStore.setState({ maxSpikeBufferSize: 5 });

  for (let i = 0; i < 10; i++) {
    const buffer = new ArrayBuffer(16 + 1);
    const view = new DataView(buffer);
    view.setUint8(0, 1); // type 1 (spikes)
    view.setUint8(1, 0); // flags
    view.setUint16(2, 0, true); // stream
    view.setUint32(4, i + 1, true); // seq
    view.setUint32(8, i * 10, true); // step0
    view.setUint16(12, 1, true); // nsteps
    view.setUint16(14, 0, true); // reserved
    view.setUint8(16, 0xFF); // payload

    mockWs.simulateBinaryMessage(buffer);
  }

  expect(useRunStore.getState().spikeBuffer.length).toBe(5);
  expect(useRunStore.getState().spikeBuffer[0].frame.seq).toBe(6);
  expect(useRunStore.getState().spikeBuffer[4].frame.seq).toBe(10);
});

test('(g) malformed JSON message does not throw and adds an error log entry', async () => {
  const mockRunInfo = {
    id: 'r_1234567890abcdef',
    project_id: 'p_123',
    status: 'running',
    step: 10,
    steps: 100,
    created_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
    finished_at: null,
    error: null,
    provenance: {},
  };

  vi.mocked(apiClient.post).mockResolvedValueOnce(mockRunInfo);

  await useRunStore.getState().startRun({ version: 2 }, 100, {}, {
    WebSocketImpl: MockWebSocket as unknown as typeof WebSocket,
  });

  const mockWs = MockWebSocket.instances[0];
  mockWs.simulateOpen();

  mockWs.simulateTextMessage('invalid { json');

  expect(useRunStore.getState().logs.some(l => l.level === 'error')).toBe(true);
});
