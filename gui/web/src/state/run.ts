import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { post } from '../api/client';
import { RunInfo, RunRequest, NetworkSpec, Issue } from '../api/types';
import { RunSocket, RunSocketOptions } from '../stream/socket';
import { Frame, SpikesFrame, VSampleFrame, ScalarFrame, WeightsFrame } from '../stream/frames';

interface GroupInfo {
  stream: number;
  name: string;
  size: number;
  inhibitory: boolean;
}

interface SynapseInfo {
  stream: number;
  name: string;
  src: string;
  dst: string;
  shape: [number, number];
}

interface ScalarSeries {
  id: string;
  name?: string;
  unit?: string;
  values: { step: number; value: number }[];
}

interface VSampleSeries {
  stream: number;
  group: string;
  indices: number[];
  values: { step: number; values: number[] }[];
}

interface WeightsMatrix {
  stream: number;
  rows: number;
  cols: number;
  rowStride: number;
  colStride: number;
  values: number[];
}

export interface RunState {
  runId: string | null;
  runInfo: RunInfo | null;
  status: string;
  step: number;
  steps: number;
  stepsPerS: number;
  groups: GroupInfo[];
  synapses: SynapseInfo[];
  scalars: Record<string, ScalarSeries>;
  vsample: Record<string, VSampleSeries>;
  weights: Record<string, WeightsMatrix>;
  spikeBuffer: { frame: SpikesFrame; timestamp: number }[];
  maxSpikeBufferSize: number;
  droppedCount: number;
  logs: { level: 'info' | 'warning' | 'error'; message: string; timestamp: number }[];
  issues: Issue[];
  artifacts: string[];
  error: { message: string; traceback: string } | null;
  schedule: unknown | null;
  isConnected: boolean;
  connectionError: string | null;
  socket: RunSocket | null;

  // Actions
  startRun: (spec: NetworkSpec, steps: number, options?: Partial<RunRequest>, socketOptions?: RunSocketOptions) => Promise<void>;
  pauseRun: () => Promise<void>;
  resumeRun: () => Promise<void>;
  cancelRun: () => Promise<void>;
  setPace: (mode: string, speed: number) => void;
  setSelect: (group: string, indices: number[]) => void;
  reset: () => void;
}

export const useRunStore = create<RunState>()(
  immer((set, get) => ({
    runId: null,
    runInfo: null,
    status: 'idle',
    step: 0,
    steps: 0,
    stepsPerS: 0,
    groups: [],
    synapses: [],
    scalars: {},
    vsample: {},
    weights: {},
    spikeBuffer: [],
    maxSpikeBufferSize: 500,
    droppedCount: 0,
    logs: [],
    issues: [],
    artifacts: [],
    error: null,
    schedule: null,
    isConnected: false,
    connectionError: null,
    socket: null,

    startRun: async (spec: NetworkSpec, steps: number, options = {}, socketOptions = {}) => {
      const existingSocket = get().socket;
      if (existingSocket) {
        existingSocket.close();
      }

      set((state) => {
        state.status = 'queued';
        state.connectionError = null;
        state.error = null;
        state.spikeBuffer = [];
        state.droppedCount = 0;
        state.logs = [];
        state.issues = [];
        state.scalars = {};
        state.vsample = {};
        state.weights = {};
      });

      try {
        const runReq: RunRequest = {
          spec,
          steps,
          ...options,
        };
        const runInfo = await post<RunInfo>('/api/runs', runReq);

        set((state) => {
          state.runId = runInfo.id;
          state.runInfo = runInfo;
          state.status = runInfo.status;
          state.steps = runInfo.steps;
          state.step = runInfo.step;
        });

        let finishedReceived = false;

        const socket = new RunSocket(runInfo.id, {
          ...socketOptions,
          onOpen: () => {
            set((state) => {
              state.isConnected = true;
            });
            socketOptions.onOpen?.();
          },
          onClose: (event) => {
            set((state) => {
              state.isConnected = false;
              state.socket = null;
              const isTerminal = ['completed', 'cancelled', 'failed'].includes(state.status);
              if (!finishedReceived && !isTerminal) {
                const msg = event.code === 1000
                  ? 'WebSocket closed before finished'
                  : `WebSocket closed unexpectedly (code ${event.code})`;
                state.status = 'failed';
                state.connectionError = msg;
                if (!state.error) {
                  state.error = {
                    message: msg,
                    traceback: event.reason || '',
                  };
                }
              }
            });
            socketOptions.onClose?.(event);
          },
          onError: (err) => {
            set((state) => {
              const errMsg = String(err);
              state.connectionError = errMsg;
              state.logs.push({
                level: 'error',
                message: errMsg,
                timestamp: Date.now(),
              });
            });
            socketOptions.onError?.(err);
          },
          onText: (msg: unknown) => {
            if (!msg || typeof msg !== 'object') return;
            const m = msg as Record<string, unknown>;
            if (typeof m.type !== 'string') return;
            set((state) => {
              switch (m.type) {
                case 'hello':
                  if (m.run && typeof m.run === 'object') {
                    state.runInfo = m.run as RunInfo;
                    state.status = (m.run as RunInfo).status;
                    state.step = (m.run as RunInfo).step;
                    state.steps = (m.run as RunInfo).steps;
                  }
                  break;
                case 'started':
                  if (Array.isArray(m.groups)) state.groups = m.groups as GroupInfo[];
                  if (Array.isArray(m.synapses)) state.synapses = m.synapses as SynapseInfo[];
                  if (m.schedule !== undefined) state.schedule = m.schedule;
                  if (Array.isArray(m.scalars)) {
                    for (const s of m.scalars as Array<{ id: string; name?: string; unit?: string }>) {
                      if (!state.scalars[s.id]) {
                        state.scalars[s.id] = { id: s.id, name: s.name, unit: s.unit, values: [] };
                      } else {
                        state.scalars[s.id].name = s.name;
                        state.scalars[s.id].unit = s.unit;
                      }
                    }
                  }
                  if (Array.isArray(m.vsample)) {
                    for (const v of m.vsample as Array<{ stream: number; group: string; indices: number[] }>) {
                      const key = String(v.stream);
                      state.vsample[key] = {
                        stream: v.stream,
                        group: v.group,
                        indices: v.indices,
                        values: state.vsample[key]?.values || [],
                      };
                    }
                  }
                  break;
                case 'status':
                  if (typeof m.status === 'string') state.status = m.status;
                  if (typeof m.step === 'number') state.step = m.step;
                  break;
                case 'progress':
                  if (typeof m.step === 'number') state.step = m.step;
                  if (typeof m.steps === 'number') state.steps = m.steps;
                  if (typeof m.steps_per_s === 'number') state.stepsPerS = m.steps_per_s;
                  break;
                case 'log':
                  state.logs.push({
                    level: (typeof m.level === 'string' ? m.level : 'info') as 'info' | 'warning' | 'error',
                    message: typeof m.message === 'string' ? m.message : '',
                    timestamp: Date.now(),
                  });
                  break;
                case 'issue':
                  if (m.issue && typeof m.issue === 'object') {
                    state.issues.push(m.issue as Issue);
                  }
                  break;
                case 'dropped':
                  if (typeof m.frames === 'number') {
                    state.droppedCount += m.frames;
                  }
                  break;
                case 'finished':
                  finishedReceived = true;
                  if (typeof m.status === 'string') state.status = m.status;
                  if (typeof m.step === 'number') state.step = m.step;
                  if (m.error && typeof m.error === 'object') {
                    state.error = m.error as { message: string; traceback: string };
                    state.status = 'failed';
                  }
                  if (Array.isArray(m.artifacts)) {
                    state.artifacts = m.artifacts as string[];
                  }
                  break;
              }
            });
            socketOptions.onText?.(msg);
          },
          onFrame: (frame: Frame) => {
            set((state) => {
              switch (frame.type) {
                case 1: {
                  state.spikeBuffer.push({ frame: frame as SpikesFrame, timestamp: Date.now() });
                  if (state.spikeBuffer.length > state.maxSpikeBufferSize) {
                    state.spikeBuffer.shift();
                  }
                  break;
                }
                case 2: {
                  const vFrame = frame as VSampleFrame;
                  const key = String(vFrame.stream);
                  if (!state.vsample[key]) {
                    state.vsample[key] = {
                      stream: vFrame.stream,
                      group: `group_${vFrame.stream}`,
                      indices: vFrame.indices,
                      values: [],
                    };
                  }
                  for (let s = 0; s < vFrame.nsteps; s++) {
                    const stepNum = vFrame.step0 + s;
                    const valRow = vFrame.values[s] || [];
                    state.vsample[key].values.push({ step: stepNum, values: valRow });
                  }
                  break;
                }
                case 3: {
                  const sFrame = frame as ScalarFrame;
                  const key = String(sFrame.stream);
                  if (!state.scalars[key]) {
                    state.scalars[key] = { id: key, values: [] };
                  }
                  for (let s = 0; s < sFrame.nsteps; s++) {
                    const stepNum = sFrame.step0 + s;
                    const val = sFrame.values[s];
                    state.scalars[key].values.push({ step: stepNum, value: val });
                  }
                  break;
                }
                case 4: {
                  const wFrame = frame as WeightsFrame;
                  const key = String(wFrame.stream);
                  state.weights[key] = {
                    stream: wFrame.stream,
                    rows: wFrame.rows,
                    cols: wFrame.cols,
                    rowStride: wFrame.rowStride,
                    colStride: wFrame.colStride,
                    values: wFrame.values,
                  };
                  break;
                }
              }
            });
            socketOptions.onFrame?.(frame);
          },
        });

        set((state) => {
          state.socket = socket;
        });
      } catch (err: unknown) {
        const errorObj = err as Error;
        const msg = errorObj.message || 'Failed to start run';
        set((state) => {
          state.status = 'failed';
          state.connectionError = msg;
          state.error = {
            message: msg,
            traceback: '',
          };
        });
        throw err;
      }
    },

    pauseRun: async () => {
      const { runId } = get();
      if (!runId) return;
      try {
        const runInfo = await post<RunInfo>(`/api/runs/${runId}/pause`);
        set((state) => {
          state.runInfo = runInfo;
          state.status = runInfo.status;
        });
      } catch (err: unknown) {
        const errorObj = err as Error;
        set((state) => {
          state.connectionError = errorObj.message || 'Failed to pause run';
        });
        throw err;
      }
    },

    resumeRun: async () => {
      const { runId } = get();
      if (!runId) return;
      try {
        const runInfo = await post<RunInfo>(`/api/runs/${runId}/resume`);
        set((state) => {
          state.runInfo = runInfo;
          state.status = runInfo.status;
        });
      } catch (err: unknown) {
        const errorObj = err as Error;
        set((state) => {
          state.connectionError = errorObj.message || 'Failed to resume run';
        });
        throw err;
      }
    },

    cancelRun: async () => {
      const { runId } = get();
      if (!runId) return;
      try {
        const runInfo = await post<RunInfo>(`/api/runs/${runId}/cancel`);
        set((state) => {
          state.runInfo = runInfo;
          state.status = runInfo.status;
        });
      } catch (err: unknown) {
        const errorObj = err as Error;
        set((state) => {
          state.connectionError = errorObj.message || 'Failed to cancel run';
        });
        throw err;
      }
    },

    setPace: (mode: string, speed: number) => {
      const { socket } = get();
      if (socket) {
        socket.sendPace(mode, speed);
      }
    },

    setSelect: (group: string, indices: number[]) => {
      const { socket } = get();
      if (socket) {
        socket.sendSelect(group, indices);
      }
    },

    reset: () => {
      const { socket } = get();
      if (socket) {
        socket.close();
      }
      set((state) => {
        state.runId = null;
        state.runInfo = null;
        state.status = 'idle';
        state.step = 0;
        state.steps = 0;
        state.stepsPerS = 0;
        state.groups = [];
        state.synapses = [];
        state.scalars = {};
        state.vsample = {};
        state.weights = {};
        state.spikeBuffer = [];
        state.droppedCount = 0;
        state.logs = [];
        state.issues = [];
        state.artifacts = [];
        state.error = null;
        state.schedule = null;
        state.isConnected = false;
        state.connectionError = null;
        state.socket = null;
      });
    },
  }))
);
