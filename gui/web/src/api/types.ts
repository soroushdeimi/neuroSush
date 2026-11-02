export interface Issue {
  path: string;
  message: string;
  fix: string;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    issues: Issue[];
  };
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    public message: string,
    public issues: Issue[] = []
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export interface BehaviorSpec {
  kind: string;
  params: Record<string, unknown>;
  enabled?: boolean;
}

export interface NetSettings {
  dt?: number;
  dtype?: 'float32' | 'float64' | 'float16' | 'bfloat16';
  device?: string;
  seed?: number;
  batch_size?: number | null;
  independent?: boolean;
  behaviors?: BehaviorSpec[];
}

export interface GroupSpec {
  shape: number | number[];
  behaviors: BehaviorSpec[];
  inhibitory: boolean;
  tags: string[];
}

export interface SynapseSpec {
  name?: string;
  src: string;
  dst: string;
  behaviors: BehaviorSpec[];
  compartment: 'proximal' | 'distal' | 'apical';
  tags?: string[];
}

export interface RecordingSpec {
  target: string;
  attributes: string[];
  interval?: number;
  max_steps: number;
}

export interface AssetSpec {
  sha256: string;
  format: 'idx' | 'csv' | 'pt' | 'png';
}

export interface NetworkSpec {
  version: number;
  network?: NetSettings;
  groups?: Record<string, GroupSpec>;
  synapses?: SynapseSpec[];
  columns?: Record<string, unknown>;
  recordings?: RecordingSpec[];
  assets?: Record<string, AssetSpec>;
}

export interface Project {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  revision: number;
  catalog_fingerprint: string;
  spec: NetworkSpec;
  layout: Record<string, { x: number; y: number }>;
}

export interface RunRequest {
  project_id?: string;
  spec: NetworkSpec;
  steps: number;
  chunk?: number;
  mode?: 'watch' | 'fast';
  speed?: number;
  stream?: {
    spikes?: string[] | 'all';
    vsample?: Record<string, number[]>;
    scalars?: string[];
    weights?: Record<string, number>;
  };
}

export interface RunInfo {
  id: string;
  project_id: string;
  status: string;
  step: number;
  steps: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: { message: string; traceback: string } | null;
  provenance: Record<string, unknown>;
}
