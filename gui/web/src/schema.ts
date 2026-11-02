export interface SchemaType { type?: string; anyOf?: SchemaType[]; enum?: (string | number)[] }
export interface Parameter { name: string; type: SchemaType; required: boolean; default?: number | string | boolean | null }
export interface Descriptor { name: string; summary: string; params: Parameter[] }
export interface Issue { path: string; message: string; fix: string }
export type Values = Record<string, number | string | boolean | null>;

export function columnSpec(values: Values) {
  const params = Object.fromEntries(Object.entries(values).filter(([, value]) => value !== ''));
  return { layers: { cortex: { groups: { E: {
    shape: 1, inhibitory: false, tags: [], behaviors: [{ kind: 'LIF', params }, { kind: 'Fire', params: {} }],
  } }, synapses: [], inputs: null, outputs: null } }, synapses: [], inputs: {}, outputs: {} };
}
