import { useEffect, useState } from 'react';
import type { Issue } from '../../schema';
import { ParameterField } from '../../components/form/ParameterField';
import { useReducedMotion } from '../../design/motion';

interface CatalogDescriptor {
  name: string;
  summary: string;
  params: Array<{
    name: string;
    type: unknown;
    required: boolean;
    default?: unknown;
  }>;
  meta?: {
    host?: string;
    category?: string;
    requires?: string[];
    params?: Record<string, { unit?: string | null }>;
  };
}

const FIXTURE_DESCRIPTORS: CatalogDescriptor[] = [
  {
    name: 'LIF',
    summary: 'Leaky integrate-and-fire neuron model with configurable membrane parameters.',
    params: [
      { name: 'tau', type: { type: 'number' }, required: true, default: 20 },
      { name: 'threshold', type: { type: 'number' }, required: true, default: -50 },
      { name: 'v_reset', type: { type: 'number' }, required: true, default: -70 },
      { name: 'v_rest', type: { type: 'number' }, required: true, default: -70 },
      { name: 'resistance', type: { type: 'number' }, required: true, default: 10 },
      { name: 'v_init', type: { anyOf: [{ type: 'number' }, { type: 'null' }] }, required: false, default: null },
    ],
    meta: {
      host: 'neurons',
      category: 'neuron_model',
      requires: ['Fire'],
      params: {
        tau: { unit: 'ms' },
        threshold: { unit: 'mV' },
        v_reset: { unit: 'mV' },
        v_rest: { unit: 'mV' },
        resistance: { unit: 'MOhm' },
        v_init: { unit: 'mV' },
      },
    },
  },
  {
    name: 'WeightInit',
    summary: 'Initialize synaptic weights with distribution or constants.',
    params: [
      { name: 'mode', type: { enum: ['uniform', 'normal', 'constant'] }, required: true, default: 'uniform' },
      { name: 'scale', type: { type: 'number' }, required: false, default: 1.0 },
      { name: 'weights', type: { anyOf: [{ type: 'array', items: { type: 'number' } }, { type: 'null' }] }, required: false, default: null },
    ],
    meta: {
      host: 'synapse',
      category: 'connectivity',
      requires: [],
      params: {
        mode: { unit: null },
        scale: { unit: null },
        weights: { unit: null },
      },
    },
  },
];

export default function Forms() {
  const [descriptors, setDescriptors] = useState<CatalogDescriptor[]>(FIXTURE_DESCRIPTORS);
  const [selectedName, setSelectedName] = useState<string>('LIF');
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [errors, setErrors] = useState<Issue[]>([]);
  const [status, setStatus] = useState('Loading catalog behaviors…');
  const [fallback, setFallback] = useState<boolean>(false);
  const [reduced, setReduced] = useReducedMotion();

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/schema', { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error(`Schema request failed (${response.status})`);
        const data: { behaviors: CatalogDescriptor[] } = await response.json();
        if (data.behaviors && data.behaviors.length > 0) {
          setDescriptors(data.behaviors);
          setSelectedName(data.behaviors[0].name);
          setStatus(`Loaded ${data.behaviors.length} catalog behaviors successfully.`);
          setFallback(false);
        }
      })
      .catch(() => {
        setFallback(true);
        setStatus('fixture descriptors, server schema unavailable');
      });
    return () => controller.abort();
  }, []);

  const currentDescriptor = descriptors.find(d => d.name === selectedName) || descriptors[0];

  useEffect(() => {
    if (currentDescriptor) {
      const initialValues: Record<string, unknown> = {};
      currentDescriptor.params.forEach(p => {
        initialValues[p.name] = p.default !== undefined ? p.default : (p.required ? '' : null);
      });
      setValues(initialValues);
      setErrors([]);
    }
  }, [currentDescriptor]);

  const handleChange = (name: string, val: unknown) => {
    setValues(prev => ({ ...prev, [name]: val }));
    setErrors([]);
    setStatus(`Updated parameter: ${name}`);
  };

  return (
    <>
      <header>
        <div className="brand">
          neuroSush Studio <span>/ forms</span>
        </div>
        <button onClick={() => setReduced(!reduced)} aria-pressed={reduced}>
          Reduce motion
        </button>
      </header>
      <main>
        <div className="kicker">05 / Catalog Behavior Forms</div>
        <h1>Interactive <em>Forms</em> & Widgets</h1>
        <p>Render auto-generated forms from schema descriptors supporting numbers with units, enum chips, nullable toggles, array editors, and tensor/init generators.</p>

        {fallback && (
          <div role="alert" style={{ background: 'var(--error)', color: '#fff', padding: '12px 18px', borderRadius: '12px', margin: '24px 0', fontSize: '13px', fontWeight: 500 }}>
            fixture descriptors, server schema unavailable
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '32px', marginTop: '32px' }}>
          {/* Behavior Selector Sidebar */}
          <aside className="panel" style={{ padding: '16px', maxHeight: '70vh', overflowY: 'auto' }} aria-label="Behavior catalog list">
            <div className="kicker" style={{ marginBottom: '12px' }}>Catalog Behaviors ({descriptors.length})</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {descriptors.map(d => (
                <button
                  key={d.name}
                  type="button"
                  onClick={() => setSelectedName(d.name)}
                  aria-current={d.name === selectedName ? 'true' : undefined}
                  style={{
                    textAlign: 'left',
                    padding: '10px 12px',
                    borderRadius: '8px',
                    border: '1px solid var(--line)',
                    background: d.name === selectedName ? 'var(--fg)' : 'var(--bg)',
                    color: d.name === selectedName ? 'var(--bg)' : 'var(--fg)',
                    fontSize: '13px',
                    cursor: 'pointer',
                  }}
                >
                  <div style={{ fontWeight: 600 }}>{d.name}</div>
                  {d.meta?.category && <div style={{ fontSize: '10px', opacity: 0.8 }}>{d.meta.category}</div>}
                </button>
              ))}
            </div>
          </aside>

          {/* Form Panel */}
          <section className="panel" aria-label="Behavior parameter form">
            {currentDescriptor && (
              <>
                <div className="kicker">{currentDescriptor.meta?.host || 'behavior'} / {currentDescriptor.meta?.category || 'general'}</div>
                <h2>{currentDescriptor.name}</h2>
                <p>{currentDescriptor.summary}</p>

                {currentDescriptor.meta?.requires && currentDescriptor.meta.requires.length > 0 && (
                  <div style={{ margin: '10px 0', fontSize: '11px', color: 'var(--muted)' }}>
                    Requires: {currentDescriptor.meta.requires.join(', ')}
                  </div>
                )}

                <form onSubmit={e => e.preventDefault()} noValidate>
                  <fieldset className="fields" style={{ border: 0, padding: 0, margin: '20px 0' }}>
                    <legend style={{ font: '11px var(--font-mono)', color: 'var(--muted)', paddingBottom: '16px' }}>
                      {currentDescriptor.name} Parameters ({currentDescriptor.params.length})
                    </legend>
                    {currentDescriptor.params.map((param, index) => {
                      const unit = currentDescriptor.meta?.params?.[param.name]?.unit;
                      const issue = errors.find(err => err.path.endsWith(`/params/${param.name}`));
                      return (
                        <ParameterField
                          key={param.name}
                          parameter={param as unknown as import('../../schema').Parameter}
                          value={values[param.name]}
                          unit={unit}
                          error={issue}
                          onChange={handleChange}
                          index={index}
                        />
                      );
                    })}
                  </fieldset>
                </form>

                {errors.filter(err => !err.path.includes('/params/')).map(err => (
                  <div key={err.path} role="alert" className="error" style={{ margin: '10px 0', padding: '8px', background: 'var(--line)', borderRadius: '4px' }}>
                    {err.path}: {err.message} — {err.fix}
                  </div>
                ))}

                <div className="status" role="status">{status}</div>

                <div style={{ marginTop: '20px' }}>
                  <div className="kicker" style={{ marginBottom: '8px' }}>Current Values JSON</div>
                  <pre>{JSON.stringify(values, null, 2)}</pre>
                </div>
              </>
            )}
          </section>
        </div>
      </main>
    </>
  );
}
