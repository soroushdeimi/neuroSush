import { useEffect, useState } from 'react';
import { BehaviorForm } from '../../components/BehaviorForm';
import { useReducedMotion } from '../../design/motion';
import { columnSpec, type Descriptor, type Issue, type Values } from '../../schema';

export function Compose() {
  const [descriptor, setDescriptor] = useState<Descriptor>();
  const [values, setValues] = useState<Values>({});
  const [errors, setErrors] = useState<Issue[]>([]);
  const [status, setStatus] = useState('Loading the behavior registry…');
  const [pending, setPending] = useState(false);
  const [summary, setSummary] = useState(false);
  const [reduced, setReduced] = useReducedMotion();
  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/schema', { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error(`Schema request failed (${response.status}). Reload to retry.`);
      const data: { behaviors: Descriptor[] } = await response.json();
      const lif = data.behaviors.find(item => item.name === 'LIF');
      if (!lif) throw new Error('LIF is missing from the registry. Register LIF and reload.');
      setDescriptor(lif);
      setValues(Object.fromEntries(lif.params.map(parameter => [parameter.name, parameter.default ?? (parameter.required ? '' : null)])));
      setStatus('Enter the required parameters, then validate your spec.');
    }).catch((error: Error) => { if (!controller.signal.aborted) setStatus(error.message); });
    return () => controller.abort();
  }, []);
  async function validate() {
    setPending(true);
    setErrors([]);
    try {
      const response = await fetch('/api/validate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(columnSpec(values)) });
      if (!response.ok) throw new Error(`Validation failed (${response.status}). Retry the request.`);
      const result: { valid: boolean; errors: Issue[] } = await response.json();
      setErrors(result.errors);
      setStatus(result.valid ? 'Spec validated. Your LIF and Fire constructors are ready.' : 'Review the named values below.');
    } catch (error) { setStatus(error instanceof Error ? error.message : 'Request failed. Retry validation.'); }
    finally { setPending(false); }
  }
  return <>
    <header><div className="brand">neuroSush Studio <span>/ compose</span></div>
      <button onClick={() => setReduced(!reduced)} aria-pressed={reduced}>Reduce motion</button>
    </header>
    <main>
      <nav className="steps" aria-label="Composition stages">
        {['Populations', 'Model', 'Learning', 'Inputs', 'Record', 'Run'].map((step, index) => <span key={step} aria-current={index === 1 ? 'step' : undefined}>{String(index + 1).padStart(2, '0')} {step}</span>)}
      </nav>
      <div className="layout">
        <section><div className="kicker">02 / Neuron model</div><h1>A membrane.<br /><em>A beginning.</em></h1>
          <p>Start with a single neuron. Shape its membrane, choose its threshold, and compose a spec from the library’s own parameters.</p>
          <p>This first chapter prepares a LIF population. Learning, recording and execution are coming in later chapters.</p>
          <div className="step-number" aria-hidden="true">02</div>
        </section>
        <section className="panel" aria-label="Model configuration"><div className="kicker">Cortex E / one neuron</div>
          <h2>{descriptor?.name ?? 'Neuron model'} + Fire</h2><p>{descriptor?.summary}</p>
          {descriptor && <form onSubmit={event => { event.preventDefault(); void validate(); }} noValidate>
            <BehaviorForm descriptor={descriptor} values={values} errors={errors} onChange={(name, value) => {
              setValues(previous => ({ ...previous, [name]: value })); setErrors([]); setStatus('Parameters changed. Validate to check this spec.');
            }} />
            <button className="primary" disabled={pending}>{pending ? 'Validating…' : 'Validate spec'}<span aria-hidden="true">↗</span></button>
          </form>}
          <div className="status" role="status">{status}</div>
          {errors.filter(error => !error.path.includes('/params/')).map(error => <p className="error" key={error.path}>{error.path}: {error.message} {error.fix}</p>)}
          <button className="primary" aria-expanded={summary} onClick={() => setSummary(!summary)}>Your network <span aria-hidden="true">{summary ? '−' : '+'}</span></button>
          {summary && <div className="summary"><div className="kicker">Native ColumnSpec / JSON</div><pre>{JSON.stringify(columnSpec(values), null, 2)}</pre></div>}
        </section>
      </div>
    </main>
  </>;
}
