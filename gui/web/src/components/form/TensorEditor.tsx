import { useState } from 'react';

interface TensorEditorProps {
  name?: string;
  value: unknown;
  onChange: (val: unknown) => void;
}

export function TensorEditor({ value, onChange }: TensorEditorProps) {
  const isObj = value && typeof value === 'object' && !Array.isArray(value);
  const objVal = isObj ? (value as Record<string, unknown>) : {};
  const hasInit = '$init' in objVal;
  const hasTensor = '$tensor' in objVal;

  const [mode, setMode] = useState<'scalar' | 'init' | 'tensor'>(
    hasInit ? 'init' : hasTensor ? 'tensor' : 'scalar'
  );

  const initType = hasInit ? String(objVal['$init']) : 'uniform';
  const [dist, setDist] = useState<string>(initType);
  const [params, setParams] = useState<Record<string, unknown>>(
    hasInit ? { ...objVal } : { '$init': 'uniform', low: 0, high: 1 }
  );

  const handleModeChange = (newMode: 'scalar' | 'init' | 'tensor') => {
    setMode(newMode);
    if (newMode === 'scalar') {
      onChange(0);
    } else if (newMode === 'init') {
      onChange({ $init: 'uniform', low: 0, high: 1 });
    } else if (newMode === 'tensor') {
      onChange({ $tensor: { shape: [1], data: [0] } });
    }
  };

  const updateParam = (k: string, v: unknown) => {
    const updated = { ...params, [k]: v };
    setParams(updated);
    onChange(updated);
  };

  return (
    <div className="tensor-editor" style={{ border: '1px solid var(--line)', borderRadius: '8px', padding: '10px', background: 'var(--bg)', margin: '8px 0' }}>
      <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
        <button
          type="button"
          onClick={() => handleModeChange('scalar')}
          style={{ padding: '4px 8px', fontSize: '11px', background: mode === 'scalar' ? 'var(--fg)' : 'transparent', color: mode === 'scalar' ? 'var(--bg)' : 'var(--fg)', border: '1px solid var(--line)', borderRadius: '4px' }}
        >
          Scalar / Value
        </button>
        <button
          type="button"
          onClick={() => handleModeChange('init')}
          style={{ padding: '4px 8px', fontSize: '11px', background: mode === 'init' ? 'var(--fg)' : 'transparent', color: mode === 'init' ? 'var(--bg)' : 'var(--fg)', border: '1px solid var(--line)', borderRadius: '4px' }}
        >
          $init (Generator)
        </button>
        <button
          type="button"
          onClick={() => handleModeChange('tensor')}
          style={{ padding: '4px 8px', fontSize: '11px', background: mode === 'tensor' ? 'var(--fg)' : 'transparent', color: mode === 'tensor' ? 'var(--bg)' : 'var(--fg)', border: '1px solid var(--line)', borderRadius: '4px' }}
        >
          $tensor
        </button>
      </div>

      {mode === 'scalar' && (
        <input
          type="text"
          value={String(value ?? '')}
          onChange={e => {
            const raw = e.target.value;
            const num = Number(raw);
            onChange(raw === '' ? '' : isNaN(num) ? raw : num);
          }}
          placeholder="Enter scalar or value"
          style={{ width: '100%', padding: '8px', borderRadius: '6px', border: '1px solid var(--line)', font: '14px var(--font-mono)' }}
        />
      )}

      {mode === 'init' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <label style={{ fontSize: '11px' }}>Distribution / Init type:</label>
          <select
            value={dist}
            onChange={e => {
              const d = e.target.value;
              setDist(d);
              const next = { '$init': d, ...(d === 'uniform' ? { low: 0, high: 1 } : d === 'normal' ? { mean: 0, std: 1 } : { value: 0 }) };
              setParams(next);
              onChange(next);
            }}
            style={{ padding: '6px', borderRadius: '6px', border: '1px solid var(--line)', font: '13px var(--font-mono)' }}
          >
            <option value="uniform">uniform</option>
            <option value="normal">normal</option>
            <option value="constant">constant</option>
          </select>

          {dist === 'uniform' && (
            <div style={{ display: 'flex', gap: '8px' }}>
              <div>
                <span style={{ fontSize: '10px' }}>low:</span>
                <input
                  type="number"
                  value={String(params['low'] ?? 0)}
                  onChange={e => updateParam('low', Number(e.target.value))}
                  style={{ width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid var(--line)' }}
                />
              </div>
              <div>
                <span style={{ fontSize: '10px' }}>high:</span>
                <input
                  type="number"
                  value={String(params['high'] ?? 1)}
                  onChange={e => updateParam('high', Number(e.target.value))}
                  style={{ width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid var(--line)' }}
                />
              </div>
            </div>
          )}

          {dist === 'normal' && (
            <div style={{ display: 'flex', gap: '8px' }}>
              <div>
                <span style={{ fontSize: '10px' }}>mean:</span>
                <input
                  type="number"
                  value={String(params['mean'] ?? 0)}
                  onChange={e => updateParam('mean', Number(e.target.value))}
                  style={{ width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid var(--line)' }}
                />
              </div>
              <div>
                <span style={{ fontSize: '10px' }}>std:</span>
                <input
                  type="number"
                  value={String(params['std'] ?? 1)}
                  onChange={e => updateParam('std', Number(e.target.value))}
                  style={{ width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid var(--line)' }}
                />
              </div>
            </div>
          )}

          {dist === 'constant' && (
            <div>
              <span style={{ fontSize: '10px' }}>value:</span>
              <input
                type="number"
                value={String(params['value'] ?? 0)}
                onChange={e => updateParam('value', Number(e.target.value))}
                style={{ width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid var(--line)' }}
              />
            </div>
          )}
        </div>
      )}

      {mode === 'tensor' && (
        <div>
          <span style={{ fontSize: '11px', display: 'block', marginBottom: '4px' }}>Tensor Spec JSON ($tensor):</span>
          <textarea
            rows={3}
            value={JSON.stringify(value, null, 2)}
            onChange={e => {
              try {
                const parsed = JSON.parse(e.target.value);
                onChange(parsed);
              } catch {
                // ignore invalid JSON while typing
              }
            }}
            style={{ width: '100%', padding: '6px', borderRadius: '6px', border: '1px solid var(--line)', font: '11px var(--font-mono)' }}
          />
        </div>
      )}
    </div>
  );
}
