interface ArrayEditorProps {
  name: string;
  value: unknown;
  onChange: (val: unknown[]) => void;
  itemType?: string;
}

export function ArrayEditor({ name, value, onChange, itemType = 'number' }: ArrayEditorProps) {
  const items = Array.isArray(value) ? value : [];

  const addItem = () => {
    const defaultVal = itemType === 'number' || itemType === 'integer' ? 0 : '';
    onChange([...items, defaultVal]);
  };

  const removeItem = (index: number) => {
    onChange(items.filter((_, i) => i !== index));
  };

  const updateItem = (index: number, val: unknown) => {
    const next = [...items];
    next[index] = val;
    onChange(next);
  };

  return (
    <div className="array-editor" aria-label={`${name} array editor`} style={{ margin: '8px 0' }}>
      {items.map((item, index) => (
        <div key={index} className="array-item-row" style={{ display: 'flex', gap: '8px', marginBottom: '6px' }}>
          <input
            type={itemType === 'number' || itemType === 'integer' ? 'number' : 'text'}
            value={String(item ?? '')}
            onChange={e => {
              const raw = e.target.value;
              updateItem(index, itemType === 'number' || itemType === 'integer' ? (raw === '' ? '' : Number(raw)) : raw);
            }}
            placeholder={`Item ${index + 1}`}
            style={{ flex: 1, padding: '8px', border: '1px solid var(--line)', borderRadius: '6px', background: 'var(--bg)', font: '14px var(--font-mono)' }}
          />
          <button
            type="button"
            onClick={() => removeItem(index)}
            aria-label={`Remove item ${index + 1}`}
            style={{ padding: '6px 10px', background: 'transparent', border: '1px solid var(--line)', borderRadius: '6px', cursor: 'pointer' }}
          >
            ✕
          </button>
        </div>
      ))}
      <button
        type="button"
        className="add-item-btn"
        onClick={addItem}
        style={{ padding: '6px 12px', background: 'var(--bg)', border: '1px solid var(--line)', borderRadius: '6px', fontSize: '12px', cursor: 'pointer' }}
      >
        + Add item
      </button>
    </div>
  );
}
