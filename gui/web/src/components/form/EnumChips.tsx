interface EnumChipsProps {
  name: string;
  enumVals: (string | number)[];
  value: unknown;
  onChange: (val: unknown) => void;
  required: boolean;
}

export function EnumChips({ name, enumVals, value, onChange, required }: EnumChipsProps) {
  return (
    <div className="enum-chips" role="group" aria-label={`${name} options`} style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', margin: '6px 0' }}>
      {!required && (
        <button
          type="button"
          className={`chip ${value === null || value === undefined ? 'active' : ''}`}
          onClick={() => onChange(null)}
          aria-pressed={value === null || value === undefined}
          style={{
            padding: '6px 12px',
            borderRadius: '999px',
            border: '1px solid var(--line)',
            background: value === null || value === undefined ? 'var(--fg)' : 'var(--bg)',
            color: value === null || value === undefined ? 'var(--bg)' : 'var(--fg)',
            fontSize: '12px',
          }}
        >
          None
        </button>
      )}
      {enumVals.map(opt => {
        const active = value === opt;
        return (
          <button
            key={String(opt)}
            type="button"
            className={`chip ${active ? 'active' : ''}`}
            onClick={() => onChange(opt)}
            aria-pressed={active}
            style={{
              padding: '6px 12px',
              borderRadius: '999px',
              border: '1px solid var(--line)',
              background: active ? 'var(--fg)' : 'var(--bg)',
              color: active ? 'var(--bg)' : 'var(--fg)',
              fontSize: '12px',
            }}
          >
            {String(opt)}
          </button>
        );
      })}
    </div>
  );
}
