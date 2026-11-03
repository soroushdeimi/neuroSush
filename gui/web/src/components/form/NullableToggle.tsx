interface NullableToggleProps {
  nullable: boolean;
  isNuled: boolean;
  onToggle: (nuled: boolean) => void;
  label: string;
}

export function NullableToggle({ nullable, isNuled, onToggle, label }: NullableToggleProps) {
  if (!nullable) return null;
  return (
    <div className="nullable-toggle" style={{ margin: '4px 0 8px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
      <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', color: 'var(--muted)' }}>
        <input
          type="checkbox"
          checked={isNuled}
          onChange={e => onToggle(e.target.checked)}
        />
        <span>Use default (None) for {label}</span>
      </label>
    </div>
  );
}
