import type { Parameter, Issue, SchemaType } from '../../schema';
import { getBaseType, isNullable, getEnumValues, isTensorOrInitValue } from './FormUtils';
import { EnumChips } from './EnumChips';
import { NullableToggle } from './NullableToggle';
import { ArrayEditor } from './ArrayEditor';
import { TensorEditor } from './TensorEditor';

interface ParameterFieldProps {
  parameter: Parameter;
  value: unknown;
  unit?: string | null;
  error?: Issue;
  onChange: (name: string, value: unknown) => void;
  index: number;
}

export function ParameterField({ parameter, value, unit, error, onChange, index }: ParameterFieldProps) {
  const { name, type, required } = parameter;
  const baseType = getBaseType(type);
  const nullable = isNullable(type);
  const enumVals = getEnumValues(type);
  const isEnum = Boolean(enumVals && enumVals.length > 0);
  const isArray = baseType === 'array';
  const isTensorInit = isTensorOrInitValue(value) || name.toLowerCase().includes('tensor') || name.toLowerCase().includes('weights') || (name.toLowerCase().includes('rates') && typeof value === 'object');

  const isNuled = nullable && (value === null || value === undefined);

  const numeric = baseType === 'number' || baseType === 'integer';

  const tAny = type as unknown as { items?: SchemaType | SchemaType[] };
  const itemSchema = Array.isArray(tAny.items) ? tAny.items[0] : tAny.items;
  const itemType = itemSchema ? getBaseType(itemSchema as SchemaType) : 'number';

  return (
    <div className="field" key={name} style={{ animationDelay: `${index * 35}ms` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <label htmlFor={name}>{name}</label>
        {unit && (
          <span className="unit-badge" style={{ font: '10px var(--font-mono)', padding: '2px 6px', borderRadius: '4px', background: 'var(--line)', color: 'var(--muted)' }}>
            {unit}
          </span>
        )}
      </div>

      <NullableToggle
        nullable={nullable}
        isNuled={isNuled}
        onToggle={nuled => onChange(name, nuled ? null : (numeric ? 0 : ''))}
        label={name}
      />

      {!isNuled && (
        <>
          {isEnum && enumVals ? (
            <EnumChips
              name={name}
              enumVals={enumVals}
              value={value}
              onChange={val => onChange(name, val)}
              required={required}
            />
          ) : isArray ? (
            <ArrayEditor
              name={name}
              value={value}
              onChange={val => onChange(name, val)}
              itemType={itemType}
            />
          ) : isTensorInit ? (
            <TensorEditor
              value={value}
              onChange={val => onChange(name, val)}
            />
          ) : (
            <input
              id={name}
              type={numeric ? 'number' : 'text'}
              step={baseType === 'integer' ? '1' : 'any'}
              required={required}
              value={String(value ?? '')}
              placeholder={nullable ? 'None · use model default' : 'Required'}
              aria-invalid={Boolean(error)}
              aria-describedby={error ? `${name}-error` : undefined}
              onChange={event => {
                const raw = event.target.value;
                const num = Number(raw);
                const val = raw === '' ? (nullable ? null : '') : numeric ? (isNaN(num) ? raw : num) : raw;
                onChange(name, val);
              }}
              style={{ width: '100%', margin: '7px 0', padding: '12px', border: '1px solid var(--line)', borderRadius: '8px', background: 'var(--bg)', font: '17px var(--font-mono)' }}
            />
          )}
        </>
      )}

      <small id={`${name}-error`} className="error">
        {error && `${error.message} ${error.fix}`}
      </small>
    </div>
  );
}
