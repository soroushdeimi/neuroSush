import type { Descriptor, Issue, Values } from '../schema';

interface Props {
  descriptor: Descriptor;
  values: Values;
  errors: Issue[];
  onChange: (name: string, value: Values[string]) => void;
}

export function BehaviorForm({ descriptor, values, errors, onChange }: Props) {
  return <fieldset className="fields"><legend>{descriptor.name} parameters</legend>
    {descriptor.params.map((parameter, index) => {
      const { name, type } = parameter;
      const issue = errors.find(error => error.path.endsWith(`/params/${name}`));
      const numeric = type.type === 'number' || type.type === 'integer' || type.anyOf?.some(t => t.type === 'number');
      const nullable = type.anyOf?.some(t => t.type === 'null');
      return <div className="field" key={name} style={{ animationDelay: `${index * 35}ms` }}>
        <label htmlFor={name}>{name}</label>
        <input id={name} type={numeric ? 'number' : 'text'} step={type.type === 'integer' ? '1' : 'any'}
          required={parameter.required} value={String(values[name] ?? '')}
          placeholder={nullable ? 'None · use model default' : 'Required'}
          aria-invalid={Boolean(issue)} aria-describedby={issue ? `${name}-error` : undefined}
          onChange={event => {
            const raw = event.target.value;
            onChange(name, raw === '' ? (nullable ? null : '') : numeric ? Number(raw) : raw);
          }} />
        <small id={`${name}-error`} className="error">{issue && `${issue.message} ${issue.fix}`}</small>
      </div>;
    })}
  </fieldset>;
}
