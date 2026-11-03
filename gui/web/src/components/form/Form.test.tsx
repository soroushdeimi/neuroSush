import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, test, vi } from 'vitest';
import { ParameterField } from './ParameterField';
import type { SchemaType } from '../../schema';

test('renders number parameter with unit badge and prevents NaN on non-numeric input', () => {
  const change = vi.fn();
  render(
    <ParameterField
      parameter={{ name: 'tau', type: { type: 'number' }, required: true }}
      value={20}
      unit="ms"
      onChange={change}
      index={0}
    />
  );
  expect(screen.getByLabelText('tau')).toBeInTheDocument();
  expect(screen.getByText('ms')).toBeInTheDocument();
  expect(screen.getByLabelText('tau')).toHaveValue(20);

  const input = screen.getByLabelText('tau');
  fireEvent.change(input, { target: { value: 'abc' } });
  expect(change).not.toHaveBeenCalledWith(NaN);
});

test('renders enum parameter as interactive chips and emits chosen value', async () => {
  const change = vi.fn();
  render(
    <ParameterField
      parameter={{ name: 'mode', type: { enum: ['uniform', 'normal'] }, required: true }}
      value="uniform"
      onChange={change}
      index={0}
    />
  );
  expect(screen.getByRole('button', { name: 'uniform' })).toHaveAttribute('aria-pressed', 'true');
  await userEvent.click(screen.getByRole('button', { name: 'normal' }));
  expect(change).toHaveBeenCalledWith('mode', 'normal');
});

test('nullable toggled OFF emits null and toggled ON restores value', async () => {
  const change = vi.fn();
  const { rerender } = render(
    <ParameterField
      parameter={{ name: 'v_init', type: { anyOf: [{ type: 'number' }, { type: 'null' }] }, required: false }}
      value={-70}
      onChange={change}
      index={0}
    />
  );
  const checkbox = screen.getByRole('checkbox', { name: /Use default \(None\) for v_init/i });
  expect(checkbox).not.toBeChecked();

  await userEvent.click(checkbox);
  expect(change).toHaveBeenCalledWith('v_init', null);

  rerender(
    <ParameterField
      parameter={{ name: 'v_init', type: { anyOf: [{ type: 'number' }, { type: 'null' }] }, required: false }}
      value={null}
      onChange={change}
      index={0}
    />
  );
  expect(checkbox).toBeChecked();

  await userEvent.click(checkbox);
  expect(change).toHaveBeenCalledWith('v_init', 0);
});

test('array item removal and addition work correctly', async () => {
  const change = vi.fn();
  const arrayType: SchemaType = { type: 'array', items: { type: 'number' } } as unknown as SchemaType;
  render(
    <ParameterField
      parameter={{ name: 'weights', type: arrayType, required: false }}
      value={[1.5, 2.5]}
      onChange={change}
      index={0}
    />
  );
  expect(screen.getByDisplayValue('1.5')).toBeInTheDocument();
  expect(screen.getByDisplayValue('2.5')).toBeInTheDocument();

  const removeBtn = screen.getByLabelText('Remove item 1');
  await userEvent.click(removeBtn);
  expect(change).toHaveBeenCalledWith('weights', [2.5]);
});

test('renders anyOf descriptor parameter correctly', () => {
  const anyOfSchema: SchemaType = { anyOf: [{ type: 'string' }, { type: 'number' }] } as unknown as SchemaType;
  render(
    <ParameterField
      parameter={{ name: 'flexible_param', type: anyOfSchema, required: true }}
      value="test"
      onChange={() => {}}
      index={0}
    />
  );
  expect(screen.getByLabelText('flexible_param')).toBeInTheDocument();
  expect(screen.getByLabelText('flexible_param')).toHaveValue('test');
});

test('associates validation issues next to field and handles global issues without crashing', () => {
  render(
    <div>
      <ParameterField
        parameter={{ name: 'tau', type: { type: 'number' }, required: true }}
        value={0}
        error={{ path: '/params/tau', message: 'tau must be > 0', fix: 'Increase tau.' }}
        onChange={() => {}}
        index={0}
      />
      <div data-testid="global-error">
        {JSON.stringify({ path: '/global', message: 'Spec invalid', fix: 'Check network structure.' })}
      </div>
    </div>
  );
  expect(screen.getByLabelText('tau')).toHaveAttribute('aria-invalid', 'true');
  expect(screen.getByText('tau must be > 0 Increase tau.')).toBeVisible();
  expect(screen.getByTestId('global-error')).toBeVisible();
});
