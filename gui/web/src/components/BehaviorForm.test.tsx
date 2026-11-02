import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, test, vi } from 'vitest';
import { BehaviorForm } from './BehaviorForm';

test('renders server fields, including nullable defaults, without hardcoded parameter names', async () => {
  const change = vi.fn();
  render(<BehaviorForm descriptor={{name: 'LIF', summary: 'Membrane', params: [
    {name: 'novel_parameter', type: {type: 'number'}, required: true},
    {name: 'v_init', type: {anyOf: [{type: 'number'}, {type: 'null'}]}, required: false, default: null},
  ]}} values={{novel_parameter: '', v_init: null}} onChange={change} errors={[]} />);
  await userEvent.type(screen.getByLabelText('novel_parameter'), '7');
  expect(change).toHaveBeenCalledWith('novel_parameter', 7);
  expect(screen.getByLabelText('v_init')).toHaveDisplayValue('');
});

test('associates server errors with the named field', () => {
  render(<BehaviorForm descriptor={{name: 'LIF', summary: '', params: [
    {name: 'tau', type: {type: 'number'}, required: true},
  ]}} values={{tau: 0}} onChange={() => {}} errors={[
    {path: '/layers/cortex/groups/E/behaviors/0/params/tau', message: 'tau = 0', fix: 'Enter above zero.'},
  ]} />);
  expect(screen.getByLabelText('tau')).toHaveAttribute('aria-invalid', 'true');
  expect(screen.getByText('tau = 0 Enter above zero.')).toBeVisible();
});
