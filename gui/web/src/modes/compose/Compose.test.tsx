import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, test, vi } from 'vitest';
import { Compose } from './Compose';

afterEach(() => { vi.unstubAllGlobals(); localStorage.clear(); });

test('loads LIF from the server, sends native spec, and displays server validation', async () => {
  const request = vi.fn().mockResolvedValueOnce({ ok: true, json: async () => ({ behaviors: [{
    name: 'LIF', summary: 'Membrane', params: [
      {name: 'tau', type: {type: 'number'}, required: true},
      {name: 'resistance', type: {type: 'number'}, required: false, default: 1},
    ],
  }] }) }).mockResolvedValueOnce({ok: true, json: async () => ({valid: false, errors: [
    {path: '/layers/cortex/groups/E/behaviors/0/params/tau', message: 'tau = 0', fix: 'Enter above zero.'},
  ]})});
  vi.stubGlobal('fetch', request);
  render(<Compose />);
  await userEvent.type(await screen.findByLabelText('tau'), '0');
  expect(screen.getByLabelText('resistance')).toHaveValue(1);
  await userEvent.click(screen.getByRole('button', {name: 'Validate spec'}));
  expect(await screen.findByText('tau = 0 Enter above zero.')).toBeVisible();
  const payload = JSON.parse(request.mock.calls[1][1].body);
  expect(payload.layers.cortex.groups.E.behaviors).toEqual([
    {kind: 'LIF', params: {tau: 0, resistance: 1}}, {kind: 'Fire', params: {}},
  ]);
});

test('schema failure surfaces a retry instruction and motion choice persists', async () => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Disconnected. Reload to retry.')));
  render(<Compose />);
  expect(await screen.findByRole('status')).toHaveTextContent('Disconnected. Reload to retry.');
  await userEvent.click(screen.getByRole('button', {name: 'Reduce motion'}));
  await waitFor(() => expect(localStorage.getItem('studio-reduced-motion')).toBe('true'));
  expect(document.documentElement.dataset.reducedMotion).toBe('true');
});
