import { render, screen, waitFor } from '@testing-library/react';
import { expect, test, vi, beforeEach, afterEach } from 'vitest';
import Forms from './index';

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

test('renders forms mode with server schema success path', async () => {
  const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      behaviors: [
        {
          name: 'ServerBehavior',
          summary: 'Fetched from server schema.',
          params: [{ name: 'param1', type: { type: 'number' }, required: true, default: 42 }],
        },
      ],
    }),
  }));

  render(<Forms />);

  await waitFor(() => {
    expect(screen.getAllByText('ServerBehavior')[0]).toBeInTheDocument();
  });

  expect(screen.queryByText('fixture descriptors, server schema unavailable')).not.toBeInTheDocument();

  expect(consoleErrorSpy).not.toHaveBeenCalled();
  consoleErrorSpy.mockRestore();
});

test('renders forms mode with server schema failure path and fallback banner', async () => {
  const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Network error')));

  render(<Forms />);

  await waitFor(() => {
    expect(screen.getByRole('alert')).toHaveTextContent('fixture descriptors, server schema unavailable');
  });

  expect(screen.getAllByText('LIF')[0]).toBeInTheDocument();

  expect(consoleErrorSpy).not.toHaveBeenCalled();
  consoleErrorSpy.mockRestore();
});
