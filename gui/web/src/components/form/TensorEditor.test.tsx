import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, test, vi } from 'vitest';
import { TensorEditor } from './TensorEditor';

test('TensorEditor handles scalar mode without emitting NaN', () => {
  const change = vi.fn();
  render(<TensorEditor value={10} onChange={change} />);

  const scalarInput = screen.getByPlaceholderText('Enter scalar or value');
  expect(scalarInput).toHaveValue('10');

  fireEvent.change(scalarInput, { target: { value: 'abc' } });
  expect(change).toHaveBeenCalledWith('abc');

  fireEvent.change(scalarInput, { target: { value: '' } });
  expect(change).toHaveBeenCalledWith('');
});

test('TensorEditor switches between scalar, init, and tensor modes and updates values', async () => {
  const change = vi.fn();
  render(<TensorEditor value={10} onChange={change} />);

  // Switch to init mode
  await userEvent.click(screen.getByRole('button', { name: /\$init \(Generator\)/i }));
  expect(change).toHaveBeenCalledWith({ $init: 'uniform', low: 0, high: 1 });

  // Switch to tensor mode
  await userEvent.click(screen.getByRole('button', { name: /\$tensor/i }));
  expect(change).toHaveBeenCalledWith({ $tensor: { shape: [1], data: [0] } });
});

test('TensorEditor supports keyboard navigation and operation', async () => {
  const change = vi.fn();
  render(<TensorEditor value={10} onChange={change} />);

  const initBtn = screen.getByRole('button', { name: /\$init \(Generator\)/i });
  initBtn.focus();
  expect(initBtn).toHaveFocus();
  await userEvent.keyboard('{Enter}');
  expect(change).toHaveBeenCalledWith({ $init: 'uniform', low: 0, high: 1 });
});
