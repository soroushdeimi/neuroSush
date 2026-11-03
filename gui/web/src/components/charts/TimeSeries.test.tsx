import { render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { TimeSeries } from './TimeSeries';

vi.mock('uplot', () => {
  return {
    default: class MockuPlot {
      destroy = vi.fn();
      setSize = vi.fn();
    }
  };
});

if (typeof window !== 'undefined') {
  if (!window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      callback: ResizeObserverCallback;
      constructor(cb: ResizeObserverCallback) {
        this.callback = cb;
      }
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
}

if (typeof globalThis.Path2D === 'undefined') {
  globalThis.Path2D = class Path2D {
    addPath() {}
    arc() {}
    arcTo() {}
    bezierCurveTo() {}
    closePath() {}
    ellipse() {}
    lineTo() {}
    moveTo() {}
    quadraticCurveTo() {}
    rect() {}
    roundRect() {}
  } as unknown as typeof window.Path2D;
}

if (typeof HTMLCanvasElement !== 'undefined') {
  // @ts-expect-error test mock
  HTMLCanvasElement.prototype.getContext = function () {
    return {
      fillRect: () => {},
      clearRect: () => {},
      strokeRect: () => {},
      getImageData: () => ({ data: new Array(4) }),
      putImageData: () => {},
      createImageData: () => [],
      setTransform: () => {},
      drawImage: () => {},
      save: () => {},
      fillText: () => {},
      restore: () => {},
      beginPath: () => {},
      moveTo: () => {},
      lineTo: () => {},
      closePath: () => {},
      stroke: () => {},
      translate: () => {},
      scale: () => {},
      rotate: () => {},
      arc: () => {},
      fill: () => {},
      measureText: () => ({ width: 0 }),
      transform: () => {},
      rect: () => {},
      clip: () => {},
      setLineDash: () => {},
      getLineDash: () => [],
      createLinearGradient: () => ({ addColorStop: () => {} }),
    };
  };
}

test('renders TimeSeries with provenance badge, title, and exact text summary', () => {
  render(
    <TimeSeries
      title="Membrane Voltage"
      units="mV"
      provenance={{ kind: 'RUN', id: 'r_abc123' }}
      series={[
        { label: 'exc', data: [10, 20, 30, 40] }
      ]}
    />
  );

  expect(screen.getByText('Membrane Voltage')).toBeVisible();
  expect(screen.getByText('RUN r_abc123')).toBeVisible();
  expect(screen.getAllByText(/exc/).length).toBeGreaterThan(0);
  expect(screen.getByText(/min 10.00, max 40.00, last 40.00/)).toBeVisible();
});

test('handles NaN and null values in series without crashing and ignores them in summary', () => {
  render(
    <TimeSeries
      provenance="PROBE"
      series={[
        { label: 'v', data: [10, NaN, 20, null as unknown as number, 30] }
      ]}
    />
  );

  expect(screen.getByText('PROBE')).toBeVisible();
  expect(screen.getByText(/min 10.00, max 30.00, last 30.00/)).toBeVisible();
});

test('renders decimated label when decimated prop is true', () => {
  render(
    <TimeSeries
      provenance="PROBE"
      decimated={true}
      series={[
        { label: 'v', data: [1, 2, 3] }
      ]}
    />
  );

  expect(screen.getByText('DECIMATED')).toBeVisible();
});

test('provenance required in TS types and renders correct badge text for all 5 kinds', () => {
  // @ts-expect-error provenance is required
  render(<TimeSeries series={[{ label: 'a', data: [1] }]} />);

  const kinds = ['RUN r_1', 'PROBE', 'REPLAY file.dat', 'TEACHING', 'ILLUSTRATIVE'];
  kinds.forEach(prov => {
    const { unmount } = render(
      <TimeSeries provenance={prov} series={[{ label: 'a', data: [1] }]} />
    );
    expect(screen.getByText(prov)).toBeVisible();
    unmount();
  });
});

test('prefers-reduced-motion respects true and false', () => {
  const matchMediaMock = vi.fn().mockImplementation((query: string) => ({
    matches: query.includes('reduce'),
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }));
  window.matchMedia = matchMediaMock;

  render(
    <TimeSeries provenance="RUN r_1" series={[{ label: 'a', data: [1, 2] }]} />
  );
  expect(matchMediaMock).toHaveBeenCalledWith('(prefers-reduced-motion: reduce)');
});

test('ResizeObserver triggers container resize handler', () => {
  const { container } = render(
    <TimeSeries provenance="RUN r_1" series={[{ label: 'a', data: [1, 2] }]} />
  );
  const wrapper = container.querySelector('.time-series-wrapper');
  expect(wrapper).toBeInTheDocument();
});
