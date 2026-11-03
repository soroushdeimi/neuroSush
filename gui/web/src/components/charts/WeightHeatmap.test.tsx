import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import { WeightHeatmap } from './WeightHeatmap';

if (typeof window !== 'undefined') {
  if (!window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
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

test('WeightHeatmap shows block-mean label when strides > 1, and does not when strides == 1', () => {
  const { unmount } = render(
    <WeightHeatmap
      provenance="RUN r_1"
      rowStride={2}
      colStride={2}
      weights={[[0.1, 0.2], [0.3, 0.4]]}
    />
  );
  expect(screen.getByText(/BLOCK MEAN/)).toBeVisible();
  unmount();

  render(
    <WeightHeatmap
      provenance="RUN r_1"
      rowStride={1}
      colStride={1}
      weights={[[0.1, 0.2], [0.3, 0.4]]}
    />
  );
  expect(screen.queryByText(/BLOCK MEAN/)).not.toBeInTheDocument();
});

test('WeightHeatmap colour scale legend shows true min and max of data', () => {
  render(
    <WeightHeatmap
      provenance="PROBE"
      weights={[
        [0.15, 0.25],
        [0.35, 0.95]
      ]}
    />
  );
  expect(screen.getByText('0.95')).toBeVisible();
  expect(screen.getByText('0.15')).toBeVisible();
  expect(screen.getByText(/Min: 0.150/)).toBeVisible();
  expect(screen.getByText(/Max: 0.950/)).toBeVisible();
});

test('handles empty weights gracefully', () => {
  render(
    <WeightHeatmap
      provenance="TEACHING"
      weights={[]}
    />
  );
  expect(screen.getByText('TEACHING')).toBeVisible();
  expect(screen.getByText(/Min: 0/)).toBeVisible();
});
