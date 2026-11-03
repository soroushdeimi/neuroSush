import React, { useEffect, useRef, useState } from 'react';

export interface WeightHeatmapProps {
  weights: number[][];
  rowStride?: number;
  colStride?: number;
  provenance: string | { kind: 'RUN' | 'PROBE' | 'REPLAY' | 'TEACHING' | 'ILLUSTRATIVE'; id?: string };
  title?: string;
  accessibleName?: string;
  className?: string;
}

export function WeightHeatmap({
  weights = [],
  rowStride = 1,
  colStride = 1,
  provenance,
  title,
  accessibleName,
  className = '',
}: WeightHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [stats, setStats] = useState({ min: 0, max: 0, mean: 0, last: 0 });

  const provString = typeof provenance === 'string'
    ? provenance
    : `${provenance.kind}${provenance.id ? ` ${provenance.id}` : ''}`;

  const safeWeights = weights || [];
  const rows = safeWeights.length;
  const cols = rows > 0 ? (safeWeights[0]?.length || 0) : 0;
  const isBlockMean = rowStride > 1 || colStride > 1;

  useEffect(() => {
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let count = 0;
    let last = 0;

    for (let r = 0; r < rows; r++) {
      const row = safeWeights[r] || [];
      for (let c = 0; c < cols; c++) {
        const val = row[c];
        if (typeof val === 'number' && !isNaN(val)) {
          if (val < min) min = val;
          if (val > max) max = val;
          sum += val;
          count++;
          last = val;
        }
      }
    }
    if (min === Infinity) min = 0;
    if (max === -Infinity) max = 1;
    const mean = count > 0 ? sum / count : 0;
    setStats({ min, max, mean, last });
  }, [safeWeights, rows, cols]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = containerRef.current?.clientWidth || 300;
    const height = 250;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    ctx.clearRect(0, 0, width, height);

    if (rows === 0 || cols === 0) {
      ctx.fillStyle = '#62665c';
      ctx.font = '12px sans-serif';
      ctx.fillText('No weight data', 10, 30);
      return;
    }

    const padding = { top: 20, right: 60, bottom: 30, left: 40 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const cellWidth = plotWidth / cols;
    const cellHeight = plotHeight / rows;

    const min = stats.min;
    const max = stats.max;
    const range = max - min === 0 ? 1 : max - min;

    for (let r = 0; r < rows; r++) {
      const row = safeWeights[r] || [];
      for (let c = 0; c < cols; c++) {
        const val = row[c] ?? 0;
        const norm = Math.max(0, Math.min(1, (val - min) / range));
        ctx.fillStyle = val >= 0
          ? `rgba(174, 60, 35, ${0.1 + 0.9 * norm})`
          : `rgba(98, 102, 92, ${0.1 + 0.9 * Math.abs(val) / Math.max(Math.abs(min), 0.001)})`;
        ctx.fillRect(padding.left + c * cellWidth, padding.top + r * cellHeight, Math.max(1, cellWidth - 0.5), Math.max(1, cellHeight - 0.5));
      }
    }

    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim() || '#62665c';
    ctx.font = '10px monospace';
    ctx.fillText('Presynaptic (rows)', padding.left, padding.top - 8);
    ctx.fillText('Postsynaptic (cols)', padding.left, height - 8);

    const legendX = width - padding.right + 15;
    const legendY = padding.top;
    const legendWidth = 12;
    const legendHeight = plotHeight;

    const grad = ctx.createLinearGradient(0, legendY + legendHeight, 0, legendY);
    grad.addColorStop(0, 'rgba(98, 102, 92, 0.2)');
    grad.addColorStop(0.5, '#f4f0e7');
    grad.addColorStop(1, '#ae3c23');
    ctx.fillStyle = grad;
    ctx.fillRect(legendX, legendY, legendWidth, legendHeight);

    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--line').trim() || '#242a2728';
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

  }, [safeWeights, rows, cols, stats]);

  return (
    <div
      className={`chart-container weight-heatmap-wrapper ${className}`}
      role="region"
      aria-label={accessibleName || title || "Weight Heatmap"}
      style={{ padding: '16px', background: 'var(--bg, #f4f0e7)', border: '1px solid var(--line, #242a2728)', borderRadius: '12px' }}
    >
      <div className="chart-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        {title && <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 500 }}>{title}</h3>}
        <div className="chart-badges" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span className="provenance-badge" style={{ font: '11px var(--font-mono, monospace)', background: 'var(--fg, #242a27)', color: 'var(--bg, #f4f0e7)', padding: '2px 8px', borderRadius: '4px', textTransform: 'uppercase' }}>
            {provString}
          </span>
          {isBlockMean && (
            <span className="block-mean-badge" style={{ font: '11px var(--font-mono, monospace)', background: 'var(--accent, #ae3c23)', color: '#fff', padding: '2px 6px', borderRadius: '4px' }}>
              BLOCK MEAN (stride {rowStride}×{colStride})
            </span>
          )}
        </div>
      </div>
      <div ref={containerRef} style={{ width: '100%', position: 'relative' }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '250px', display: 'block' }} />
        <div className="heatmap-legend-labels" style={{ position: 'absolute', right: '15px', top: '20px', bottom: '30px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', fontSize: '10px', fontFamily: 'var(--font-mono, monospace)', pointerEvents: 'none' }}>
          <span>{stats.max.toFixed(2)}</span>
          <span>{stats.min.toFixed(2)}</span>
        </div>
      </div>
      <div className="chart-summary" style={{ display: 'flex', gap: '16px', marginTop: '8px', fontSize: '12px', fontFamily: 'var(--font-mono, monospace)', color: 'var(--muted, #62665c)' }}>
        <div>Min: {stats.min.toFixed(3)}</div>
        <div>Max: {stats.max.toFixed(3)}</div>
        <div>Mean: {stats.mean.toFixed(3)}</div>
        <div>Last: {stats.last.toFixed(3)}</div>
      </div>
    </div>
  );
}
