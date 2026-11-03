import React, { useEffect, useRef, useState } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

if (typeof window !== 'undefined' && !window.matchMedia) {
  window.matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  });
}

export interface TimeSeriesSeries {
  label: string;
  data: number[];
  color?: string;
}

export interface TimeSeriesProps {
  series: TimeSeriesSeries[];
  x?: number[];
  units?: string;
  provenance: string | { kind: 'RUN' | 'PROBE' | 'REPLAY' | 'TEACHING' | 'ILLUSTRATIVE'; id?: string };
  decimated?: boolean;
  title?: string;
  accessibleName?: string;
  className?: string;
}

export function TimeSeries({
  series,
  x,
  units,
  provenance,
  decimated = false,
  title,
  accessibleName,
  className = '',
}: TimeSeriesProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const uplotRef = useRef<uPlot | null>(null);

  const [minMaxLast, setMinMaxLast] = useState<{ label: string; min: number; max: number; last: number }[]>([]);

  useEffect(() => {
    const summary = (series || []).map(s => {
      const data = (s.data || []).filter(v => typeof v === 'number' && !isNaN(v));
      const min = data.length ? Math.min(...data) : 0;
      const max = data.length ? Math.max(...data) : 0;
      const last = data.length ? data[data.length - 1] : 0;
      return { label: s.label, min, max, last };
    });
    setMinMaxLast(summary);
  }, [series]);

  const provString = !provenance
    ? ''
    : typeof provenance === 'string'
      ? provenance
      : `${provenance.kind}${provenance.id ? ` ${provenance.id}` : ''}`;

  useEffect(() => {
    if (!containerRef.current) return;
    if (!series || series.length === 0 || !series[0]?.data) return;

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const width = containerRef.current.clientWidth || 400;
    const height = containerRef.current.clientHeight || 220;

    const xData = x && x.length === series[0]?.data.length
      ? x
      : Array.from({ length: series[0]?.data.length || 0 }, (_, i) => i);

    const uplotData: uPlot.AlignedData = [
      xData,
      ...series.map(s => s.data)
    ];

    const getCssVar = (name: string) => {
      if (typeof window === 'undefined') return '#242a27';
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#242a27';
    };

    const muted = getCssVar('--muted') || '#62665c';
    const line = getCssVar('--line') || '#242a2728';
    const accent = getCssVar('--accent') || '#ae3c23';

    const opts: uPlot.Options = {
      width,
      height,
      cursor: {
        show: !reducedMotion,
      },
      series: [
        { label: 'Time' },
        ...series.map((s) => ({
          label: s.label,
          stroke: s.color || accent,
          width: 2,
          spanGaps: true,
        }))
      ],
      axes: [
        {
          stroke: muted,
          grid: { stroke: line },
          ticks: { stroke: line },
        },
        {
          stroke: muted,
          grid: { stroke: line },
          ticks: { stroke: line },
          label: units,
        },
      ],
    };

    if (uplotRef.current) {
      uplotRef.current.destroy();
    }

    containerRef.current.innerHTML = '';
    const u = new uPlot(opts, uplotData, containerRef.current);
    uplotRef.current = u;

    const handleResize = () => {
      if (containerRef.current && uplotRef.current) {
        const w = containerRef.current.clientWidth || 400;
        const h = containerRef.current.clientHeight || 220;
        uplotRef.current.setSize({ width: w, height: h });
      }
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      if (uplotRef.current) {
        uplotRef.current.destroy();
        uplotRef.current = null;
      }
    };
  }, [series, x, units]);

  return (
    <div
      className={`chart-container time-series-wrapper ${className}`}
      role="region"
      aria-label={accessibleName || title || "Time Series Chart"}
      style={{ padding: '16px', background: 'var(--bg, #f4f0e7)', border: '1px solid var(--line, #242a2728)', borderRadius: '12px' }}
    >
      <div className="chart-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        {title && <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 500 }}>{title}</h3>}
        <div className="chart-badges" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span className="provenance-badge" style={{ font: '11px var(--font-mono, monospace)', background: 'var(--fg, #242a27)', color: 'var(--bg, #f4f0e7)', padding: '2px 8px', borderRadius: '4px', textTransform: 'uppercase' }}>
            {provString}
          </span>
          {decimated && (
            <span className="decimated-label" style={{ font: '11px var(--font-mono, monospace)', background: 'var(--muted, #62665c)', color: '#fff', padding: '2px 6px', borderRadius: '4px' }}>
              DECIMATED
            </span>
          )}
        </div>
      </div>
      <div ref={containerRef} style={{ width: '100%', minHeight: '200px' }} />
      <div className="chart-summary" style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', marginTop: '8px', fontSize: '12px', fontFamily: 'var(--font-mono, monospace)', color: 'var(--muted, #62665c)' }}>
        {minMaxLast.map(s => (
          <div key={s.label}>
            <strong>{s.label}</strong>: min {s.min.toFixed(2)}, max {s.max.toFixed(2)}, last {s.last.toFixed(2)} {units || ''}
          </div>
        ))}
      </div>
    </div>
  );
}
