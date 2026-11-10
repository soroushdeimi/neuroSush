// Native vector figures use the site's theme and measured values.
//
// Entry schema (one JSON object per figure):
//   kind: 'line' | 'bars' | 'histograms' | 'heatmaps'
//   title, source, provenance, approx, xLabel, yLabel
//   line/bars/histograms: series: [{name, color: 'accent'|'blue', points: [{x: number|string, y, sd?, low?, high?}]}]
//   heatmaps: groups: [{name, maps: [{label, values: number[][]}]}] (values normalized 0..1)
//
// highlight(index) per kind: line = point index in the first series,
// bars = category index, histograms = series index, heatmaps = group index. -1 clears.

const SVG_NS = 'http://www.w3.org/2000/svg';
const KINDS = new Set(['line', 'bars', 'histograms', 'heatmaps']);
const COLORS = new Set(['accent', 'blue']);

export function clamp(v, lo, hi) {
  if (!Number.isFinite(v)) throw new RangeError('value must be finite');
  return Math.min(hi, Math.max(lo, v));
}

export function linearScale(d0, d1, r0, r1) {
  for (const v of [d0, d1, r0, r1]) if (!Number.isFinite(v)) throw new RangeError('scale bounds must be finite');
  if (d0 === d1) return () => (r0 + r1) / 2;
  const k = (r1 - r0) / (d1 - d0);
  return v => r0 + (v - d0) * k;
}

export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) throw new RangeError('tick bounds must be finite');
  if (min === max) return [min];
  const step0 = Math.abs(max - min) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(step0));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) {
    ticks.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  }
  return ticks;
}

export function formatValue(value, unit = '') {
  if (!Number.isFinite(value)) throw new RangeError('value must be finite');
  const a = Math.abs(value);
  const text = value === 0 ? '0' : (a >= 10000 || a < 0.001) ? value.toExponential(2) : String(Number(value.toPrecision(4)));
  return unit ? `${text} ${unit}` : text;
}

export function errorBarEnds(y, point, domainMin, domainMax) {
  if (!Number.isFinite(y)) throw new RangeError('y must be finite');
  let top = y, bottom = y;
  if (point && point.sd != null) { top = y + point.sd; bottom = y - point.sd; }
  else if (point && point.low != null && point.high != null) { top = point.high; bottom = point.low; }
  return {top: clamp(top, domainMin, domainMax), bottom: clamp(bottom, domainMin, domainMax)};
}

function requireFinite(v, what) {
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new TypeError(`${what} must be a finite number`);
}

function requireString(v, what) {
  if (typeof v !== 'string' || v.length === 0) throw new TypeError(`${what} must be a non-empty string`);
}

export function validateEntry(entry) {
  if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) throw new TypeError('entry must be an object');
  if (!KINDS.has(entry.kind)) throw new TypeError(`unknown kind: ${String(entry.kind)}`);
  requireString(entry.title, 'entry.title');
  for (const key of ['source', 'provenance', 'xLabel', 'yLabel']) {
    if (entry[key] !== undefined) requireString(entry[key], `entry.${key}`);
  }
  if (entry.kind === 'heatmaps') {
    if (!Array.isArray(entry.groups) || entry.groups.length === 0) throw new TypeError('heatmaps need non-empty groups');
    for (const g of entry.groups) {
      requireString(g.name, 'group.name');
      if (!Array.isArray(g.maps) || g.maps.length === 0) throw new TypeError(`group ${g.name} needs non-empty maps`);
      for (const m of g.maps) {
        requireString(m.label, 'map.label');
        if (!Array.isArray(m.values) || m.values.length === 0) throw new TypeError(`map ${m.label} needs non-empty values`);
        const rows = m.values.length;
        for (const row of m.values) {
          if (!Array.isArray(row) || row.length === 0) throw new TypeError(`map ${m.label} rows must be non-empty arrays`);
          if (row.length !== m.values[0].length) throw new TypeError(`map ${m.label} rows must share a width`);
          for (const v of row) {
            requireFinite(v, 'map value');
            if (v < 0 || v > 1) throw new RangeError(`map ${m.label} values must be normalized 0..1`);
          }
        }
        void rows;
      }
    }
    return entry;
  }
  if (!Array.isArray(entry.series) || entry.series.length === 0) throw new TypeError(`${entry.kind} needs non-empty series`);
  for (const s of entry.series) {
    requireString(s.name, 'series.name');
    if (!COLORS.has(s.color)) throw new TypeError(`series ${s.name} color must be 'accent' or 'blue'`);
    if (!Array.isArray(s.points) || s.points.length === 0) throw new TypeError(`series ${s.name} needs non-empty points`);
    for (const p of s.points) {
      if (typeof p.x !== 'number' && typeof p.x !== 'string') throw new TypeError(`series ${s.name} point x must be a number or string`);
      if (typeof p.x === 'number') requireFinite(p.x, 'point.x');
      requireFinite(p.y, 'point.y');
      for (const k of ['sd', 'low', 'high']) if (p[k] !== undefined) requireFinite(p[k], `point.${k}`);
    }
  }
  return entry;
}

const W = 720;
const H = 446;
const M = {top: 24, right: 20, bottom: 64, left: 72};
let uid = 0;

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function text(parent, x, y, str, cls, anchor = 'middle') {
  const t = el('text', {x, y, class: cls, 'text-anchor': anchor}, parent);
  t.textContent = str;
  return t;
}

function showTooltip(ctx, target, lines) {
  const {tooltip, root} = ctx;
  tooltip.textContent = '';
  for (const line of lines.filter(Boolean)) {
    const p = document.createElement('div');
    p.textContent = line;
    tooltip.appendChild(p);
  }
  tooltip.hidden = false;
  const rb = root.getBoundingClientRect();
  const tb = target.getBoundingClientRect();
  tooltip.style.left = `${clamp(tb.left + tb.width / 2 - rb.left, 4, Math.max(4, rb.width - 4))}px`;
  tooltip.style.top = `${tb.top - rb.top - 6}px`;
}

function bindTip(ctx, node, lines) {
  node.setAttribute('tabindex', '0');
  node.setAttribute('role', 'img');
  node.setAttribute('aria-label', lines.filter(Boolean).join(', '));
  node.addEventListener('mouseenter', () => showTooltip(ctx, node, lines));
  node.addEventListener('mouseleave', () => { ctx.tooltip.hidden = true; });
  node.addEventListener('focus', () => showTooltip(ctx, node, lines));
  node.addEventListener('blur', () => { ctx.tooltip.hidden = true; });
}

function yGrid(svg, yScale, dLo, dHi, x0, x1) {
  const g = el('g', {class: 'nc-grid'}, svg);
  for (const t of niceTicks(dLo, dHi)) {
    const y = yScale(t);
    el('line', {x1: x0, x2: x1, y1: y, y2: y, class: 'nc-rule'}, g);
    text(g, x0 - 8, y + 4, formatValue(t), 'nc-tick', 'end');
  }
  return g;
}

function xLabels(svg, labels, xAt, y1, cls = 'nc-tick') {
  const g = el('g', {class: 'nc-xlabels'}, svg);
  const n = labels.length;
  const skip = Math.max(1, Math.ceil(n / 12)); // keep mobile labels from colliding
  labels.forEach((lab, i) => {
    if (i % skip !== 0 && i !== n - 1) return;
    text(g, xAt(i), y1 + 20, String(lab), cls);
  });
  return g;
}

function axisTitles(svg, entry, x0, x1, y0, y1) {
  if (entry.xLabel) text(svg, (x0 + x1) / 2, y1 + 48, entry.xLabel, 'nc-axis');
  if (entry.yLabel) {
    const t = text(svg, 0, 0, entry.yLabel, 'nc-axis');
    t.setAttribute('transform', `translate(16 ${(y0 + y1) / 2}) rotate(-90)`);
  }
}

function legend(svg, series, x0, y0) {
  const g = el('g', {class: 'nc-legend'}, svg);
  let x = x0;
  for (const s of series) {
    el('line', {x1: x, x2: x + 18, y1: y0 - 12, y2: y0 - 12, class: `nc-swatch nc-${s.color}`}, g);
    const t = text(g, x + 24, y0 - 8, s.name, 'nc-tick', 'start');
    x += 24 + Math.max(1, s.name.length) * 7 + 24;
  }
  return g;
}

function seriesColorVar(color) { return color === 'blue' ? 'var(--inh)' : 'var(--accent-text)'; }

function animateIn(ctx, node, cls, delayMs) {
  if (ctx.reducedMotion) return;
  node.classList.add(cls);
  node.style.animationDelay = `${delayMs}ms`;
}

function lineChart(ctx, entry) {
  const {svg, plot} = ctx;
  const x0 = M.left, x1 = W - M.right, y0 = M.top, y1 = H - M.bottom;
  const numeric = entry.series.every(s => s.points.every(p => typeof p.x === 'number'));
  let xLo, xHi, labels;
  if (numeric) {
    const xs = entry.series.flatMap(s => s.points.map(p => p.x));
    xLo = Math.min(...xs); xHi = Math.max(...xs);
  } else {
    labels = [...new Set(entry.series.flatMap(s => s.points.map(p => String(p.x))))];
    xLo = 0; xHi = Math.max(1, labels.length - 1);
  }
  const xScale = linearScale(xLo, xHi, x0 + 12, x1 - 12);
  const ys = entry.series.flatMap(s => s.points.map(p => {
    const {top, bottom} = errorBarEnds(p.y, p, -Infinity, Infinity);
    return [p.y, top, bottom];
  })).flat();
  const yLo = Math.min(0, ...ys), yHi = Math.max(0, ...ys);
  const pad = (yHi - yLo) * 0.08 || 1;
  const yScale = linearScale(yLo - pad, yHi + pad, y1, y0);
  yGrid(svg, yScale, yLo - pad, yHi + pad, x0, x1);
  if (numeric) {
    const g = el('g', {class: 'nc-xlabels'}, svg);
    for (const t of niceTicks(xLo, xHi, 6)) text(g, xScale(t), y1 + 20, formatValue(t), 'nc-tick');
  } else {
    xLabels(svg, labels, xScale, y1);
  }
  axisTitles(svg, entry, x0, x1, y0, y1);
  legend(svg, entry.series, x0, y0);
  const hl = el('g', {class: 'nc-highlight'}, svg);
  ctx.highlightEls = [];
  entry.series.forEach((s, si) => {
    const pts = s.points;
    const px = pts.map(p => xScale(numeric ? p.x : labels.indexOf(String(p.x))));
    const py = pts.map(p => yScale(p.y));
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${px[i].toFixed(2)} ${py[i].toFixed(2)}`).join(' ');
    const path = el('path', {d, class: `nc-line nc-${s.color}`, fill: 'none'}, plot);
    animateIn(ctx, path, 'nc-draw', si * 150);
    pts.forEach((p, i) => {
      const {top, bottom} = errorBarEnds(p.y, p, yLo - pad, yHi + pad);
      if (top !== bottom) {
        const eb = el('g', {class: 'nc-eb nc-eb-in'}, plot);
        el('line', {x1: px[i], x2: px[i], y1: yScale(top), y2: yScale(bottom), class: `nc-ebline nc-${s.color}`}, eb);
        el('line', {x1: px[i] - 4, x2: px[i] + 4, y1: yScale(top), y2: yScale(top), class: `nc-ebline nc-${s.color}`}, eb);
        el('line', {x1: px[i] - 4, x2: px[i] + 4, y1: yScale(bottom), y2: yScale(bottom), class: `nc-ebline nc-${s.color}`}, eb);
        animateIn(ctx, eb, 'nc-fade', 400 + i * 60);
      }
      const c = el('circle', {cx: px[i], cy: py[i], r: 4, class: `nc-dot nc-${s.color} nc-fade`}, plot);
      animateIn(ctx, c, 'nc-fade', 300 + i * 60);
      bindTip(ctx, c, [s.name, `${typeof p.x === 'number' ? formatValue(p.x) : p.x} / ${formatValue(p.y, entry.yLabel)}`, entry.provenance]);
      ctx.highlightEls.push({si, i, els: [c, path]});
    });
  });
  ctx.highlight = idx => {
    hl.replaceChildren();
    ctx.highlightEls.forEach(h => h.els.forEach(n => n.classList.remove('nc-hl')));
    if (idx == null || idx < 0) return;
    const h = ctx.highlightEls[idx];
    if (!h) return;
    h.els.forEach(n => n.classList.add('nc-hl'));
    const [cx, cy] = [h.els[0].getAttribute('cx'), h.els[0].getAttribute('cy')];
    el('circle', {cx, cy, r: 9, class: 'nc-hl-ring'}, hl);
  };
}

function barsChart(ctx, entry) {
  const {svg, plot} = ctx;
  const x0 = M.left, x1 = W - M.right, y0 = M.top, y1 = H - M.bottom;
  const cats = [...new Set(entry.series.flatMap(s => s.points.map(p => String(p.x))))];
  const n = cats.length;
  const xScale = linearScale(0, n, x0, x1);
  const bw = (x1 - x0) / n * 0.7 / entry.series.length;
  const ys = entry.series.flatMap(s => s.points.map(p => {
    const {top, bottom} = errorBarEnds(p.y, p, -Infinity, Infinity);
    return [p.y, top, bottom];
  })).flat();
  const yLo = Math.min(0, ...ys), yHi = Math.max(0, ...ys);
  const pad = (yHi - yLo) * 0.08 || 1;
  const yScale = linearScale(yLo - pad, yHi + pad, y1, y0);
  yGrid(svg, yScale, yLo - pad, yHi + pad, x0, x1);
  xLabels(svg, cats, i => (xScale(i) + xScale(i + 1)) / 2, y1);
  axisTitles(svg, entry, x0, x1, y0, y1);
  legend(svg, entry.series, x0, y0);
  const hl = el('g', {class: 'nc-highlight'}, svg);
  ctx.highlightEls = [];
  entry.series.forEach((s, si) => {
    s.points.forEach(p => {
      const ci = cats.indexOf(String(p.x));
      const cx = xScale(ci) + (xScale(1) - xScale(0)) * 0.15 + si * bw;
      const vTop = Math.max(p.y, 0), vBot = Math.min(p.y, 0);
      const rect = el('rect', {
        x: cx, width: bw,
        y: yScale(Math.max(vTop, yLo - pad)),
        height: Math.max(0, Math.abs(yScale(vTop) - yScale(vBot))),
        class: `nc-bar nc-${s.color} nc-grow`,
      }, plot);
      animateIn(ctx, rect, 'nc-grow', ci * 80 + si * 60);
      const {top, bottom} = errorBarEnds(p.y, p, yLo - pad, yHi + pad);
      if (top !== bottom) {
        el('line', {x1: cx + bw / 2, x2: cx + bw / 2, y1: yScale(top), y2: yScale(bottom), class: `nc-ebline nc-${s.color}`}, plot);
      }
      bindTip(ctx, rect, [s.name, `${p.x}: ${formatValue(p.y, entry.yLabel)}`, entry.provenance]);
      ctx.highlightEls.push({ci, els: [rect]});
    });
  });
  ctx.highlight = idx => {
    hl.replaceChildren();
    ctx.highlightEls.forEach(h => h.els.forEach(e => e.classList.remove('nc-hl')));
    if (idx == null || idx < 0) return;
    const h = ctx.highlightEls.filter(e => e.ci === idx);
    h.forEach(e => e.els.forEach(e2 => e2.classList.add('nc-hl')));
    if (!h.length) return;
    const first = h[0].els[0];
    const cx = Number(first.getAttribute('x')) + Number(first.getAttribute('width')) / 2;
    el('rect', {x: cx - 10, y: y0 - 4, width: 20, height: y1 - y0 + 8, class: 'nc-hl-band'}, hl);
  };
}

function histogramsChart(ctx, entry) {
  const {svg, plot} = ctx;
  const n = entry.series.length;
  const gap = 24;
  const cw = (W - M.left - M.right - gap * (n - 1)) / n;
  const ch = H - M.top - M.bottom;
  const y0 = M.top, y1 = M.top + ch;
  const hl = el('g', {class: 'nc-highlight'}, svg);
  ctx.highlightEls = [];
  entry.series.forEach((s, si) => {
    const x0 = M.left + si * (cw + gap);
    const x1 = x0 + cw;
    const g = el('g', {class: 'nc-panel nc-fade'}, plot);
    animateIn(ctx, g, 'nc-fade', si * 150);
    const cats = s.points.map(p => String(p.x));
    const xScale = linearScale(0, cats.length, x0, x1);
    const bw = (x1 - x0) / cats.length * 0.7;
    const ys = s.points.map(p => {
      const {top, bottom} = errorBarEnds(p.y, p, -Infinity, Infinity);
      return [p.y, top, bottom];
    }).flat();
    const yLo = Math.min(0, ...ys), yHi = Math.max(0, ...ys);
    const pad = (yHi - yLo) * 0.08 || 1;
    const yScale = linearScale(yLo - pad, yHi + pad, y1, y0);
    for (const t of niceTicks(yLo - pad, yHi + pad, 4)) {
      const y = yScale(t);
      el('line', {x1: x0, x2: x1, y1: y, y2: y, class: 'nc-rule'}, g);
      text(g, x0 - 6, y + 4, formatValue(t), 'nc-tick', 'end');
    }
    const skip = Math.max(1, Math.ceil(cats.length / 6));
    cats.forEach((lab, i) => {
      if (i % skip !== 0 && i !== cats.length - 1) return;
      text(g, (xScale(i) + xScale(i + 1)) / 2, y1 + 18, lab, 'nc-tick');
    });
    text(g, (x0 + x1) / 2, y0 - 8, s.name, 'nc-panel-title');
    s.points.forEach((p, i) => {
      const cx = xScale(i) + (x1 - x0) / cats.length * 0.15;
      const vTop = Math.max(p.y, 0), vBot = Math.min(p.y, 0);
      const rect = el('rect', {
        x: cx, width: bw,
        y: yScale(Math.max(vTop, yLo - pad)),
        height: Math.max(0, Math.abs(yScale(vTop) - yScale(vBot))),
        class: `nc-bar nc-${s.color} nc-grow`,
      }, g);
      animateIn(ctx, rect, 'nc-grow', si * 150 + i * 60);
      const {top, bottom} = errorBarEnds(p.y, p, yLo - pad, yHi + pad);
      if (top !== bottom) {
        el('line', {x1: cx + bw / 2, x2: cx + bw / 2, y1: yScale(top), y2: yScale(bottom), class: `nc-ebline nc-${s.color}`}, g);
      }
      bindTip(ctx, rect, [s.name, `${p.x}: ${formatValue(p.y, entry.yLabel)}`, entry.provenance]);
    });
    ctx.highlightEls.push({els: [g]});
  });
  axisTitles(svg, entry, M.left, W - M.right, y0, y1);
  ctx.highlight = idx => {
    hl.replaceChildren();
    ctx.highlightEls.forEach(h => h.els.forEach(e => e.classList.remove('nc-hl')));
    if (idx == null || idx < 0) return;
    const h = ctx.highlightEls[idx];
    if (!h) return;
    h.els.forEach(e => e.classList.add('nc-hl'));
  };
}

function heatmapsChart(ctx, entry) {
  const {svg, plot} = ctx;
  const labelW = 110, legendW = 90;
  const x0 = M.left + labelW, x1 = W - M.right - legendW;
  const groups = entry.groups;
  const mapRows = groups[0].maps[0].values.length;
  const totalCols = groups[0].maps.reduce((a, m) => a + m.values[0].length, 0);
  const cellH = 18, cellGap = 3, mapGap = 24, rowGap = 30;
  const y0 = M.top + 16;
  const groupH = mapRows * (cellH + cellGap) - cellGap + rowGap;
  const y1 = y0 + groups.length * groupH - rowGap;
  const cellW = (x1 - x0 - (groups[0].maps.length - 1) * mapGap) / totalCols;
  // Legend ramp: accent at full activity fading to transparent at zero.
  const defs = el('defs', {}, svg);
  const grad = el('linearGradient', {id: `nc-ramp-${uid}`, x1: 0, y1: 0, x2: 0, y2: 1}, defs);
  el('stop', {offset: '0%', class: 'nc-ramp-1'}, grad);
  el('stop', {offset: '100%', class: 'nc-ramp-0'}, grad);
  const hl = el('g', {class: 'nc-highlight'}, svg);
  ctx.highlightEls = [];
  groups.forEach((g, gi) => {
    const gy = y0 + gi * groupH;
    const gEls = [];
    text(svg, x0 - 10, gy + groupH / 2, g.name, 'nc-group-label', 'end');
    let mx = x0;
    g.maps.forEach((m, mi) => {
      text(svg, mx, gy - 5, m.label, 'nc-map-label', 'start');
      m.values.forEach((row, ri) => {
        row.forEach((v, ci) => {
          const rect = el('rect', {
            x: mx + ci * cellW + 1, y: gy + ri * (cellH + cellGap),
            width: cellW - 2, height: cellH,
            class: 'nc-cell nc-fade',
            'fill-opacity': String(clamp(v, 0, 1)),
          }, plot);
          animateIn(ctx, rect, 'nc-fade', gi * 120 + mi * 80 + ri * 30);
          bindTip(ctx, rect, [`${g.name} · ${m.label}`, `cell (${ri}, ${ci}): ${formatValue(v, 'normalized')}`, entry.provenance]);
          gEls.push(rect);
        });
      });
      mx += m.values[0].length * cellW + mapGap;
    });
    ctx.highlightEls.push({els: gEls});
  });
  // Legend: normalized activity ramp with 0 and 1 endpoints.
  const lx = x1 + 16, ly = y0, lh = Math.min(y1 - y0, 160);
  el('rect', {x: lx, y: ly, width: 14, height: lh, fill: `url(#nc-ramp-${uid})`, class: 'nc-legend-bar'}, svg);
  text(svg, lx + 22, ly + 6, '1', 'nc-tick', 'start');
  text(svg, lx + 22, ly + lh, '0', 'nc-tick', 'start');
  const lt = text(svg, 0, 0, entry.yLabel || 'normalized activity', 'nc-axis');
  lt.setAttribute('transform', `translate(${lx - 26} ${ly + lh / 2}) rotate(-90)`);
  ctx.highlight = idx => {
    hl.replaceChildren();
    ctx.highlightEls.forEach(h => h.els.forEach(e => e.classList.remove('nc-hl')));
    if (idx == null || idx < 0) return;
    const h = ctx.highlightEls[idx];
    if (!h) return;
    h.els.forEach(e => e.classList.add('nc-hl'));
  };
}

const RENDERERS = {line: lineChart, bars: barsChart, histograms: histogramsChart, heatmaps: heatmapsChart};

export function renderNativeChart(container, entry, {playing = true, reducedMotion = false} = {}) {
  if (!container || typeof container.appendChild !== 'function') throw new TypeError('container must be a DOM element');
  const data = validateEntry(entry);
  uid += 1;
  const root = document.createElement('div');
  root.className = 'nc-root';
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': data.title, class: 'nc-svg'});
  const plot = el('g', {class: 'nc-plot'}, svg);
  const tooltip = document.createElement('div');
  tooltip.className = 'nc-tooltip';
  tooltip.setAttribute('role', 'status');
  tooltip.hidden = true;
  root.append(svg, tooltip);
  container.appendChild(root);
  const ctx = {svg, plot, tooltip, root, reducedMotion, highlight: () => {}};
  RENDERERS[data.kind](ctx, data);
  let isPlaying = playing;
  const applyPlay = () => root.classList.toggle('nc-paused', !isPlaying);
  applyPlay();
  return {
    setPlaying(v) { isPlaying = Boolean(v); applyPlay(); },
    highlight(idx) { ctx.highlight(idx); },
    destroy() {
      root.remove();
      ctx.highlight = () => {};
      ctx.highlightEls = [];
    },
  };
}
