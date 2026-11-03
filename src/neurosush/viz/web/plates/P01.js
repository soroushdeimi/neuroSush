/**
 * PLATE P01 — Population raster (recorded run)
 * Renders recorded spike/activity traces across a population of neurons over time.
 * Replay only: reads exclusively from recorded Trace data; no simulation in JS.
 */
(function (root) {
  'use strict';

  const Atlas = typeof globalThis !== 'undefined' && globalThis.Atlas
    ? globalThis.Atlas
    : (typeof module !== 'undefined' && module.exports ? require('../atlas.js') : root.Atlas);

  const S = {
    title: 'Population raster (recorded run)',
    neurons: 'Neurons',
    steps: 'Steps',
    activeAtStep: 'Active at step',
    timeWindow: 'Time window',
    neuronRange: 'Neuron range',
    emptyState: 'No recorded spike data available',
    decimatedLabel: 'Decimated / Truncated run',
    footer: 'source: replay · recorded Trace population raster'
  };

  const core = {
    /**
     * Extract spike raster points from a Trace object with time window [stepStart, stepEnd] (inclusive)
     * and neuron selection (either range or explicit array of neuron indices).
     * Out-of-range neurons are ignored without crashing.
     */
    extractRaster(trace, options = {}) {
      if (!trace || !trace.series || !trace.series.spikes) {
        return [];
      }
      const spkSeries = trace.series.spikes;
      const shape = spkSeries.shape || [trace.steps, 1];
      const nNeurons = shape.length > 1 ? shape.slice(1).reduce((a, b) => a * b, 1) : 1;

      const stepStart = options.stepStart ?? 0;
      const stepEnd = options.stepEnd ?? (Math.max(0, trace.steps - 1));
      const sStart = Math.max(0, Math.min(Math.max(0, trace.steps - 1), stepStart));
      const sEnd = Math.max(0, Math.min(Math.max(0, trace.steps - 1), stepEnd));

      let targetNeurons;
      if (Array.isArray(options.neurons)) {
        targetNeurons = options.neurons.filter(n => Number.isInteger(n) && n >= 0 && n < nNeurons);
      } else {
        const nStart = Math.max(0, options.neuronStart ?? 0);
        const nEnd = Math.min(nNeurons, options.neuronEnd ?? nNeurons);
        targetNeurons = [];
        for (let n = nStart; n < nEnd; n++) targetNeurons.push(n);
      }

      const spikes = [];
      for (let s = sStart; s <= sEnd; s++) {
        let row;
        try {
          row = trace.at('spikes', s);
        } catch (e) {
          continue;
        }
        for (const n of targetNeurons) {
          if (n < row.length && row[n] > 0) {
            spikes.push({ neuron: n, step: s });
          }
        }
      }
      return spikes;
    },

    getStepStats(trace, t) {
      if (!trace || !trace.series || !trace.series.spikes) {
        return { activeNeurons: 0, totalNeurons: 0, empty: true };
      }
      try {
        const row = trace.at('spikes', t);
        let active = 0;
        for (let i = 0; i < row.length; i++) {
          if (row[i] > 0) active++;
        }
        return { activeNeurons: active, totalNeurons: row.length, empty: row.length === 0 };
      } catch (e) {
        return { activeNeurons: 0, totalNeurons: 0, empty: true };
      }
    },

    isDecimatedOrTruncated(trace) {
      if (!trace || !trace.meta) return false;
      return Boolean(trace.meta.decimated || trace.meta.truncated || trace.meta.max_steps_hit || trace.meta.flags);
    }
  };

  function mount(rootEl, ctx) {
    const doc = (rootEl && rootEl.ownerDocument) || (typeof document !== 'undefined' ? document : (globalThis.document || {}));
    const { trace, clock, colors, canvas: createCanvas, onFrame, margin } = ctx;

    const spkSeries = trace && trace.series && trace.series.spikes;
    const shape = spkSeries && spkSeries.shape ? spkSeries.shape : [trace ? trace.steps : 0, 1];
    const totalNeurons = shape.length > 1 ? shape.slice(1).reduce((a, b) => a * b, 1) : 1;
    const totalSteps = trace ? trace.steps : 0;
    const isEmpty = !trace || totalSteps === 0 || totalNeurons === 0 || !spkSeries;
    const isDecimated = core.isDecimatedOrTruncated(trace);

    let stepStart = 0;
    let stepEnd = Math.max(0, totalSteps - 1);
    let neuronStart = 0;
    let neuronEnd = totalNeurons;

    if (margin) {
      margin.innerHTML = `
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.neurons}</div>
          <div class="atlas-margin-value atlas-mono" id="p01-neurons">${totalNeurons}</div>
        </div>
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.steps}</div>
          <div class="atlas-margin-value atlas-mono" id="p01-steps">${totalSteps}</div>
        </div>
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.activeAtStep}</div>
          <div class="atlas-margin-value atlas-mono" id="p01-active">${isEmpty ? '0' : '0 / ' + totalNeurons}</div>
        </div>
        ${isDecimated ? `<div class="atlas-margin-section"><div style="color:var(--volt,#b45309);font-weight:600;" id="p01-decimated">${S.decimatedLabel}</div></div>` : ''}
        <div class="atlas-margin-section">
          <div class="atlas-margin-label">${S.footer}</div>
        </div>
      `;
    }

    const neuronsEl = doc.getElementById ? doc.getElementById('p01-neurons') : null;
    const stepsEl = doc.getElementById ? doc.getElementById('p01-steps') : null;
    const activeEl = doc.getElementById ? doc.getElementById('p01-active') : null;

    const plotCanvas = createCanvas(800, 500);
    const plotCtx = plotCanvas.getContext('2d');

    function updateMargin(t) {
      if (isEmpty) return;
      const stats = core.getStepStats(trace, t);
      if (activeEl) {
        activeEl.textContent = `${stats.activeNeurons} / ${stats.totalNeurons}`;
      }
      if (neuronsEl) neuronsEl.textContent = String(totalNeurons);
      if (stepsEl) stepsEl.textContent = String(totalSteps);
    }

    function render() {
      const { width, height } = Atlas.fitCanvas(plotCanvas);
      if (width <= 0 || height <= 0) return;

      const c = colors();
      const t = clock ? clock.t : 0;

      plotCtx.direction = 'ltr';
      plotCtx.clearRect(0, 0, width, height);

      if (isEmpty) {
        plotCtx.fillStyle = c.inkMuted || '#888';
        plotCtx.font = '14px ui-sans-serif, system-ui, sans-serif';
        plotCtx.textAlign = 'center';
        plotCtx.textBaseline = 'middle';
        plotCtx.fillText(S.emptyState, width / 2, height / 2);
        return;
      }

      const padLeft = 60;
      const padRight = 30;
      const padTop = 30;
      const padBottom = 40;
      const plotW = Math.max(10, width - padLeft - padRight);
      const plotH = Math.max(10, height - padTop - padBottom);

      const visibleSteps = Math.max(1, stepEnd - stepStart + 1);
      const visibleNeurons = Math.max(1, neuronEnd - neuronStart);

      const xToPx = (s) => padLeft + ((s - stepStart) / visibleSteps) * plotW;
      const yToPx = (n) => padTop + ((n - neuronStart) / visibleNeurons) * plotH;

      plotCtx.strokeStyle = c.border;
      plotCtx.lineWidth = 1;
      plotCtx.strokeRect(padLeft, padTop, plotW, plotH);

      plotCtx.save();
      plotCtx.beginPath();
      plotCtx.rect(padLeft, padTop, plotW, plotH);
      plotCtx.clip();

      const spikes = core.extractRaster(trace, { neuronStart, neuronEnd, stepStart, stepEnd });
      plotCtx.fillStyle = c.volt;
      for (const spk of spikes) {
        const sx = xToPx(spk.step);
        const sy = yToPx(spk.neuron);
        const cellH = Math.max(1, plotH / visibleNeurons);
        plotCtx.fillRect(sx, sy, Math.max(1, plotW / visibleSteps), Math.max(1, cellH - 0.5));
      }

      if (clock && t >= stepStart && t <= stepEnd) {
        const curX = xToPx(t);
        plotCtx.strokeStyle = c.exc;
        plotCtx.lineWidth = 1.5;
        plotCtx.beginPath();
        plotCtx.moveTo(curX, padTop);
        plotCtx.lineTo(curX, padTop + plotH);
        plotCtx.stroke();
      }

      plotCtx.restore();

      plotCtx.fillStyle = c.inkMuted;
      plotCtx.font = '11px ui-monospace, monospace';
      plotCtx.textAlign = 'right';
      plotCtx.textBaseline = 'middle';
      plotCtx.fillText(`N${neuronEnd}`, padLeft - 8, padTop);
      plotCtx.fillText(`N${neuronStart}`, padLeft - 8, padTop + plotH);

      plotCtx.textAlign = 'center';
      plotCtx.textBaseline = 'top';
      plotCtx.fillText(String(stepStart), padLeft, padTop + plotH + 8);
      plotCtx.fillText(String(stepEnd), padLeft + plotW, padTop + plotH + 8);

      updateMargin(t);
    }

    if (ctx.onFrame) {
      ctx.onFrame(() => render());
    }
    render();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { core };
  }

  if (typeof Atlas !== 'undefined' && Atlas.Plate) {
    Atlas.Plate.register({
      id: 'P01',
      title: { en: S.title },
      source: 'replay',
      mount
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
