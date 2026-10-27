/**
 * Demo Plate: Membrane Potential & Spikes
 * Oscilloscope view of membrane potential, spikes, and closed-form theory curve.
 */
(function (root) {
  'use strict';

  const Atlas = typeof globalThis !== 'undefined' && globalThis.Atlas
    ? globalThis.Atlas
    : (typeof module !== 'undefined' && module.exports ? require('../atlas.js') : root.Atlas);

  if (!Atlas || !Atlas.Plate) return;

  Atlas.Plate.register({
    id: 'demo',
    title: 'Membrane Potential & Spikes',
    source: 'replay',

    mount(rootEl, ctx) {
      const { trace, clock, colors, canvas: createCanvas, onFrame, margin, figure } = ctx;

      // Extract series & metadata
      const vSeries = trace.series.v;
      const spikesSeries = trace.series.spikes;
      const theorySeries = trace.series.v_theory;
      const hasTheory = Boolean(theorySeries || trace.meta.theory);
      let showTheory = true;

      // Listen for theory toggle
      window.addEventListener('atlas:theory', (e) => {
        showTheory = e.detail && typeof e.detail.active === 'boolean' ? e.detail.active : !showTheory;
        render();
      });

      // Compute voltage min and max bounds for plotting
      let vMin = -75;
      let vMax = -45;
      if (vSeries && vSeries.data && vSeries.data.length > 0) {
        let min = Infinity;
        let max = -Infinity;
        for (let i = 0; i < vSeries.data.length; i++) {
          const val = vSeries.data[i];
          if (val < min) min = val;
          if (val > max) max = val;
        }
        if (Number.isFinite(min) && Number.isFinite(max)) {
          const pad = Math.max(2, (max - min) * 0.1);
          vMin = min - pad;
          vMax = max + pad;
        }
      }

      // Build Margin notes
      if (margin) {
        margin.innerHTML = `
          <div class="atlas-margin-section">
            <div class="atlas-margin-label" data-i18n="instantaneous_voltage">${Atlas.t('instantaneous_voltage')}</div>
            <div class="atlas-margin-value atlas-mono" id="demo-v-val">-65.00 mV</div>
          </div>
          <div class="atlas-margin-section">
            <div class="atlas-margin-label" data-i18n="spike_event">${Atlas.t('spike_event')}</div>
            <div class="atlas-margin-value" id="demo-spike-val">${Atlas.t('status_subthreshold')}</div>
          </div>
          ${hasTheory ? `
          <div class="atlas-margin-section" id="demo-theory-section">
            <div class="atlas-margin-label" data-i18n="theoretical_voltage">${Atlas.t('theoretical_voltage')}</div>
            <div class="atlas-margin-value atlas-mono" id="demo-v-theory">-65.00 mV</div>
            <div class="atlas-margin-label" style="margin-top:6px;" data-i18n="difference">${Atlas.t('difference')}</div>
            <div class="atlas-margin-value atlas-mono" id="demo-v-diff">0.00 mV</div>
          </div>
          ` : ''}
          <div class="atlas-margin-section">
            <div class="atlas-margin-label" data-i18n="equation">${Atlas.t('equation')}</div>
            <div class="atlas-equation-box atlas-mono">τ_m · dV/dt = -(V - V_rest) + R · I(t)</div>
          </div>
        `;
      }

      const vValEl = document.getElementById('demo-v-val');
      const spikeValEl = document.getElementById('demo-spike-val');
      const vTheoryEl = document.getElementById('demo-v-theory');
      const vDiffEl = document.getElementById('demo-v-diff');

      // Create figure canvas
      const plotCanvas = createCanvas(800, 500);
      const plotCtx = plotCanvas.getContext('2d');

      function updateMargin(t) {
        if (!vSeries) return;
        const vNow = trace.at('v', t)[0];
        if (vValEl) {
          vValEl.textContent = `${Atlas.formatNumber(vNow, 2)} mV`;
        }

        const spiked = spikesSeries ? trace.at('spikes', t)[0] > 0 : false;
        if (spikeValEl) {
          spikeValEl.textContent = spiked ? Atlas.t('status_active') : Atlas.t('status_subthreshold');
          spikeValEl.style.color = spiked ? colors().volt : colors().ink;
        }

        if (hasTheory && theorySeries) {
          const vTheo = trace.at('v_theory', t)[0];
          const diff = Math.abs(vNow - vTheo);
          if (vTheoryEl) vTheoryEl.textContent = `${Atlas.formatNumber(vTheo, 2)} mV`;
          if (vDiffEl) vDiffEl.textContent = `${Atlas.formatNumber(diff, 2)} mV`;
        }
      }

      function render() {
        const { width, height } = Atlas.fitCanvas(plotCanvas);
        if (width <= 0 || height <= 0) return;

        const c = colors();
        const t = clock.t;
        const totalSteps = Math.max(1, clock.steps);

        plotCtx.direction = 'ltr';
        plotCtx.clearRect(0, 0, width, height);

        // Chart dimensions with margins for axes
        const padLeft = 60;
        const padRight = 30;
        const padTop = 30;
        const padBottom = 40;
        const plotW = Math.max(10, width - padLeft - padRight);
        const plotH = Math.max(10, height - padTop - padBottom);

        // Oscilloscope window: show either full trace or scrolling window
        const windowSize = Math.min(totalSteps, 120);
        let startStep = 0;
        let endStep = totalSteps;
        if (totalSteps > windowSize) {
          startStep = Math.max(0, Math.min(totalSteps - windowSize, t - Math.floor(windowSize * 0.7)));
          endStep = startStep + windowSize;
        }
        const visibleCount = Math.max(1, endStep - startStep);

        const xToPx = (s) => padLeft + ((s - startStep) / (visibleCount - 1)) * plotW;
        const yToPx = (v) => padTop + (1 - (v - vMin) / (vMax - vMin)) * plotH;

        // Draw grid lines & Y ticks
        const yTicks = Atlas.niceTicks(vMin, vMax, 5);
        plotCtx.textAlign = 'right';
        plotCtx.textBaseline = 'middle';
        plotCtx.font = '11px ui-monospace, monospace';

        for (const yVal of yTicks) {
          const y = yToPx(yVal);
          if (y < padTop || y > padTop + plotH) continue;

          // Grid line
          plotCtx.strokeStyle = c.border;
          plotCtx.lineWidth = 0.5;
          plotCtx.beginPath();
          plotCtx.moveTo(padLeft, y);
          plotCtx.lineTo(padLeft + plotW, y);
          plotCtx.stroke();

          // Tick label
          plotCtx.fillStyle = c.inkMuted;
          plotCtx.fillText(`${yVal} mV`, padLeft - 8, y);
        }

        // Draw X axis ticks
        const xTicks = Atlas.niceTicks(startStep, endStep - 1, 6);
        plotCtx.textAlign = 'center';
        plotCtx.textBaseline = 'top';

        for (const sVal of xTicks) {
          const x = xToPx(sVal);
          if (x < padLeft || x > padLeft + plotW) continue;

          plotCtx.strokeStyle = c.border;
          plotCtx.lineWidth = 0.5;
          plotCtx.beginPath();
          plotCtx.moveTo(x, padTop);
          plotCtx.lineTo(x, padTop + plotH);
          plotCtx.stroke();

          plotCtx.fillStyle = c.inkMuted;
          plotCtx.fillText(String(Math.round(sVal)), x, padTop + plotH + 8);
        }

        // Axis border box
        plotCtx.strokeStyle = c.border;
        plotCtx.lineWidth = 1;
        plotCtx.strokeRect(padLeft, padTop, plotW, plotH);

        // Threshold line if threshold is specified in metadata
        if (typeof trace.meta.threshold === 'number') {
          const thY = yToPx(trace.meta.threshold);
          if (thY >= padTop && thY <= padTop + plotH) {
            plotCtx.save();
            plotCtx.strokeStyle = c.inkMuted;
            plotCtx.setLineDash([4, 4]);
            plotCtx.lineWidth = 1;
            plotCtx.beginPath();
            plotCtx.moveTo(padLeft, thY);
            plotCtx.lineTo(padLeft + plotW, thY);
            plotCtx.stroke();
            plotCtx.restore();

            // Label on small background-colored pad strictly above the line, inside plot area, right-aligned
            const labelText = 'V_th';
            plotCtx.save();
            plotCtx.font = '11px ui-monospace, monospace';
            const padW = 28;
            const padH = 13;
            const padX = padLeft + plotW - padW - 4;
            const padY = thY - padH - 3;
            plotCtx.fillStyle = c.paperCard;
            plotCtx.fillRect(padX, padY, padW, padH);
            plotCtx.fillStyle = c.inkMuted;
            plotCtx.textAlign = 'center';
            plotCtx.textBaseline = 'middle';
            plotCtx.fillText(labelText, padX + padW / 2, padY + padH / 2);
            plotCtx.restore();
          }
        }

        // Clip to plot area for curves
        plotCtx.save();
        plotCtx.beginPath();
        plotCtx.rect(padLeft, padTop, plotW, plotH);
        plotCtx.clip();

        // 1. Draw Theory curve if active
        if (showTheory && hasTheory && theorySeries) {
          const theoXs = [];
          const theoYs = [];
          for (let s = startStep; s < endStep; s++) {
            theoXs.push(xToPx(s));
            theoYs.push(yToPx(theorySeries.data[s]));
          }
          Atlas.lineSeries(plotCtx, theoXs, theoYs, {
            strokeStyle: c.theory,
            lineWidth: 1.5,
            lineDash: [5, 4]
          });
        }

        // 2. Draw Recorded voltage curve
        if (vSeries) {
          // Future trace (faint)
          if (t < endStep - 1) {
            const futXs = [];
            const futYs = [];
            const fStart = Math.max(startStep, t);
            for (let s = fStart; s < endStep; s++) {
              futXs.push(xToPx(s));
              futYs.push(yToPx(vSeries.data[s]));
            }
            plotCtx.save();
            plotCtx.globalAlpha = 0.25;
            Atlas.lineSeries(plotCtx, futXs, futYs, {
              strokeStyle: c.volt,
              lineWidth: 1.5
            });
            plotCtx.restore();
          }

          // Past/Current trace (solid)
          const pastXs = [];
          const pastYs = [];
          const pEnd = Math.min(endStep, t + 1);
          for (let s = startStep; s < pEnd; s++) {
            pastXs.push(xToPx(s));
            pastYs.push(yToPx(vSeries.data[s]));
          }
          Atlas.lineSeries(plotCtx, pastXs, pastYs, {
            strokeStyle: c.volt,
            lineWidth: 2
          });

          // 3. Mark spikes
          if (spikesSeries) {
            plotCtx.fillStyle = c.volt;
            for (let s = startStep; s < endStep; s++) {
              if (spikesSeries.data[s] > 0) {
                const spkX = xToPx(s);
                const spkY = yToPx(vSeries.data[s]);
                // Vertical marker pip
                plotCtx.fillRect(spkX - 1.5, padTop, 3, 10);
                // Dot on peak
                plotCtx.beginPath();
                plotCtx.arc(spkX, spkY, 3, 0, Math.PI * 2);
                plotCtx.fill();
              }
            }
          }

          // 4. Current playhead point & scanning cursor
          if (t >= startStep && t < endStep) {
            const curX = xToPx(t);
            const curY = yToPx(vSeries.data[t]);

            // Vertical cursor line
            plotCtx.strokeStyle = c.inkMuted;
            plotCtx.lineWidth = 0.75;
            plotCtx.beginPath();
            plotCtx.moveTo(curX, padTop);
            plotCtx.lineTo(curX, padTop + plotH);
            plotCtx.stroke();

            // Halo around current point
            plotCtx.strokeStyle = c.volt;
            plotCtx.lineWidth = 1.5;
            plotCtx.beginPath();
            plotCtx.arc(curX, curY, 6, 0, Math.PI * 2);
            plotCtx.stroke();

            plotCtx.fillStyle = c.paper;
            plotCtx.beginPath();
            plotCtx.arc(curX, curY, 4, 0, Math.PI * 2);
            plotCtx.fill();

            plotCtx.fillStyle = c.volt;
            plotCtx.beginPath();
            plotCtx.arc(curX, curY, 2.5, 0, Math.PI * 2);
            plotCtx.fill();
          }
        }

        plotCtx.restore();

        // Update margin note
        updateMargin(t);
      }

      onFrame(() => render());
      render();
    }
  });
})(typeof globalThis !== 'undefined' ? globalThis : this);
