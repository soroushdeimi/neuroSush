/**
 * Live TEM Replay and Spatial Probe Explorer
 */
(function() {
  const TEM = {
    runs: {},
    currentVariant: 'v2',
    currentWorldIndex: 0,
    currentStep: 0,
    isPlaying: false,
    timer: null,
    speed: 1,

    async init() {
      try {
        const [v2Res, v1Res] = await Promise.all([
          fetch('data/tem-run-v2-s0.json').catch(() => null),
          fetch('data/tem-run-v1-s0.json').catch(() => null)
        ]);

        let loadedAny = false;
        if (v2Res && v2Res.ok) {
          try {
            this.runs['v2'] = await v2Res.json();
            loadedAny = true;
          } catch (e) {
            console.error('Failed to parse v2 json', e);
          }
        }
        if (v1Res && v1Res.ok) {
          try {
            this.runs['v1'] = await v1Res.json();
            loadedAny = true;
          } catch (e) {
            console.error('Failed to parse v1 json', e);
          }
        }

        if (!loadedAny) {
          this.showFriendlyMessage('No recorded run published yet');
          return;
        }

        this.setupListeners();
        this.updateFromVariant();
      } catch (err) {
        this.showFriendlyMessage('No recorded run published yet');
      }
    },

    showFriendlyMessage(msg) {
      const banner = document.getElementById('tem-message');
      if (banner) {
        banner.textContent = msg;
        if (banner.style) banner.style.display = 'block';
      }
    },

    setupListeners() {
      const variantSel = document.getElementById('tem-variant-select');
      const worldSel = document.getElementById('tem-world-select');
      const playBtn = document.getElementById('tem-btn-play');
      const pauseBtn = document.getElementById('tem-btn-pause');
      const stepBtn = document.getElementById('tem-btn-step');
      const speedRange = document.getElementById('tem-speed-range');

      if (variantSel) {
        variantSel.addEventListener('change', (e) => {
          this.currentVariant = e.target.value;
          this.currentStep = 0;
          this.updateFromVariant();
        });
      }

      if (worldSel) {
        worldSel.addEventListener('change', (e) => {
          this.currentWorldIndex = parseInt(e.target.value, 10) || 0;
          this.currentStep = 0;
          this.render();
        });
      }

      if (playBtn) playBtn.addEventListener('click', () => this.play());
      if (pauseBtn) pauseBtn.addEventListener('click', () => this.pause());
      if (stepBtn) stepBtn.addEventListener('click', () => this.stepForward());
      if (speedRange) speedRange.addEventListener('input', (e) => {
        this.speed = parseFloat(e.target.value) || 1;
        if (this.isPlaying) {
          this.pause();
          this.play();
        }
      });
    },

    updateFromVariant() {
      const run = this.runs[this.currentVariant];
      const worldSel = document.getElementById('tem-world-select');
      if (!run) {
        this.showFriendlyMessage('No recorded run published yet');
        return;
      }

      // Check fixture label check
      const label = (run.meta && run.meta.label) || '';
      const fixtureBanner = document.getElementById('tem-fixture-banner');
      if (label.toUpperCase().startsWith('FIXTURE')) {
        if (fixtureBanner) fixtureBanner.style.display = 'block';
        this.showFriendlyMessage('Test fixture detected; not a real model run.');
        return;
      } else {
        if (fixtureBanner) fixtureBanner.style.display = 'none';
      }

      if (worldSel) {
        worldSel.innerHTML = '';
        run.worlds.forEach((w, idx) => {
          const opt = document.createElement('option');
          opt.value = idx;
          opt.textContent = `World ${idx} (size ${w.size}x${w.size})`;
          worldSel.appendChild(opt);
        });
        worldSel.value = this.currentWorldIndex;
      }

      this.updateMetaBanner(run.meta);
      this.render();
    },

    updateMetaBanner(meta) {
      const labelEl = document.getElementById('tem-label-display');
      const noteEl = document.getElementById('tem-note-display');
      if (labelEl) labelEl.textContent = `Label: ${meta.label || 'learned structure'}`;
      if (noteEl) noteEl.textContent = `Note: ${meta.note || 'one seed; no claim of grid cells'}`;
    },

    getWorld() {
      const run = this.runs[this.currentVariant];
      if (!run || !run.worlds || !run.worlds[this.currentWorldIndex]) return null;
      return run.worlds[this.currentWorldIndex];
    },

    play() {
      if (this.isPlaying) return;
      this.isPlaying = true;
      const intervalMs = Math.max(100, 500 / this.speed);
      this.timer = setInterval(() => {
        this.stepForward();
      }, intervalMs);
    },

    pause() {
      this.isPlaying = false;
      if (this.timer) {
        clearInterval(this.timer);
        this.timer = null;
      }
    },

    stepForward() {
      const world = this.getWorld();
      if (!world || !world.walk || world.walk.length === 0) return;
      this.currentStep = (this.currentStep + 1) % world.walk.length;
      this.render();
    },

    render() {
      const world = this.getWorld();
      const run = this.runs[this.currentVariant];
      if (!world || !run) return;

      const label = (run.meta && run.meta.label) || '';
      if (label.toUpperCase().startsWith('FIXTURE')) {
        return; // Refuse to present as model result
      }

      const stepEl = document.getElementById('tem-step-counter');
      const accEl = document.getElementById('tem-accuracy-counter');
      if (stepEl) stepEl.textContent = `Step: ${this.currentStep + 1} / ${world.walk.length}`;

      const accuracy = this.computeNewEdgeAccuracy(world, this.currentStep);
      if (accEl) accEl.textContent = `New-edge accuracy: ${(accuracy * 100).toFixed(1)}%`;

      this.renderGrid(world, this.currentStep);
      this.renderRateMaps(world);
      this.renderProbe(run.probe);
    },

    computeNewEdgeAccuracy(world, upToStep) {
      let correctNewEdge = 0;
      let totalNewEdge = 0;
      const limit = Math.min(upToStep + 1, world.walk.length);
      for (let i = 0; i < limit; i++) {
        if (world.first_visit_new_edge && world.first_visit_new_edge[i] === 1) {
          totalNewEdge++;
          if (world.pred_correct && world.pred_correct[i] === 1) {
            correctNewEdge++;
          }
        }
      }
      return totalNewEdge > 0 ? correctNewEdge / totalNewEdge : 0;
    },

    renderGrid(world, step) {
      const canvas = document.getElementById('tem-grid-canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const size = world.size || 8;
      const w = canvas.width;
      const h = canvas.height;
      const cellW = w / size;
      const cellH = h / size;

      ctx.clearRect(0, 0, w, h);

      for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
          const nodeIdx = r * size + c;
          const sensoryId = world.sensory && world.sensory[nodeIdx] !== undefined ? world.sensory[nodeIdx] : nodeIdx;

          const hue = (sensoryId * 37) % 360;
          ctx.fillStyle = `hsl(${hue}, 40%, 25%)`;
          ctx.fillRect(c * cellW + 2, r * cellH + 2, cellW - 4, cellH - 4);

          ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
          ctx.strokeRect(c * cellW, r * cellH, cellW, cellH);

          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.font = '10px monospace';
          ctx.fillText(`s:${sensoryId}`, c * cellW + 4, r * cellH + 14);
        }
      }

      if (world.walk && world.walk[step]) {
        const [ax, ay] = world.walk[step];
        const col = ax;
        const row = ay;

        ctx.fillStyle = 'var(--accent, #3b82f6)';
        ctx.beginPath();
        ctx.arc((col + 0.5) * cellW, (row + 0.5) * cellH, Math.min(cellW, cellH) * 0.35, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        const isCorrect = world.pred_correct && world.pred_correct[step] === 1;
        ctx.fillStyle = isCorrect ? '#22c55e' : '#ef4444';
        ctx.font = 'bold 16px sans-serif';
        ctx.fillText(isCorrect ? '✓' : '✕', (col + 0.1) * cellW, (row + 0.8) * cellH);
      }
    },

    renderRateMaps(world) {
      if (!world.rate_maps) return;
      this.drawHeatmap('tem-ratemap-g', world.rate_maps.g, world.size || 8);
      this.drawHeatmap('tem-ratemap-p', world.rate_maps.p, world.size || 8);
    },

    drawHeatmap(canvasId, unitsMatrix, size) {
      const canvas = document.getElementById(canvasId);
      if (!canvas || !unitsMatrix || unitsMatrix.length === 0) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      const cellW = w / size;
      const cellH = h / size;

      ctx.clearRect(0, 0, w, h);
      const rowData = unitsMatrix[0];
      if (!rowData) return;

      const minVal = Math.min(...rowData);
      const maxVal = Math.max(...rowData);
      const range = (maxVal - minVal) || 1;

      for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
          const valIdx = r * size + c;
          const val = rowData[valIdx] !== undefined ? rowData[valIdx] : 0;
          const norm = (val - minVal) / range;

          const red = Math.floor(norm * 255);
          const blue = Math.floor((1 - norm) * 255);
          ctx.fillStyle = `rgb(${red}, 60, ${blue})`;
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
      }
    },

    renderProbe(probe) {
      const panel = document.getElementById('tem-probe-panel');
      const interp = document.getElementById('tem-interpretation');
      if (!probe || !probe.stream) {
        if (panel) panel.textContent = 'No probe stream available.';
        if (interp) interp.textContent = 'Interpretation: N/A';
        return;
      }

      const g = probe.stream.g || {};
      const p = probe.stream.p || {};

      if (panel) {
        panel.innerHTML = `
          <div><strong>g (structure):</strong> spatial_info = ${g.spatial_info ?? '—'} (null: ${g.null ?? '—'}), cross_world_corr = ${g.cross_world_corr ?? '—'}, rsa = ${g.rsa ?? '—'} (null p95: ${g.rsa_null_p95 ?? '—'})</div>
          <div style="margin-top: 6px;"><strong>p (conjunctive):</strong> spatial_info = ${p.spatial_info ?? '—'} (null: ${p.null ?? '—'}), cross_world_corr = ${p.cross_world_corr ?? '—'}, rsa = ${p.rsa ?? '—'} (null p95: ${p.rsa_null_p95 ?? '—'})</div>
        `;
      }

      if (interp) {
        const supported = (g.cross_world_corr !== undefined && g.cross_world_corr > 0.5);
        interp.textContent = `Interpretation: Within-world tuning is real; cross-world sharing is ${supported ? 'supported' : 'not supported'}.`;
      }
    }
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = TEM;
  } else {
    window.NeuroTEM = TEM;
  }

  if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
      TEM.init();
    });
  }
})();
