/** T03: Recall from a partial cue in a Hebbian attractor memory. Mirrors neurosush.memory.hebbian.hebbian_update and attractor; all states and weights are dimensionless. */
(function () {
  'use strict';
  const zeros = n => Array(n).fill(0);
  function matrix(m, n) {
    if (!Array.isArray(m) || m.length !== n || m.some(r => r.length !== n)) throw Error('memory shape');
  }
  function hebbianUpdate(m, p, q, {decay = .9999, rate = .5, clip = 1} = {}) {
    if (!(decay >= 0 && decay <= 1) || !(rate >= 0) || (clip !== null && !(clip > 0))) throw Error('update parameters');
    if (p.length !== q.length) throw Error('pattern shape');
    matrix(m, p.length);
    return m.map((row, i) => row.map((v, j) => {
      const u = decay * v + rate * (p[i] - q[i]) * (p[j] + q[j]);
      return clip === null ? u : Math.max(-clip, Math.min(clip, u));
    }));
  }
  function attractor(cue, m, iterations = 5, {retention = .8, negative_slope = .01, cap = 1, mask = null} = {}) {
    if (!Number.isInteger(iterations) || iterations < 0) throw Error('iterations');
    matrix(m, cue.length);
    if (mask && mask.length !== cue.length) throw Error('mask shape');
    let h = cue.slice();
    for (let k = 0; k < iterations; k++) h = m.map((row, i) => {
      const v = retention * h[i] + row.reduce((s, w, j) => s + w * h[j], 0);
      const updated = Math.min(cap, v >= 0 ? v : negative_slope * v);
      return mask && !mask[i] ? cue[i] : updated;
    });
    return h;
  }
  function dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function overlap(a, b) {
    const den = norm(a) * norm(b);
    return den < 1e-8 ? 0 : dot(a, b) / den;
  }
  function createPatterns(k, n = 16, seed = 42) {
    let s = seed;
    const random = () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
    const patterns = [];
    for (let i = 0; i < k; i++) {
      const p = zeros(n);
      const indices = Array.from({length: n}, (_, idx) => idx);
      for (let j = indices.length - 1; j > 0; j--) {
        const r = Math.floor(random() * (j + 1));
        [indices[j], indices[r]] = [indices[r], indices[j]];
      }
      for (let j = 0; j < 4; j++) p[indices[j]] = 1.0;
      patterns.push(p);
    }
    return patterns;
  }
  function corruptCue(pattern, corruptionFraction, seed = 123) {
    let s = seed;
    const random = () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
    return pattern.map(v => {
      if (v > 0 && random() < corruptionFraction) return 0.0;
      return v;
    });
  }

  const core = {hebbianUpdate, attractor, createPatterns, corruptCue, overlap};

  function mount(el, ctx) {
    const doc = el.ownerDocument;
    el.closest(".atlas-plate").classList.add("t03-shell");
    el.style.aspectRatio = 'auto';
    el.style.maxHeight = 'none';
    el.style.overflow = 'visible';
    el.innerHTML = `<style>
      .atlas-plate.t03-shell{height:auto;max-height:none}
      .t03-shell .atlas-main{flex:none}
      .t03{padding:20px;color:var(--ink);font-family:var(--font-sans)}
      .t03 h2{font:24px var(--font-serif);margin:0 0 12px}.t03 h3{font-size:14px;margin:0 0 10px}
      .t03-controls{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0;align-items:center}
      .t03 button,.t03 select,.t03 input{background:var(--paper);color:var(--ink);border:1px solid var(--border);padding:6px 10px}
      .t03 :focus-visible{outline:3px solid var(--focus-ring)}
      .t03-panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
      .t03 section{padding:14px;background:var(--paper-card);border:1px solid var(--border);min-width:0}
      .t03 canvas{width:100%;max-width:200px;image-rendering:pixelated}
      .t03 code{overflow-wrap:anywhere;font:12px var(--font-mono)}
      .t03 p{line-height:1.5}
      .t03-badge{color:var(--exc);font:12px var(--font-mono)}
      .t03-warning{color:#d97706;font-weight:600;margin-top:8px}
      .t03-bars{display:flex;gap:4px;align-items:flex-end;height:60px;padding-top:10px;border-bottom:1px solid var(--border)}
      .t03-bar{flex:1;background:var(--exc);transition:height .2s}
      @media(max-width:600px){.t03{padding:12px}.t03-panels{grid-template-columns:1fr}}
    </style><div class="t03">
      <span class="t03-badge">TEACHING · Fast Hebbian attractor memory & partial cue recall</span>
      <h2>Recall from a partial cue</h2>
      <p>Store binary sparse patterns into a Hebbian matrix. Corrupt or occlude a cue and watch attractor iterations recover the pattern—or show interference depending on the stored patterns and retained cue.</p>
      <div class="t03-controls">
        <label>Patterns K <select data-patterns>${[1,2,3,4,5,6,7,8].map(n => `<option ${n===4?'selected':''}>${n}</option>`).join('')}</select></label>
        <label>Test target <select data-target><option value="0">Pattern 1</option></select></label>
        <label>Corruption <input type="range" data-corruption min="0" max="0.9" step="0.1" value="0.5" aria-label="Probability of deleting each active cue bit"><span data-corruption-val>50%</span></label>
        <label>Iterations <select data-iterations>${[0,1,2,3,5,8,10].map(n => `<option ${n===5?'selected':''}>${n}</option>`).join('')}</select></label>
        <button data-reset>New random patterns</button>
      </div>
      <div class="t03-panels">
        <section>
          <h3>(a) Memory Matrix M (16 × 16)</h3>
          <canvas data-m width="16" height="16" role="img" aria-label="Hebbian weight matrix heatmap"></canvas>
          <p data-capacity></p>
          <div data-warning></div>
        </section>
        <section>
          <h3>(b) Pattern, Cue & Recovery</h3>
          <p><strong>Target pattern p:</strong> <code data-target-str></code></p>
          <p><strong>Corrupted cue c:</strong> <code data-cue-str></code></p>
          <p><strong>Retrieved state h:</strong> <code data-retrieved-str></code></p>
        </section>
        <section style="grid-column:1/-1">
          <h3>(c) Cosine similarity with target over attractor iterations (k = 0 .. N)</h3>
          <div class="t03-bars" data-overlap-bars role="img" aria-label="Overlap curve over iterations"></div>
          <p data-overlap-desc></p>
        </section>
      </div>
      <p style="margin-top:16px">Implemented exactly: the library’s abstract Hebbian update and attractor retrieval equations (<code>HebbianMemory</code>, <code>hebbian_update</code>, <code>attractor</code>). Simplified: 16 units, sparse binary patterns, fixed retention and cap. This is a teaching model of the memory component only and makes no claim about grid cells.</p>
      <a href="https://github.com/soroushdeimi/neuroSush/blob/main/docs/TEM.md">Read docs/TEM.md</a>
    </div>`;

    ctx.margin.innerHTML = `<h3>Hebbian attractor retrieval</h3>
      <p>Store: <code>M ← clamp(λM + η(p)(p)ᵀ, −1, 1)</code></p>
      <p>Attractor: <code>h^{(k+1)} ← min(1, leaky_relu(0.8h^{(k)} + Mh^{(k)}, 0.01))</code></p>
      <p>A partial or occluded cue starts the network state. With predicted pattern zero, storage creates symmetric nonnegative weights. Recurrent feedback may complete a cue or spread activity into other stored patterns; exact recovery is not guaranteed.</p>
      <p><strong>Interference:</strong> Recall depends on pattern overlap, cue size, weight clipping and recurrence settings. No universal K/units capacity law is established for this model.</p>
      <div class="atlas-margin-section" id="t03-margin-learn">
        <details ${window.innerWidth >= 640 ? 'open' : ''} style="font-size: 11px; line-height: 1.4; margin-top: 12px;">
          <summary style="font-weight: 600; cursor: pointer; margin-bottom: 6px;">Learn: the math, the brain, the ML analogy</summary>
          <div style="display: flex; flex-direction: column; gap: 6px; margin-top: 6px;">
            <div><b>What you see:</b> Sparse binary patterns (4 active out of 16 units), Hebbian weight matrix M (16×16), corrupted cue c, retrieved state h, and cosine similarity over attractor iterations.</div>
            <div><b>The math:</b> M_new = clip(λM + η·p·pᵀ, -1, 1). Retrieval: h_new = min(cap, leakyReLU(κh + Mh, α)). Here λ = 0.9999, η = 0.5, κ = 0.8, all states and weights are dimensionless.</div>
            <div><b>In the brain:</b> Recurrent autoassociative memory model of hippocampal CA3 pattern completion and partial cue recall.</div>
            <div><b>Like classic ML:</b> Recurrent autoassociative pattern completion, related to Hopfield memory and denoising autoencoders. Break: no trained denoising loss, learned inhibitory competition, or guaranteed energy descent; 5 iterations do not prove stable convergence.</div>
          </div>
        </details>
      </div>`;

    const q = s => el.querySelector(s);
    let seed = 42;
    function getState() {
      const K = Number(q('[data-patterns]').value);
      const targetIdx = Math.min(K - 1, Number(q('[data-target]').value));
      const corruption = Number(q('[data-corruption]').value);
      const iterations = Number(q('[data-iterations]').value);
      return {K, targetIdx, corruption, iterations};
    }

    function updateTargetSelect(K, currentIdx) {
      const sel = q('[data-target]');
      sel.innerHTML = Array.from({length: K}, (_, i) => `<option value="${i}" ${i === currentIdx ? 'selected' : ''}>Pattern ${i + 1}</option>`).join('');
    }

    function heat(canvas, rows) {
      const css = getComputedStyle(el), c = canvas.getContext('2d');
      c.fillStyle = css.getPropertyValue('--paper'); c.fillRect(0, 0, canvas.width, canvas.height);
      rows.forEach((row, i) => row.forEach((v, j) => {
        c.globalAlpha = Math.min(1, Math.abs(v));
        c.fillStyle = css.getPropertyValue(v < 0 ? '--inh' : '--exc');
        c.fillRect(j, i, 1, 1);
      })); c.globalAlpha = 1;
    }

    function render() {
      const {K, targetIdx, corruption, iterations} = getState();
      updateTargetSelect(K, targetIdx);
      q('[data-corruption-val]').textContent = Math.round(corruption * 100) + '%';

      const patterns = createPatterns(8, 16, seed);
      const activePatterns = patterns.slice(0, K);

      let m = Array.from({length: 16}, () => zeros(16));
      for (const p of activePatterns) {
        m = hebbianUpdate(m, p, zeros(16), {decay: 0.9999, rate: 0.5, clip: 1.0});
      }

      const target = activePatterns[targetIdx];
      const cue = corruptCue(target, corruption, seed + targetIdx * 100);

      let h = cue.slice();
      const overlaps = [overlap(h, target)];
      for (let i = 0; i < iterations; i++) {
        h = attractor(h, m, 1, {retention: 0.8, negative_slope: 0.01, cap: 1.0});
        overlaps.push(overlap(h, target));
      }

      heat(q('[data-m]'), m);
      q('[data-capacity]').textContent = `Stored: ${K} patterns in 16 units.`;
      const warningEl = q('[data-warning]');
      if (K >= 6) {
        warningEl.innerHTML = `<div class="t03-warning">Inspect retrieved state and cosine similarity; interference varies with patterns and cue.</div>`;
      } else {
        warningEl.innerHTML = '';
      }

      q('[data-target-str]').textContent = `[${target.map(v => v ? '1' : '0').join(', ')}]`;
      q('[data-cue-str]').textContent = `[${cue.map(v => v > 0.5 ? '1' : v < 0 ? '-' : '0').join(', ')}]`;
      q('[data-retrieved-str]').textContent = `[${h.map(v => v.toFixed(2)).join(', ')}]`;

      const barsContainer = q('[data-overlap-bars]');
      barsContainer.innerHTML = overlaps.map((ov, idx) => `
        <div style="flex:1;display:flex;flex-direction:column;align-items:center;height:100%;justify-content:flex-end">
          <div style="font-size:10px;margin-bottom:2px">${ov.toFixed(2)}</div>
          <div class="t03-bar" style="width:100%;height:${Math.max(4, Math.round(ov * 50))}px;background:${ov > 0.85 ? 'var(--exc)' : ov > 0.5 ? '#d97706' : 'var(--inh)'}" title="Iteration ${idx}: overlap ${ov.toFixed(3)}"></div>
          <div style="font-size:10px;margin-top:2px">${idx}</div>
        </div>
      `).join('');

      q('[data-overlap-desc]').textContent = `Final cosine similarity with target after ${iterations} iterations: ${overlaps[overlaps.length - 1].toFixed(3)} (${overlaps[overlaps.length - 1] > 0.85 ? 'High cosine similarity (not necessarily exact recall)' : 'Partial recovery'}).`;
    }

    q('[data-reset]').onclick = () => { seed = Math.floor(Math.random() * 10000); render(); };
    for (const s of ['[data-patterns]', '[data-target]', '[data-corruption]', '[data-iterations]']) {
      q(s).onchange = render;
      q(s).oninput = render;
    }
    doc.addEventListener('visibilitychange', () => {});
    new MutationObserver(render).observe(doc.documentElement, {attributes: true, attributeFilter: ['data-theme']});
    render();
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = {core};
  if (typeof Atlas !== 'undefined') Atlas.Plate.register({id:'T03', title:{en:'Recall from a partial cue: fast Hebbian attractor memory'}, source:'live', mount});
})();
