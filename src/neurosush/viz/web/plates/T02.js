/** T02: imposed position-indexed associative lookup; mirrors neurosush.memory.hebbian.hebbian_update and attractor. */
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
      return mask && !mask[i] ? cue[i] : Math.min(cap, v >= 0 ? v : negative_slope * v);
    });
    return h;
  }
  const argmax = a => a.indexOf(Math.max(...a));
  function neighbors(n) {
    return [n - 6, n + 6, n % 6 ? n - 1 : -1, n % 6 < 5 ? n + 1 : -1].filter(v => v >= 0 && v < 36);
  }
  function createWorld(random = Math.random, {memory = true, iterations = 5, worldType = 'A', sharedStructure = true} = {}) {
    const symbolsA = Array.from({length: 36}, () => Math.floor(random() * 4));
    const symbolsB = Array.from({length: 36}, () => Math.floor(random() * 4));
    const world = {
      worldType,
      sharedStructure,
      symbolsA,
      symbolsB,
      get symbols() { return this.worldType === 'A' ? this.symbolsA : this.symbolsB; },
      pos: 0,
      seen: new Set(),
      edges: new Set(),
      counts: zeros(4),
      sensory: Array.from({length: 4}, () => zeros(4)),
      m: Array.from({length: 144}, () => zeros(144)),
      filtered: zeros(4),
      scores: zeros(3),
      trials: 0,
      steps: 0
    };
    observe(world, memory, iterations);
    return world;
  }
  function cueAt(pos) {
    return Array.from({length: 144}, (_, i) => Math.floor(i / 4) === pos ? .25 : 0);
  }
  function observe(w, memory, iterations) {
    const x = zeros(4);
    x[w.symbols[w.pos]] = 1;
    w.filtered = x.map((v, i) => .9 * v + .1 * w.filtered[i]);
    w.p = Array.from({length: 144}, (_, i) => Math.floor(i / 4) === w.pos ? w.filtered[i % 4] : 0);
    if (memory) w.m = hebbianUpdate(w.m, w.p, attractor(cueAt(w.pos), w.m, iterations));
    w.seen.add(w.pos);
    w.counts[w.symbols[w.pos]]++;
  }
  function step(w, {memory = true, iterations = 5, target = null, random = Math.random} = {}) {
    const options = neighbors(w.pos), from = w.pos;
    const to = target === null ? options[Math.floor(random() * options.length)] : target;
    if (!options.includes(to)) throw Error('not an edge');
    const edge = `${from}>${to}`, eligible = w.seen.has(to) && !w.edges.has(edge);
    const h = memory ? attractor(cueAt(to), w.m, iterations) : cueAt(to);
    const guesses = [argmax(h.slice(to * 4, to * 4 + 4)), argmax(w.sensory[w.symbols[from]]), argmax(w.counts)];
    const actual = w.symbols[to];
    if (eligible) { w.trials++; guesses.forEach((g, i) => { w.scores[i] += Number(g === actual); }); }
    w.last = {from, to, guesses, actual, eligible};
    w.sensory[w.symbols[from]][actual]++;
    w.edges.add(edge);
    w.pos = to;
    w.steps++;
    observe(w, memory, iterations);
    return w.last;
  }
  function switchWorld(w, type, memory = true, iterations = 5) {
    w.worldType = type;
    w.pos = 0;
    w.seen = new Set();
    w.edges = new Set();
    w.counts = zeros(4);
    w.sensory = Array.from({length: 4}, () => zeros(4));
    w.filtered = zeros(4);
    w.trials = 0;
    w.scores = zeros(3);
    w.last = null;
    w.steps = 0;
    w.m = Array.from({length: 144}, () => zeros(144));
    observe(w, memory, iterations);
    return w;
  }
  const core = {hebbianUpdate, attractor, neighbors, createWorld, step, cueAt, switchWorld};
  function mount(el, ctx) {
    const doc = el.ownerDocument;
    el.closest(".atlas-plate").classList.add("t02-shell");
    el.style.aspectRatio = 'auto';
    el.style.maxHeight = 'none';
    el.style.overflow = 'visible';
    el.innerHTML = `<style>
      .atlas-plate.t02-shell{height:auto;max-height:none}
      .t02-shell .atlas-main{flex:none}
      .t02 canvas[data-p]{width:60px;height:240px}
      .t02{padding:20px;color:var(--ink);font-family:var(--font-sans)}
      .t02 h2{font:24px var(--font-serif);margin:0 0 12px}.t02 h3{font-size:14px;margin:0 0 10px}
      .t02-controls{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0;align-items:center}
      .t02 button,.t02 select{background:var(--paper);color:var(--ink);border:1px solid var(--border);padding:8px}
      .t02 :focus-visible{outline:3px solid var(--focus-ring)}
      .t02-panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
      .t02 section{padding:14px;background:var(--paper-card);border:1px solid var(--border);min-width:0}
      .t02 svg{width:100%;max-height:260px}.t02 canvas{width:100%;max-width:260px;image-rendering:pixelated}
      .t02 code{overflow-wrap:anywhere;font:12px var(--font-mono)}.t02 p{line-height:1.5}
      .t02 table{width:100%;font-size:13px;text-align:left}.t02 td{padding:5px 0}
      .t02-badge{color:var(--exc);font:12px var(--font-mono)}
      @media(max-width:600px){.t02{padding:12px}.t02-panels{grid-template-columns:1fr}}
    </style><div class="t02">
      <span class="t02-badge">TEACHING: structure imposed, not learned; no grid-cell claim</span>
      <h2>Same map, new world: remapping with an imposed structure</h2>
      <p>Walk the 6 × 6 graph. Switch between World A and World B. How does a position-indexed memory retrieve sensory observations at previously visited locations?</p>
      <div class="t02-controls">
        <button data-step aria-label="Step one random edge">Step</button><button data-run aria-pressed="false">Run</button>
        <label>Speed <select data-speed aria-label="Steps per second"><option>1</option><option selected>3</option><option>6</option></select></label>
        <button data-world>Switch to World B</button>
        <button data-reset>Reset World</button>
        <label><input type="checkbox" data-memory checked> Memory on</label>
        <label><input type="checkbox" data-shared disabled checked> Structure is fixed (no structure ablation implemented)</label>
        <label>Iterations <select data-iterations aria-label="Attractor iterations">${[0,1,3,5].map(n => `<option ${n === 3 ? 'selected' : ''}>${n}</option>`).join('')}</select></label>
      </div>
      <div class="t02-panels">
        <section><h3>World (6 × 6) · <span data-world-label>World A</span></h3><svg viewBox="0 0 360 360" role="img" aria-label="6 by 6 graph with agent and revealed sensory symbols"></svg><p data-position></p></section>
        <section><h3>(a) Structure g · Imposed 36-dim</h3><p>Actions path-integrate an exact position across 36 nodes.</p><code data-g></code><h3>(b) Sensory x</h3><p data-x></p><code data-filter></code></section>
        <section><h3>(c) Conjunction p = g ⊗ filtered x</h3><p>36 rows × 4 sensory channels (144 dims).</p><canvas data-p width="4" height="36" role="img" aria-label="Conjunctive pattern"></canvas></section>
        <section><h3>(d) Hebbian Memory M · 144 × 144</h3><canvas data-m width="144" height="144" role="img" aria-label="Hebbian matrix"></canvas><p>Positive and negative abstract couplings; individual biological cell types are not represented.</p></section>
        <section style="grid-column:1/-1"><h3>(e) Prediction comparison on new edges</h3><p data-last aria-live="polite"></p><table><thead><tr><th>Predictor</th><th>Correct / eligible</th><th>Accuracy</th></tr></thead><tbody data-scores></tbody></table><p>Eligible = first traversal of a directed edge to a seen node. Position-indexed memory retrieves previously observed sensations; all predictors are evaluated before observing the destination; sensory-only uses symbol transition counts; frequency uses symbol base rates.</p></section>
      </div><p>TEACHING: structure imposed, not learned; no grid-cell claim. Switching worlds resets memory and sensory bindings while preserving the imposed position code.</p>
      <a href="https://github.com/soroushdeimi/neuroSush/blob/main/docs/TEM.md">Read docs/TEM.md</a>
    </div>`;
    ctx.margin.innerHTML = `<h3>Remapping & Generalisation</h3><p>The same spatial code g binds new sensory observations in World B.</p>
      <p><code>M ← clamp(λM + η(p − p̂)(p + p̂)ᵀ, −1, 1)</code></p>
      <p>Exact position cues retrieve within-world sensory associations at familiar destinations; this does not test learned structural transfer.</p>
      <div class="atlas-margin-section" id="t02-margin-learn">
        <details ${window.innerWidth >= 640 ? 'open' : ''} style="font-size: 11px; line-height: 1.4; margin-top: 12px;">
          <summary style="font-weight: 600; cursor: pointer; margin-bottom: 6px;">Learn: the math, the brain, the ML analogy</summary>
          <div style="display: flex; flex-direction: column; gap: 6px; margin-top: 6px;">
            <div><b>What you see:</b> A 6×6 graph world with 36 position locations and 4 sensory symbols, conjunctive position-sensory pattern p (144 dims), Hebbian memory M (144×144), and prediction accuracy scores.</div>
            <div><b>The math:</b> M_new = clip(λM + η(p - p̂)(p + p̂)ᵀ, -c, c). Retrieval: h_new = min(cap, leakyReLU(κh + Mh, α)). Here λ = 0.9999 per observation, η = 0.5, κ = 0.8, all states and weights are dimensionless.</div>
            <div><b>In the brain:</b> Inspired by Tolman-Eichenbaum Machine (TEM) models of hippocampus and entorhinal cortex, illustrating position-indexed associative lookup across environments.</div>
            <div><b>Like classic ML:</b> Associative key-value memory keyed by exact position. Break: supplying the key bypasses learning spatial structure, localization, and cross-world transfer.</div>
          </div>
        </details>
      </div>`;
    const q = s => el.querySelector(s), symbols = ['A','B','C','D'];
    let w = createWorld(Math.random, {memory:q('[data-memory]').checked, iterations:Number(q('[data-iterations]').value), sharedStructure:q('[data-shared]').checked}), timer = null;
    const opts = () => ({memory:q('[data-memory]').checked, iterations:Number(q('[data-iterations]').value), sharedStructure:q('[data-shared]').checked});
    function heat(canvas, rows) {
      const css = getComputedStyle(el), c = canvas.getContext('2d');
      c.fillStyle = css.getPropertyValue('--paper'); c.fillRect(0,0,canvas.width,canvas.height);
      rows.forEach((row,i) => row.forEach((v,j) => {
        c.globalAlpha = Math.min(1,Math.abs(v)); c.fillStyle = css.getPropertyValue(v < 0 ? '--inh' : '--exc'); c.fillRect(j,i,1,1);
      })); c.globalAlpha = 1;
    }
    function render() {
      let svg = '';
      for (let n=0;n<36;n++) {
        const x=30+(n%6)*60,y=30+Math.floor(n/6)*60;
        neighbors(n).filter(v=>v>n).forEach(v => { svg += `<path d="M${x},${y} L${30+v%6*60},${30+Math.floor(v/6)*60}" stroke="var(--border)"/>`; });
        svg += `<circle cx="${x}" cy="${y}" r="16" fill="var(--paper)" stroke="var(--${n===w.pos?'exc':'border'})" stroke-width="${n===w.pos?3:1}"/><text x="${x}" y="${y+4}" font-size="11" text-anchor="middle" fill="var(--ink)">${w.seen.has(n)?symbols[w.symbols[n]]:'?'}</text>`;
      }
      q('svg').innerHTML = svg;
      q('[data-world-label]').textContent = `World ${w.worldType}`;
      q('[data-position]').textContent = `Agent: (${w.pos%6}, ${Math.floor(w.pos/6)}) · step ${w.steps} · ${w.seen.size}/36 seen`;
      q('[data-g]').textContent = Array.from({length:36},(_,i)=>i===w.pos?1:0).join(' ');
      q('[data-x]').textContent = `Observed ${symbols[w.symbols[w.pos]]} · x = [${symbols.map((_,i)=>Number(i===w.symbols[w.pos])).join(', ')}]`;
      q('[data-filter]').textContent = `filtered x = [${w.filtered.map(v=>v.toFixed(3)).join(', ')}]`;
      heat(q('[data-p]'),Array.from({length:36},(_,i)=>w.p.slice(i*4,i*4+4))); heat(q('[data-m]'),w.m);
      const last=w.last;
      q('[data-last]').textContent = last ? `${last.from} → ${last.to}: predicted ${last.guesses.map(i=>symbols[i]).join(' / ')} (memory / sensory / frequency); revealed ${symbols[last.actual]}. ${last.eligible?'Scored: new edge to a seen node.':'Not scored.'}` : 'Step to predict, then reveal a destination.';
      q('[data-scores]').innerHTML = ['Hebbian memory','Sensory only','Frequency'].map((s,i)=>`<tr><td>${s}</td><td>${w.scores[i]} / ${w.trials}</td><td>${w.trials?(100*w.scores[i]/w.trials).toFixed(1)+'%':'—'}</td></tr>`).join('');
    }
    function advance() { step(w,opts()); render(); }
    function stop() { clearInterval(timer); timer=null; q('[data-run]').textContent='Run'; q('[data-run]').setAttribute('aria-pressed','false'); }
    function run() { timer=setInterval(advance,1000/Number(q('[data-speed]').value)); q('[data-run]').textContent='Pause'; q('[data-run]').setAttribute('aria-pressed','true'); }
    q('[data-step]').onclick=advance;
    q('[data-run]').onclick=()=>timer?stop():run();
    q('[data-speed]').onchange=()=>{if(timer){stop();run();}};
    q('[data-reset]').onclick=()=>{stop();w=createWorld(Math.random,opts());w.worldType=q('[data-world-label]').textContent.includes('B')?'B':'A';render();};
    q('[data-world]').onclick=()=>{
      stop();
      const nextType = w.worldType === 'A' ? 'B' : 'A';
      q('[data-world]').textContent = nextType === 'A' ? 'Switch to World B' : 'Switch to World A';
      switchWorld(w, nextType, q('[data-memory]').checked, Number(q('[data-iterations]').value));
      render();
    };
    for(const s of ['[data-memory]','[data-iterations]']) q(s).onchange=()=>{w.trials=0;w.scores=zeros(3);w.last=null;render();};
    doc.addEventListener('visibilitychange',()=>{if(doc.hidden)stop();});
    new MutationObserver(render).observe(doc.documentElement,{attributes:true,attributeFilter:['data-theme']});
    render();
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {core};
  if (typeof Atlas !== 'undefined') Atlas.Plate.register({id:'T02',title:{en:'Same map, new world: remapping with an imposed structure'},source:'live',mount});
})();
