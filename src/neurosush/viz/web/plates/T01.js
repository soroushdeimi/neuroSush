/** T01: imposed structure and world-specific Hebbian binding. Mirrors neurosush.memory.hebbian.hebbian_update and attractor. All states/weights are dimensionless; updates are per observation. */
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
    return [n - 4, n + 4, n % 4 ? n - 1 : -1, n % 4 < 3 ? n + 1 : -1].filter(v => v >= 0 && v < 16);
  }
  function createWorld(random = Math.random, {memory = true, iterations = 5} = {}) {
    const world = {symbols: Array.from({length: 16}, () => Math.floor(random() * 4)),
      pos: 0, seen: new Set(), edges: new Set(), counts: zeros(4), sensory: Array.from({length: 4}, () => zeros(4)),
      m: Array.from({length: 64}, () => zeros(64)), filtered: zeros(4), scores: zeros(3), trials: 0, steps: 0};
    observe(world, memory, iterations);
    return world;
  }
  function cueAt(pos) { return Array.from({length: 64}, (_, i) => Math.floor(i / 4) === pos ? .25 : 0); }
  function observe(w, memory, iterations) {
    const x = zeros(4); x[w.symbols[w.pos]] = 1;
    w.filtered = x.map((v, i) => .9 * v + .1 * w.filtered[i]);
    w.p = Array.from({length: 64}, (_, i) => Math.floor(i / 4) === w.pos ? w.filtered[i % 4] : 0);
    if (memory) w.m = hebbianUpdate(w.m, w.p, attractor(cueAt(w.pos), w.m, iterations));
    w.seen.add(w.pos); w.counts[w.symbols[w.pos]]++;
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
    w.edges.add(edge); w.pos = to; w.steps++; observe(w, memory, iterations);
    return w.last;
  }
  const core = {hebbianUpdate, attractor, neighbors, createWorld, step, cueAt};
  function mount(el, ctx) {
    const doc = el.ownerDocument;
    el.closest(".atlas-plate").classList.add("t01-shell");
    el.style.aspectRatio = 'auto';
    el.style.maxHeight = 'none';
    el.style.overflow = 'visible';
    el.innerHTML = `<style>
      .atlas-plate.t01-shell{height:auto;max-height:none}
      .t01-shell .atlas-main{flex:none}
      .t01 canvas[data-p]{width:60px;height:240px}
      .t01{padding:20px;color:var(--ink);font-family:var(--font-sans)}
      .t01 h2{font:24px var(--font-serif);margin:0 0 12px}.t01 h3{font-size:14px;margin:0 0 10px}
      .t01-controls{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0;align-items:center}
      .t01 button,.t01 select{background:var(--paper);color:var(--ink);border:1px solid var(--border);padding:8px}
      .t01 :focus-visible{outline:3px solid var(--focus-ring)}
      .t01-panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
      .t01 section{padding:14px;background:var(--paper-card);border:1px solid var(--border);min-width:0}
      .t01 svg{width:100%;max-height:240px}.t01 canvas{width:100%;max-width:260px;image-rendering:pixelated}
      .t01 code{overflow-wrap:anywhere;font:12px var(--font-mono)}.t01 p{line-height:1.5}
      .t01 table{width:100%;font-size:13px;text-align:left}.t01 td{padding:5px 0}
      .t01-badge{color:var(--exc);font:12px var(--font-mono)}
      @media(max-width:600px){.t01{padding:12px}.t01-panels{grid-template-columns:1fr}}
    </style><div class="t01">
      <span class="t01-badge">TEACHING · JS model, parity-tested</span>
      <h2>Bind a sensation to a place</h2>
      <p>Walk the graph. Close a loop through a new edge. Can memory recall what comes next?</p>
      <div class="t01-controls">
        <button data-step aria-label="Step one random edge">Step</button><button data-run aria-pressed="false">Run</button>
        <label>Speed <select data-speed aria-label="Steps per second"><option>1</option><option selected>3</option><option>6</option></select></label>
        <button data-reset>New world / reset memory</button>
        <label><input type="checkbox" data-memory checked> Memory on</label>
        <label>Attractor iterations <select data-iterations aria-label="Attractor iterations">${[0,1,3,5,8].map(n => `<option ${n === 5 ? 'selected' : ''}>${n}</option>`).join('')}</select></label>
      </div>
      <div class="t01-panels">
        <section><h3>World · 4 × 4, no wrap</h3><svg viewBox="0 0 240 240" role="img" aria-label="Graph with agent and revealed sensory symbols"></svg><p data-position></p></section>
        <section><h3>(a) Structure g · GIVEN / imposed</h3><p>Moves update the exact graph-node index; this is a supplied position code, not learned or noisy neural path integration. One of 16 location units is active. No transition weights are learned.</p><code data-g></code><h3>(b) Sensory x</h3><p data-x></p><code data-filter></code></section>
        <section><h3>(c) Conjunction p = g ⊗ filtered x</h3><p>16 rows × 4 sensory channels. Filter: xᶠ ← 0.1xᶠ + 0.9x.</p><canvas data-p width="4" height="16" role="img" aria-label="Conjunctive pattern, location by sensory channel"></canvas></section>
        <section><h3>(d) World-specific memory M · 64 × 64</h3><canvas data-m width="64" height="64" role="img" aria-label="Hebbian matrix, positive and negative weights"></canvas><p>Positive and negative abstract memory coefficients; colours indicate sign and intensity indicates magnitude. These are not identified excitatory/inhibitory neurons.</p></section>
        <section style="grid-column:1/-1"><h3>(e) Predict before revealing</h3><p data-last aria-live="polite"></p><table><thead><tr><th>Predictor</th><th>Correct / eligible</th><th>Accuracy</th></tr></thead><tbody data-scores></tbody></table><p>Eligible = first traversal of a directed edge to an already seen node. This score tests recall at previously visited destinations, not generalization to unvisited nodes; each eligible directed edge is scored once. All predictors use only past observations; ties choose A. Sensory-only predicts from counts of successive symbols, without position. Frequency predicts the most common past symbol.</p></section>
      </div><p>Implemented exactly: the library’s abstract Hebbian update and iterative retrieval equations. Simplified: a small graph, imposed g and hand-set binding/filter. This is not the trained TEM model and shows no grid cells. New world keeps the imposed transition structure and resets sensory bindings, memory and scores.</p>
      <a href="https://github.com/soroushdeimi/neuroSush/blob/main/docs/TEM.md">Read docs/TEM.md</a>
    </div>`;
    ctx.margin.innerHTML = `<h3>Binding and recall</h3><p>First predict with the old M; then reveal x and store p.</p>
      <p><code>M ← clamp(λM + η(p − p̂)(p + p̂)ᵀ, −1, 1)</code></p>
      <p>λ = 0.9999 per observation, η = 0.5; states and weights are dimensionless</p><p><code>h ← min(1, leaky_relu(0.8h + Mh, 0.01))</code></p>
      <p>Start h from g tiled across four sensory channels (0.25 each). Read the destination row. p̂ uses the same retrieval before storing.</p>
      <p>Memory off disables retrieval and updates; existing M is retained until reset. Scores restart when memory or iterations change.</p>
      <div class="atlas-margin-section" id="t01-margin-learn">
        <details ${window.innerWidth >= 640 ? 'open' : ''} style="font-size: 11px; line-height: 1.4; margin-top: 12px;">
          <summary style="font-weight: 600; cursor: pointer; margin-bottom: 6px;">Learn: the math, the brain, the ML analogy</summary>
          <div style="display: flex; flex-direction: column; gap: 6px; margin-top: 6px;">
            <div><b>What you see:</b> A 4×4 graph world with an agent walking edges, conjunctive position-sensory pattern p, Hebbian binding matrix M (64×64), and recall accuracy scores across predictors.</div>
            <div><b>The math:</b> M_new = clip(λM + η(p - p̂)(p + p̂)ᵀ, -c, c). Retrieval: h_new = min(cap, leakyReLU(κh + Mh, α)). Here λ = 0.9999 per observation, η = 0.5, κ = 0.8, all states and weights are dimensionless.</div>
            <div><b>In the brain:</b> Inspired by Tolman-Eichenbaum Machine (TEM) models of hippocampus and entorhinal cortex, simulating Hebbian association between location codes and sensory inputs.</div>
            <div><b>Like classic ML:</b> Fast associative key/value memory conditioned on a supplied location code. Break: spatial structure is provided exactly, not learned; 5 recurrent updates need not converge; eligible-score accuracy cannot establish trained TEM generalization or grid-cell emergence.</div>
          </div>
        </details>
      </div>`;
    const q = s => el.querySelector(s), symbols = ['A','B','C','D'];
    let w = createWorld(), timer = null;
    const opts = () => ({memory:q('[data-memory]').checked, iterations:Number(q('[data-iterations]').value)});
    function heat(canvas, rows) {
      const css = getComputedStyle(el), c = canvas.getContext('2d');
      c.fillStyle = css.getPropertyValue('--paper'); c.fillRect(0,0,canvas.width,canvas.height);
      rows.forEach((row,i) => row.forEach((v,j) => {
        c.globalAlpha = Math.min(1,Math.abs(v)); c.fillStyle = css.getPropertyValue(v < 0 ? '--inh' : '--exc'); c.fillRect(j,i,1,1);
      })); c.globalAlpha = 1;
    }
    function render() {
      let svg = '';
      for (let n=0;n<16;n++) {
        const x=30+(n%4)*60,y=30+Math.floor(n/4)*60;
        neighbors(n).filter(v=>v>n).forEach(v => { svg += `<path d="M${x},${y} L${30+v%4*60},${30+Math.floor(v/4)*60}" stroke="var(--border)"/>`; });
        svg += `<circle cx="${x}" cy="${y}" r="20" fill="var(--paper)" stroke="var(--${n===w.pos?'exc':'border'})" stroke-width="${n===w.pos?4:1}"/><text x="${x}" y="${y+5}" text-anchor="middle" fill="var(--ink)">${w.seen.has(n)?symbols[w.symbols[n]]:'?'}</text>`;
      }
      q('svg').innerHTML = svg;
      q('[data-position]').textContent = `Agent: (${w.pos%4}, ${Math.floor(w.pos/4)}) · step ${w.steps} · ${w.seen.size}/16 seen`;
      q('[data-g]').textContent = Array.from({length:16},(_,i)=>i===w.pos?1:0).join(' ');
      q('[data-x]').textContent = `Observed ${symbols[w.symbols[w.pos]]} · x = [${symbols.map((_,i)=>Number(i===w.symbols[w.pos])).join(', ')}]`;
      q('[data-filter]').textContent = `filtered x = [${w.filtered.map(v=>v.toFixed(3)).join(', ')}]`;
      heat(q('[data-p]'),Array.from({length:16},(_,i)=>w.p.slice(i*4,i*4+4))); heat(q('[data-m]'),w.m);
      const last=w.last;
      q('[data-last]').textContent = last ? `${last.from} → ${last.to}: predicted ${last.guesses.map(i=>symbols[i]).join(' / ')} (memory / sensory / frequency); revealed ${symbols[last.actual]}. ${last.eligible?'Scored: new edge to a seen node.':'Not scored.'}` : 'Step to predict, then reveal a destination. Unvisited symbols stay hidden.';
      q('[data-scores]').innerHTML = ['Hebbian memory','Sensory only','Frequency'].map((s,i)=>`<tr><td>${s}</td><td>${w.scores[i]} / ${w.trials}</td><td>${w.trials?(100*w.scores[i]/w.trials).toFixed(1)+'%':'—'}</td></tr>`).join('');
    }
    function advance() { step(w,opts()); render(); }
    function stop() { clearInterval(timer); timer=null; q('[data-run]').textContent='Run'; q('[data-run]').setAttribute('aria-pressed','false'); }
    function run() { timer=setInterval(advance,1000/Number(q('[data-speed]').value)); q('[data-run]').textContent='Pause'; q('[data-run]').setAttribute('aria-pressed','true'); }
    q('[data-step]').onclick=advance;
    q('[data-run]').onclick=()=>timer?stop():run();
    q('[data-speed]').onchange=()=>{if(timer){stop();run();}};
    q('[data-reset]').onclick=()=>{stop();w=createWorld(Math.random,opts());render();};
    for(const s of ['[data-memory]','[data-iterations]']) q(s).onchange=()=>{w.trials=0;w.scores=zeros(3);w.last=null;render();};
    doc.addEventListener('visibilitychange',()=>{if(doc.hidden)stop();});
    new MutationObserver(render).observe(doc.documentElement,{attributes:true,attributeFilter:['data-theme']});
    render();
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {core};
  if (typeof Atlas !== 'undefined') Atlas.Plate.register({id:'T01',title:{en:'TEM: binding structure and sensation in a Hebbian memory'},source:'live',mount});
})();
