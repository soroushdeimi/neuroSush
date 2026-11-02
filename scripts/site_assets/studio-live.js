(function () {
  'use strict';

  // Seeded PRNG (mulberry32)
  function mulberry32(a) {
    return function () {
      let t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  let rng = mulberry32(42);

  // Network State
  let config = {
    seed: 42,
    model: 'lif',
    tau: 20.0,
    v_th: -40.0,
    v_rest: -70.0,
    v_reset: -75.0,
    resistance: 10.0,
    poisson_rate: 10.0,
    current: 2.0,
    stdp_enabled: false,
    a_plus: 0.02,
    a_minus: 0.021,
    populations: [
      { id: 0, name: 'Exc1', size: 40, excitatory: true },
      { id: 1, name: 'Inh1', size: 10, excitatory: false },
    ],
    connections: [
      { src: 0, dst: 0, p: 0.2, weight: 0.5, delay: 1 },
      { src: 0, dst: 1, p: 0.3, weight: 0.8, delay: 1 },
      { src: 1, dst: 0, p: 0.3, weight: -0.6, delay: 1 },
    ],
  };

  // Simulation Runtime State
  let simState = {
    stepCount: 0,
    isPlaying: false,
    speed: 1,
    neurons: [], // flat array of neuron states across populations
    synapses: [], // list of connections with weight matrices
    raster: [], // recent spikes history for raster
    voltages: [], // history for selected neuron
    rates: [], // history for population rates
    weights: [], // histogram / sample weights
    selectedNeuron: 0,
    pulses: [], // active traveling pulse packets
  };

  // Presets definition
  const PRESETS = {
    balanced: {
      model: 'lif',
      tau: 20.0,
      v_th: -40.0,
      v_rest: -70.0,
      v_reset: -75.0,
      resistance: 10.0,
      poisson_rate: 12.0,
      current: 2.5,
      stdp_enabled: false,
      a_plus: 0.02,
      a_minus: 0.021,
      populations: [
        { id: 0, name: 'Exc', size: 50, excitatory: true },
        { id: 1, name: 'Inh', size: 12, excitatory: false },
      ],
      connections: [
        { src: 0, dst: 0, p: 0.2, weight: 0.4, delay: 1 },
        { src: 0, dst: 1, p: 0.3, weight: 0.6, delay: 1 },
        { src: 1, dst: 0, p: 0.3, weight: -0.5, delay: 1 },
      ],
    },
    wta: {
      model: 'lif',
      tau: 20.0,
      v_th: -40.0,
      v_rest: -70.0,
      v_reset: -75.0,
      resistance: 10.0,
      poisson_rate: 15.0,
      current: 3.0,
      stdp_enabled: true,
      a_plus: 0.02,
      a_minus: 0.021,
      populations: [
        { id: 0, name: 'PopA', size: 30, excitatory: true },
        { id: 1, name: 'PopB', size: 30, excitatory: true },
        { id: 2, name: 'InhWTA', size: 15, excitatory: false },
      ],
      connections: [
        { src: 0, dst: 2, p: 0.4, weight: 0.5, delay: 1 },
        { src: 1, dst: 2, p: 0.4, weight: 0.5, delay: 1 },
        { src: 2, dst: 0, p: 0.5, weight: -0.8, delay: 1 },
        { src: 2, dst: 1, p: 0.5, weight: -0.8, delay: 1 },
      ],
    },
    chain: {
      model: 'elif',
      tau: 15.0,
      v_th: -45.0,
      v_rest: -70.0,
      v_reset: -75.0,
      resistance: 10.0,
      poisson_rate: 8.0,
      current: 2.0,
      stdp_enabled: false,
      a_plus: 0.02,
      a_minus: 0.021,
      populations: [
        { id: 0, name: 'InputL1', size: 25, excitatory: true },
        { id: 1, name: 'RelayL2', size: 25, excitatory: true },
        { id: 2, name: 'OutputL3', size: 25, excitatory: true },
      ],
      connections: [
        { src: 0, dst: 1, p: 0.4, weight: 1.2, delay: 2 },
        { src: 1, dst: 2, p: 0.4, weight: 1.2, delay: 2 },
      ],
    },
  };

  function initSimulation() {
    rng = mulberry32(config.seed);
    simState.stepCount = 0;
    simState.neurons = [];
    simState.raster = [];
    simState.voltages = [];
    simState.rates = [];
    simState.weights = [];
    simState.pulses = [];

    let offset = 0;
    for (let pIdx = 0; pIdx < config.populations.length; pIdx++) {
      const pop = config.populations[pIdx];
      pop.startIdx = offset;
      for (let i = 0; i < pop.size; i++) {
        simState.neurons.push({
          id: offset + i,
          popId: pop.id,
          v: config.v_rest + (rng() - 0.5) * 5.0,
          u: 0.0, // for izhikevich / adex
          refractory: 0.0,
          flashing: 0.0,
          x_trace: 0.0, // STDP pre trace
          v_trace: 0.0, // STDP post trace
          x: 0,
          y: 0,
        });
      }
      offset += pop.size;
    }

    // Initialize synapse weights and delay structures
    simState.synapses = config.connections.map(conn => {
      const srcPop = config.populations.find(p => p.id === conn.src);
      const dstPop = config.populations.find(p => p.id === conn.dst);
      const weights = [];
      if (srcPop && dstPop) {
        for (let i = 0; i < srcPop.size; i++) {
          const row = [];
          for (let j = 0; j < dstPop.size; j++) {
            if (rng() < conn.p) {
              row.push({
                w: conn.weight * (0.8 + 0.4 * rng()),
                delay: conn.delay || 1,
                queue: new Array(conn.delay || 1).fill(0),
              });
            } else {
              row.push(null);
            }
          }
          weights.push(row);
        }
      }
      return { ...conn, weights };
    });

    simState.selectedNeuron = Math.min(0, simState.neurons.length - 1);
  }

  // Neuron model step functions (matching library / Atlas plates)
  function stepNeuron(neuron, I_ext, dt = 1.0) {
    if (neuron.refractory > 0) {
      neuron.refractory -= dt;
      neuron.v = config.v_reset;
      return false;
    }

    let spiked = false;
    if (config.model === 'lif') {
      const dv = dt * ((config.v_rest - neuron.v) + config.resistance * I_ext) / config.tau;
      neuron.v += dv;
      if (neuron.v >= config.v_th) {
        spiked = true;
        neuron.v = config.v_reset;
        neuron.refractory = 2.0;
      }
    } else if (config.model === 'elif') {
      const deltaT = 2.0;
      const thetaRh = config.v_th;
      const expTerm = deltaT * Math.exp(Math.min(50, (neuron.v - thetaRh) / deltaT));
      const dv = dt * ((config.v_rest - neuron.v) + expTerm + config.resistance * I_ext) / config.tau;
      neuron.v += dv;
      if (neuron.v >= config.v_th + 15.0) {
        spiked = true;
        neuron.v = config.v_reset;
        neuron.refractory = 2.0;
      }
    } else if (config.model === 'adex') {
      const deltaT = 2.0;
      const tauW = 30.0;
      const a = 0.5;
      const b = 5.0;
      const dv = dt * ((config.v_rest - neuron.v) + deltaT * Math.exp(Math.min(50, (neuron.v - config.v_th) / deltaT)) - neuron.u + config.resistance * I_ext) / config.tau;
      const du = dt * (a * (neuron.v - config.v_rest) - neuron.u) / tauW;
      neuron.v += dv;
      neuron.u += du;
      if (neuron.v >= config.v_th + 20.0) {
        spiked = true;
        neuron.v = config.v_reset;
        neuron.u += b;
        neuron.refractory = 2.0;
      }
    } else if (config.model === 'izhikevich') {
      // Izhikevich regular spiking (a=0.02, b=0.2, c=-65, d=8)
      const v = neuron.v;
      const u = neuron.u;
      const I = I_ext;
      // 2 substeps per ms
      let vv = v,
        uu = u;
      for (let s = 0; s < 2; s++) {
        const dv = (0.04 * vv * vv + 5 * vv + 140 - uu + I) / 2.0;
        const du = (0.02 * (0.2 * vv - uu)) / 2.0;
        vv += dv;
        uu += du;
      }
      neuron.v = vv;
      neuron.u = uu;
      if (neuron.v >= 30.0) {
        spiked = true;
        neuron.v = -65.0;
        neuron.u += 8.0;
        neuron.refractory = 1.0;
      }
    }
    return spiked;
  }

  function simulateStep() {
    simState.stepCount++;
    const currentSpikes = [];

    // Poisson background & external drive input per neuron
    simState.neurons.forEach(neuron => {
      let I_ext = config.current;
      if (rng() * 1000 < config.poisson_rate * 1.0) {
        I_ext += 5.0; // Poisson spike input current bump
      }

      // Add synaptic input collected from delay queues
      simState.synapses.forEach(syn => {
        const srcPop = config.populations.find(p => p.id === syn.src);
        const dstPop = config.populations.find(p => p.id === syn.dst);
        if (neuron.popId === syn.dst && srcPop && dstPop) {
          const localIdx = neuron.id - dstPop.startIdx;
          if (localIdx >= 0 && localIdx < dstPop.size) {
            for (let i = 0; i < srcPop.size; i++) {
              const conn = syn.weights[i][localIdx];
              if (conn) {
                // Pop spike from queue
                const arriving = conn.queue.shift();
                conn.queue.push(0); // push empty for next
                if (arriving) {
                  I_ext += conn.w * 10.0;
                }
              }
            }
          }
        }
      });

      const spiked = stepNeuron(neuron, I_ext);
      if (spiked) {
        currentSpikes.push(neuron.id);
        neuron.flashing = 1.0;
        if (simState.selectedNeuron === neuron.id) {
          neuron.v = 30.0; // visual spike peak for selected trace
        }
      } else {
        if (neuron.flashing > 0) neuron.flashing -= 0.1;
      }
    });

    // Propagate spikes into synapse delay queues & STDP
    simState.synapses.forEach(syn => {
      const srcPop = config.populations.find(p => p.id === syn.src);
      const dstPop = config.populations.find(p => p.id === syn.dst);
      if (!srcPop || !dstPop) return;

      for (let i = 0; i < srcPop.size; i++) {
        const srcNeuron = simState.neurons[srcPop.startIdx + i];
        const srcSpiked = currentSpikes.includes(srcNeuron.id);

        if (srcSpiked) {
          srcNeuron.x_trace = 1.0; // pre trace jump
        } else {
          srcNeuron.x_trace *= Math.exp(-1.0 / 20.0); // decay tau=20
        }

        for (let j = 0; j < dstPop.size; j++) {
          const conn = syn.weights[i][j];
          if (!conn) continue;

          const dstNeuron = simState.neurons[dstPop.startIdx + j];
          const dstSpiked = currentSpikes.includes(dstNeuron.id);

          if (srcSpiked) {
            // enqueue spike at delay index
            const delayIdx = Math.min(conn.queue.length - 1, conn.delay - 1);
            conn.queue[delayIdx] = 1;

            // Traveling pulse packet record for canvas animation
            if (simState.pulses.length < 30 && rng() < 0.2) {
              simState.pulses.push({
                x1: srcNeuron.x,
                y1: srcNeuron.y,
                x2: dstNeuron.x,
                y2: dstNeuron.y,
                progress: 0.0,
                speed: 0.05 + rng() * 0.05,
              });
            }
          }

          if (config.stdp_enabled && srcSpiked) {
            // Post before pre -> LTD
            conn.w -= config.a_minus * dstNeuron.v_trace;
            conn.w = Math.max(0.0, Math.min(5.0, conn.w));
          }
          if (config.stdp_enabled && dstSpiked) {
            // Pre before post -> LTP
            conn.w += config.a_plus * srcNeuron.x_trace;
            conn.w = Math.max(0.0, Math.min(5.0, conn.w));
          }
        }
      }
    });

    // Update post traces for STDP
    simState.neurons.forEach(neuron => {
      const spiked = currentSpikes.includes(neuron.id);
      if (spiked) {
        neuron.v_trace = 1.0;
      } else {
        neuron.v_trace *= Math.exp(-1.0 / 20.0);
      }
    });

    // Record raster and trace history
    simState.raster.push(currentSpikes);
    if (simState.raster.length > 80) simState.raster.shift();

    const sel = simState.neurons[simState.selectedNeuron];
    if (sel) {
      simState.voltages.push(sel.v);
      if (simState.voltages.length > 120) simState.voltages.shift();
    }

    // Population firing rates
    const rates = config.populations.map(pop => {
      const spkCount = currentSpikes.filter(id => {
        const n = simState.neurons[id];
        return n && n.popId === pop.id;
      }).length;
      return (spkCount / Math.max(1, pop.size)) * 1000.0 / 1.0; // Hz approx
    });
    simState.rates.push(rates);
    if (simState.rates.length > 100) simState.rates.shift();

    // Synapse weights sample for histogram
    const wSample = [];
    simState.synapses.forEach(syn => {
      syn.weights.forEach(row => {
        row.forEach(conn => {
          if (conn) wSample.push(conn.w);
        });
      });
    });
    simState.weights = wSample;
  }

  // Canvas Rendering
  function renderCanvas() {
    const canvas = document.getElementById('studio-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--inset').trim() || '#121414';
    ctx.fillRect(0, 0, w, h);

    // Draw population clusters and neurons
    const popCount = config.populations.length;
    const cols = Math.min(2, popCount);
    const rows = Math.ceil(popCount / cols);
    const cellW = w / cols;
    const cellH = h / rows;

    config.populations.forEach((pop, pIdx) => {
      const cx = (pIdx % cols) * cellW + cellW / 2;
      const cy = Math.floor(pIdx / cols) * cellH + cellH / 2;

      // Draw cluster bounding circle / box
      ctx.strokeStyle = pop.excitatory ? 'var(--exc-mark, #B13A0A)' : 'var(--inh-mark, #1F64A8)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(cx, cy, Math.min(cellW, cellH) * 0.38, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = '11px sans-serif';
      ctx.fillText(`${pop.name} (N=${pop.size})`, cx - 35, cy - Math.min(cellW, cellH) * 0.42);

      // Position neurons in circular layout within cluster
      const radius = Math.min(cellW, cellH) * 0.28;
      for (let i = 0; i < pop.size; i++) {
        const angle = (i / pop.size) * Math.PI * 2;
        const nx = cx + Math.cos(angle) * radius;
        const ny = cy + Math.sin(angle) * radius;
        const neuron = simState.neurons[pop.startIdx + i];
        if (neuron) {
          neuron.x = nx;
          neuron.y = ny;

          ctx.beginPath();
          ctx.arc(nx, ny, neuron.id === simState.selectedNeuron ? 5 : 3.5, 0, Math.PI * 2);
          if (neuron.flashing > 0) {
            ctx.fillStyle = '#B87400'; // Amber active flash
          } else {
            ctx.fillStyle = pop.excitatory ? '#D55E00' : '#0072B2';
          }
          ctx.fill();
          if (neuron.id === simState.selectedNeuron) {
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }
      }
    });

    // Draw synapse bundle connections between populations
    simState.synapses.forEach(syn => {
      const srcPop = config.populations.find(p => p.id === syn.src);
      const dstPop = config.populations.find(p => p.id === syn.dst);
      if (srcPop && dstPop) {
        const srcIdx = config.populations.indexOf(srcPop);
        const dstIdx = config.populations.indexOf(dstPop);
        const x1 = ((srcIdx % cols) * cellW) + cellW / 2;
        const y1 = (Math.floor(srcIdx / cols) * cellH) + cellH / 2;
        const x2 = ((dstIdx % cols) * cellW) + cellW / 2;
        const y2 = (Math.floor(dstIdx / cols) * cellH) + cellH / 2;

        ctx.strokeStyle = syn.weight > 0 ? 'rgba(213, 94, 0, 0.25)' : 'rgba(0, 114, 178, 0.25)';
        ctx.lineWidth = Math.min(4, Math.max(1, Math.abs(syn.weight) * 2));
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    });

    // Draw traveling pulse packets
    ctx.fillStyle = '#B87400';
    simState.pulses.forEach(p => {
      p.progress += p.speed;
      const px = p.x1 + (p.x2 - p.x1) * p.progress;
      const py = p.y1 + (p.y2 - p.y1) * p.progress;
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fill();
    });
    simState.pulses = simState.pulses.filter(p => p.progress < 1.0);
  }

  function renderOscilloscopes() {
    // Raster Plot
    const rCanvas = document.getElementById('raster-canvas');
    if (rCanvas) {
      const ctx = rCanvas.getContext('2d');
      const w = rCanvas.width;
      const h = rCanvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#090A0A';
      ctx.fillRect(0, 0, w, h);

      const nTotal = simState.neurons.length;
      simState.raster.forEach((spikes, tIdx) => {
        const x = (tIdx / 80) * w;
        spikes.forEach(id => {
          const y = (id / Math.max(1, nTotal)) * h;
          ctx.fillStyle = '#B87400';
          ctx.fillRect(x, y, 2, 2);
        });
      });
    }

    // Voltage Trace
    const vCanvas = document.getElementById('voltage-canvas');
    if (vCanvas) {
      const ctx = vCanvas.getContext('2d');
      const w = vCanvas.width;
      const h = vCanvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#090A0A';
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = '#B87400';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      simState.voltages.forEach((v, idx) => {
        const x = (idx / 120) * w;
        // map v from [-80, 40] to [h, 0]
        const y = h - ((v - (-80)) / 120.0) * h;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // Firing Rate
    const rtCanvas = document.getElementById('rate-canvas');
    if (rtCanvas) {
      const ctx = rtCanvas.getContext('2d');
      const w = rtCanvas.width;
      const h = rtCanvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#090A0A';
      ctx.fillRect(0, 0, w, h);

      config.populations.forEach((pop, pIdx) => {
        ctx.strokeStyle = pop.excitatory ? '#D55E00' : '#0072B2';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        simState.rates.forEach((rates, idx) => {
          const x = (idx / 100) * w;
          const r = rates[pIdx] || 0;
          const y = h - Math.min(1.0, r / 100.0) * h;
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
    }

    // Weight Histogram
    const wCanvas = document.getElementById('weight-canvas');
    if (wCanvas) {
      const ctx = wCanvas.getContext('2d');
      const w = wCanvas.width;
      const h = wCanvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#090A0A';
      ctx.fillRect(0, 0, w, h);

      if (simState.weights.length > 0) {
        const bins = new Array(20).fill(0);
        simState.weights.forEach(wt => {
          const b = Math.min(19, Math.floor((wt / 5.0) * 20));
          bins[b]++;
        });
        const maxBin = Math.max(...bins, 1);
        const bw = w / 20;
        bins.forEach((cnt, idx) => {
          const bh = (cnt / maxBin) * h;
          ctx.fillStyle = '#B87400';
          ctx.fillRect(idx * bw + 1, h - bh, bw - 2, bh);
        });
      }
    }
  }

  // Python Code Export Generator
  function generatePythonCode() {
    let code = `\"\"\"Generated by neuroSush Studio Live.
pip install neurosush>=0.7.0
\"\"\"
from __future__ import annotations
import torch
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.models import ${config.model === 'lif' ? 'LIF' : config.model === 'elif' ? 'ELIF' : config.model === 'adex' ? 'AdaptiveELIF' : 'Izhikevich'}, Fire
from neurosush.neurons.inputs import PoissonInput
from neurosush.neurons.axon import Axon
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
`;

    if (config.stdp_enabled) {
      code += `from neurosush.synapses.plasticity import STDP\nfrom neurosush.synapses.traces import Traces, SpikeGather\nfrom neurosush.synapses.constraints import WeightClip\n`;
    }

    code += `\ndef run_studio_network():\n    net = Network(seed=${config.seed}, dt=1.0)\n\n`;

    config.populations.forEach(pop => {
      code += `    ${pop.name.toLowerCase()} = NeuronGroup(\n`;
      code += `        net,\n`;
      code += `        ${pop.size},\n`;
      code += `        [\n`;
      code += `            PoissonInput(${config.poisson_rate / 1000.0}),\n`;
      code += `            ${config.model === 'lif' ? `LIF(tau=${config.tau}, threshold=${config.v_th}, v_reset=${config.v_reset}, v_rest=${config.v_rest})` : config.model === 'elif' ? `ELIF(tau=${config.tau}, threshold=${config.v_th})` : config.model === 'adex' ? `AdaptiveELIF(tau=${config.tau})` : `Izhikevich()`},\n`;
      code += `            Fire(),\n`;
      code += `            Axon(),\n`;
      code += `        ],\n`;
      code += `        inhibitory=${!pop.excitatory},\n`;
      code += `        name="${pop.name}",\n`;
      code += `    )\n\n`;
    });

    config.connections.forEach((conn, cIdx) => {
      const srcPop = config.populations.find(p => p.id === conn.src);
      const dstPop = config.populations.find(p => p.id === conn.dst);
      if (!srcPop || !dstPop) return;

      code += `    SynapseGroup(\n`;
      code += `        net,\n`;
      code += `        ${srcPop.name.toLowerCase()},\n`;
      code += `        ${dstPop.name.toLowerCase()},\n`;
      code += `        [\n`;
      code += `            WeightInit(weights=torch.full((${srcPop.size}, ${dstPop.size}), ${conn.weight})),\n`;
      code += `            DenseInput(),\n`;
      if (config.stdp_enabled) {
        code += `            SpikeGather(),\n`;
        code += `            Traces(tau_pre=20.0),\n`;
        code += `            STDP(a_plus=${config.a_plus}, a_minus=${config.a_minus}),\n`;
        code += `            WeightClip(w_min=0.0, w_max=5.0),\n`;
      }
      code += `        ],\n`;
      code += `        name="conn_${cIdx}",\n`;
      code += `    )\n\n`;
    });

    code += `    net.initialize()\n    for step in range(100):\n        net.step()\n    print("Studio Live network executed successfully for 100 steps.")\n\nif __name__ == "__main__":\n    run_studio_network()\n`;
    return code;
  }

  // UI Wireup & Stepper
  let currentStep = 0;
  const totalSteps = 6;

  function updateStepView() {
    document.querySelectorAll('.step-tab').forEach((tab, idx) => {
      if (idx === currentStep) tab.classList.add('primary');
      else tab.classList.remove('primary');
    });
    document.querySelectorAll('.step-panel').forEach((panel, idx) => {
      if (idx === currentStep) panel.hidden = false;
      else panel.hidden = true;
    });
    const ind = document.getElementById('step-indicator');
    if (ind) ind.textContent = `Stage ${currentStep + 1} of ${totalSteps}`;
  }

  function renderPopulationsUI() {
    const list = document.getElementById('pop-list');
    if (!list) return;
    list.innerHTML = '';
    config.populations.forEach((pop, idx) => {
      const row = document.createElement('div');
      row.style.cssText = 'display: flex; gap: 8px; align-items: center; background: var(--inset); padding: 6px; border-radius: 4px;';
      row.innerHTML = `
        <input type="text" value="${pop.name}" data-pop="${idx}" class="pop-name-input" style="width: 80px; padding: 2px 4px; background: var(--surface); border: 1px solid var(--border); color: var(--text);">
        <input type="number" min="10" max="200" value="${pop.size}" data-pop="${idx}" class="pop-size-input" style="width: 60px; padding: 2px 4px; background: var(--surface); border: 1px solid var(--border); color: var(--text);">
        <label><input type="checkbox" ${pop.excitatory ? 'checked' : ''} data-pop="${idx}" class="pop-exc-check"> Exc</label>
        <button class="button" data-del-pop="${idx}" style="min-height: 26px; padding: 0 6px; margin-left: auto;">✕</button>
      `;
      list.appendChild(row);
    });
  }

  function loadPreset(presetName) {
    const p = PRESETS[presetName];
    if (!p) return;
    config = JSON.parse(JSON.stringify(p));
    initSimulation();
    renderPopulationsUI();
    // Update inputs
    document.getElementById('model-select').value = config.model;
    document.getElementById('param-tau').value = config.tau;
    document.getElementById('param-vth').value = config.v_th;
    document.getElementById('slider-poisson').value = config.poisson_rate;
    document.getElementById('val-poisson-rate').textContent = config.poisson_rate;
    document.getElementById('slider-current').value = config.current;
    document.getElementById('val-current').textContent = config.current;
    document.getElementById('stdp-enabled').checked = config.stdp_enabled;
    document.getElementById('param-aplus').value = config.a_plus;
    document.getElementById('param-aminus').value = config.a_minus;
  }

  document.addEventListener('DOMContentLoaded', () => {
    initSimulation();
    renderPopulationsUI();

    // Stepper buttons
    document.getElementById('btn-next-step')?.addEventListener('click', () => {
      if (currentStep < totalSteps - 1) {
        currentStep++;
        updateStepView();
      }
    });
    document.getElementById('btn-prev-step')?.addEventListener('click', () => {
      if (currentStep > 0) {
        currentStep--;
        updateStepView();
      }
    });
    document.querySelectorAll('.step-tab').forEach((tab, idx) => {
      tab.addEventListener('click', () => {
        currentStep = idx;
        updateStepView();
      });
    });

    // Play/Pause/Reset
    const playBtn = document.getElementById('btn-play');
    playBtn?.addEventListener('click', () => {
      simState.isPlaying = !simState.isPlaying;
      playBtn.textContent = simState.isPlaying ? 'Pause' : 'Play';
    });
    document.getElementById('btn-step')?.addEventListener('click', () => {
      simulateStep();
      renderCanvas();
      renderOscilloscopes();
    });
    document.getElementById('btn-reset')?.addEventListener('click', () => {
      initSimulation();
      renderCanvas();
      renderOscilloscopes();
    });

    // Pulse packet button
    document.getElementById('btn-pulse')?.addEventListener('click', () => {
      simState.neurons.forEach(n => {
        if (n.popId === 0) n.v = 0.0; // excite pop 0
      });
    });

    // Canvas click to select neuron or inject current
    document.getElementById('studio-canvas')?.addEventListener('click', e => {
      const rect = e.target.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      // Find closest neuron
      let closestId = 0;
      let minDist = Infinity;
      simState.neurons.forEach(n => {
        const dist = Math.hypot(n.x - x, n.y - y);
        if (dist < minDist) {
          minDist = dist;
          closestId = n.id;
        }
      });
      if (minDist < 30) {
        simState.selectedNeuron = closestId;
        const selLabel = document.getElementById('selected-trace-label');
        if (selLabel) selLabel.textContent = `Neuron #${closestId} Voltage (mV)`;
      } else {
        // inject current to all
        simState.neurons.forEach(n => { n.v += 10.0; });
      }
    });

    // Preset select
    document.getElementById('preset-select')?.addEventListener('change', e => {
      loadPreset(e.target.value);
    });

    // Model & parameters change
    document.getElementById('model-select')?.addEventListener('change', e => {
      config.model = e.target.value;
    });
    document.getElementById('param-tau')?.addEventListener('input', e => {
      config.tau = parseFloat(e.target.value) || 20.0;
    });
    document.getElementById('param-vth')?.addEventListener('input', e => {
      config.v_th = parseFloat(e.target.value) || -40.0;
    });
    document.getElementById('slider-poisson')?.addEventListener('input', e => {
      config.poisson_rate = parseFloat(e.target.value);
      document.getElementById('val-poisson-rate').textContent = config.poisson_rate;
    });
    document.getElementById('slider-current')?.addEventListener('input', e => {
      config.current = parseFloat(e.target.value);
      document.getElementById('val-current').textContent = config.current;
    });
    document.getElementById('stdp-enabled')?.addEventListener('change', e => {
      config.stdp_enabled = e.target.checked;
    });

    // Add Population button
    document.getElementById('btn-add-pop')?.addEventListener('click', () => {
      if (config.populations.length < 4) {
        const newId = config.populations.length;
        config.populations.push({
          id: newId,
          name: `Pop${newId + 1}`,
          size: 25,
          excitatory: newId % 2 === 0,
        });
        initSimulation();
        renderPopulationsUI();
      }
    });

    // Export Python code button
    document.getElementById('btn-export')?.addEventListener('click', () => {
      const code = generatePythonCode();
      const codePreview = document.getElementById('code-preview');
      if (codePreview) codePreview.textContent = code;
      currentStep = 5;
      updateStepView();
    });

    document.getElementById('btn-copy-code')?.addEventListener('click', () => {
      const code = generatePythonCode();
      navigator.clipboard.writeText(code).then(() => {
        const status = document.getElementById('copy-status');
        if (status) {
          status.textContent = 'Copied to clipboard!';
          setTimeout(() => { status.textContent = ''; }, 2000);
        }
      }).catch(() => {
        // Fallback
        const ta = document.createElement('textarea');
        ta.value = code;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        const status = document.getElementById('copy-status');
        if (status) status.textContent = 'Copied (fallback)!';
      });
    });

    // Main Animation Loop (respects reduced motion)
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function animationLoop() {
      if (simState.isPlaying && !prefersReducedMotion) {
        simulateStep();
        renderCanvas();
        renderOscilloscopes();
      }
      requestAnimationFrame(animationLoop);
    }
    requestAnimationFrame(animationLoop);
  });
})();
