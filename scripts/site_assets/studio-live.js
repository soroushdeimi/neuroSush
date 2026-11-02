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
    poisson_rate: 25.0,
    current: 3.0,
    stdp_enabled: false,
    a_plus: 0.02,
    a_minus: 0.021,
    populations: [
      { id: 0, name: 'Exc', size: 40, excitatory: true },
      { id: 1, name: 'Inh', size: 12, excitatory: false },
    ],
    connections: [
      { src: 0, dst: 0, p: 0.25, weight: 0.6, delay: 1 },
      { src: 0, dst: 1, p: 0.35, weight: 0.8, delay: 1 },
      { src: 1, dst: 0, p: 0.35, weight: -0.7, delay: 1 },
    ],
  };

  // Simulation Runtime State
  let simState = {
    stepCount: 0,
    isPlaying: true,
    speed: 1,
    neurons: [],
    synapses: [],
    raster: [],
    voltages: [],
    rates: [],
    weights: [],
    selectedNeuron: 0,
    pulses: [],
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
      poisson_rate: 30.0,
      current: 3.5,
      stdp_enabled: false,
      a_plus: 0.02,
      a_minus: 0.021,
      populations: [
        { id: 0, name: 'Exc', size: 48, excitatory: true },
        { id: 1, name: 'Inh', size: 16, excitatory: false },
      ],
      connections: [
        { src: 0, dst: 0, p: 0.25, weight: 0.5, delay: 1 },
        { src: 0, dst: 1, p: 0.35, weight: 0.7, delay: 1 },
        { src: 1, dst: 0, p: 0.35, weight: -0.6, delay: 1 },
      ],
    },
    wta: {
      model: 'lif',
      tau: 20.0,
      v_th: -40.0,
      v_rest: -70.0,
      v_reset: -75.0,
      resistance: 10.0,
      poisson_rate: 28.0,
      current: 3.0,
      stdp_enabled: true,
      a_plus: 0.02,
      a_minus: 0.021,
      populations: [
        { id: 0, name: 'PopA', size: 30, excitatory: true },
        { id: 1, name: 'PopB', size: 30, excitatory: true },
        { id: 2, name: 'InhWTA', size: 16, excitatory: false },
      ],
      connections: [
        { src: 0, dst: 2, p: 0.4, weight: 0.6, delay: 1 },
        { src: 1, dst: 2, p: 0.4, weight: 0.6, delay: 1 },
        { src: 2, dst: 0, p: 0.5, weight: -0.9, delay: 1 },
        { src: 2, dst: 1, p: 0.5, weight: -0.9, delay: 1 },
      ],
    },
    chain: {
      model: 'elif',
      tau: 15.0,
      v_th: -45.0,
      v_rest: -70.0,
      v_reset: -75.0,
      resistance: 10.0,
      poisson_rate: 25.0,
      current: 3.0,
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
          v: config.v_rest + (rng() - 0.5) * 4.0,
          u: 0.0,
          refractory: 0.0,
          flashing: 0.0, // 0 to 1 for amber glow trail (~300ms)
          x_trace: 0.0,
          v_trace: 0.0,
          x: 0,
          y: 0,
        });
      }
      offset += pop.size;
    }

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
                w: conn.weight * (0.85 + 0.3 * rng()),
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
      const expTerm = deltaT * Math.exp(Math.min(50, (neuron.v - config.v_th) / deltaT));
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
      const dv = dt * ((config.v_rest - neuron.v) + deltaT * Math.exp(Math.min(50, (neuron.v - config.v_th) / deltaT)) - neuron.u + config.resistance * I_ext) / config.tau;
      const du = dt * (0.5 * (neuron.v - config.v_rest) - neuron.u) / tauW;
      neuron.v += dv;
      neuron.u += du;
      if (neuron.v >= config.v_th + 20.0) {
        spiked = true;
        neuron.v = config.v_reset;
        neuron.u += 5.0;
        neuron.refractory = 2.0;
      }
    } else if (config.model === 'izhikevich') {
      let vv = neuron.v,
        uu = neuron.u;
      for (let s = 0; s < 2; s++) {
        const dv = (0.04 * vv * vv + 5 * vv + 140 - uu + I_ext) / 2.0;
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

    simState.neurons.forEach(neuron => {
      let I_ext = config.current;
      if (rng() * 1000 < config.poisson_rate * 1.0) {
        I_ext += 6.0; // robust Poisson drive
      }

      simState.synapses.forEach(syn => {
        const srcPop = config.populations.find(p => p.id === syn.src);
        const dstPop = config.populations.find(p => p.id === syn.dst);
        if (neuron.popId === syn.dst && srcPop && dstPop) {
          const localIdx = neuron.id - dstPop.startIdx;
          if (localIdx >= 0 && localIdx < dstPop.size) {
            for (let i = 0; i < srcPop.size; i++) {
              const conn = syn.weights[i][localIdx];
              if (conn) {
                const arriving = conn.queue.shift();
                conn.queue.push(0);
                if (arriving) {
                  I_ext += conn.w * 8.0;
                }
              }
            }
          }
        }
      });

      const spiked = stepNeuron(neuron, I_ext);
      if (spiked) {
        currentSpikes.push(neuron.id);
        neuron.flashing = 1.0; // full amber glow
        if (simState.selectedNeuron === neuron.id) {
          neuron.v = 30.0;
        }
      } else {
        if (neuron.flashing > 0) neuron.flashing = Math.max(0, neuron.flashing - 0.05); // ~300ms fade trail at 60fps
      }
    });

    simState.synapses.forEach(syn => {
      const srcPop = config.populations.find(p => p.id === syn.src);
      const dstPop = config.populations.find(p => p.id === syn.dst);
      if (!srcPop || !dstPop) return;

      for (let i = 0; i < srcPop.size; i++) {
        const srcNeuron = simState.neurons[srcPop.startIdx + i];
        const srcSpiked = currentSpikes.includes(srcNeuron.id);

        if (srcSpiked) {
          srcNeuron.x_trace = 1.0;
        } else {
          srcNeuron.x_trace *= Math.exp(-1.0 / 20.0);
        }

        for (let j = 0; j < dstPop.size; j++) {
          const conn = syn.weights[i][j];
          if (!conn) continue;

          const dstNeuron = simState.neurons[dstPop.startIdx + j];
          const dstSpiked = currentSpikes.includes(dstNeuron.id);

          if (srcSpiked) {
            const delayIdx = Math.min(conn.queue.length - 1, conn.delay - 1);
            conn.queue[delayIdx] = 1;

            if (simState.pulses.length < 40 && rng() < 0.25) {
              simState.pulses.push({
                x1: srcNeuron.x,
                y1: srcNeuron.y,
                x2: dstNeuron.x,
                y2: dstNeuron.y,
                progress: 0.0,
                speed: 0.06 + rng() * 0.06,
                isExc: syn.weight >= 0,
              });
            }
          }

          if (config.stdp_enabled && srcSpiked) {
            conn.w -= config.a_minus * dstNeuron.v_trace;
            conn.w = Math.max(0.0, Math.min(5.0, conn.w));
          }
          if (config.stdp_enabled && dstSpiked) {
            conn.w += config.a_plus * srcNeuron.x_trace;
            conn.w = Math.max(0.0, Math.min(5.0, conn.w));
          }
        }
      }
    });

    simState.neurons.forEach(neuron => {
      const spiked = currentSpikes.includes(neuron.id);
      if (spiked) {
        neuron.v_trace = 1.0;
      } else {
        neuron.v_trace *= Math.exp(-1.0 / 20.0);
      }
    });

    simState.raster.push(currentSpikes);
    if (simState.raster.length > 100) simState.raster.shift();

    const sel = simState.neurons[simState.selectedNeuron];
    if (sel) {
      simState.voltages.push(sel.v);
      if (simState.voltages.length > 120) simState.voltages.shift();
    }

    const rates = config.populations.map(pop => {
      const spkCount = currentSpikes.filter(id => {
        const n = simState.neurons[id];
        return n && n.popId === pop.id;
      }).length;
      return (spkCount / Math.max(1, pop.size)) * 1000.0;
    });
    simState.rates.push(rates);
    if (simState.rates.length > 100) simState.rates.shift();

    const wSample = [];
    simState.synapses.forEach(syn => {
      syn.weights.forEach(row => {
        row.forEach(conn => {
          if (conn) wSample.push(conn.w);
        });
      });
    });
    simState.weights = wSample;

    // Update code preview continuously
    updateCodePreview();
  }

  function renderCanvas() {
    const canvas = document.getElementById('studio-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--inset').trim() || '#121414';
    ctx.fillRect(0, 0, w, h);

    const popCount = config.populations.length;
    const cols = Math.min(2, popCount);
    const rows = Math.ceil(popCount / cols);
    const cellW = w / cols;
    const cellH = h / rows;

    // 1. Draw thin curved synapse bundles with opacity/width following weight
    simState.synapses.forEach(syn => {
      const srcPop = config.populations.find(p => p.id === syn.src);
      const dstPop = config.populations.find(p => p.id === syn.dst);
      if (!srcPop || !dstPop) return;
      const srcIdx = config.populations.indexOf(srcPop);
      const dstIdx = config.populations.indexOf(dstPop);
      const cx1 = ((srcIdx % cols) * cellW) + cellW / 2;
      const cy1 = (Math.floor(srcIdx / cols) * cellH) + cellH / 2;
      const cx2 = ((dstIdx % cols) * cellW) + cellW / 2;
      const cy2 = (Math.floor(dstIdx / cols) * cellH) + cellH / 2;

      // Sample a subset of weights for clean rendering
      syn.weights.forEach((row, i) => {
        if (i % 3 !== 0) return;
        row.forEach((conn, j) => {
          if (!conn || j % 3 !== 0) return;
          const srcN = simState.neurons[srcPop.startIdx + i];
          const dstN = simState.neurons[dstPop.startIdx + j];
          if (!srcN || !dstN) return;

          ctx.strokeStyle = conn.w >= 0 ? 'rgba(213, 94, 0, 0.35)' : 'rgba(31, 100, 168, 0.45)';
          ctx.lineWidth = Math.min(2.0, Math.max(0.5, conn.w * 0.8));
          ctx.beginPath();
          ctx.moveTo(srcN.x, srcN.y);
          // quadratic curve bowing towards center
          const mx = (srcN.x + dstN.x) / 2 + (cx2 - cx1) * 0.1;
          const my = (srcN.y + dstN.y) / 2 + (cy2 - cy1) * 0.1;
          ctx.quadraticCurveTo(mx, my, dstN.x, dstN.y);
          ctx.stroke();
        });
      });
    });

    // 2. Draw population clusters and neurons with amber firing glow & trail
    config.populations.forEach((pop, pIdx) => {
      const cx = (pIdx % cols) * cellW + cellW / 2;
      const cy = Math.floor(pIdx / cols) * cellH + cellH / 2;

      ctx.strokeStyle = pop.excitatory ? 'rgba(213, 94, 0, 0.6)' : 'rgba(31, 100, 168, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.arc(cx, cy, Math.min(cellW, cellH) * 0.40, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = '12px sans-serif';
      ctx.fillText(`${pop.name} (N=${pop.size})`, cx - 40, cy - Math.min(cellW, cellH) * 0.44);

      const radius = Math.min(cellW, cellH) * 0.30;
      for (let i = 0; i < pop.size; i++) {
        const angle = (i / pop.size) * Math.PI * 2;
        const nx = cx + Math.cos(angle) * radius;
        const ny = cy + Math.sin(angle) * radius;
        const neuron = simState.neurons[pop.startIdx + i];
        if (neuron) {
          neuron.x = nx;
          neuron.y = ny;

          ctx.beginPath();
          ctx.arc(nx, ny, neuron.id === simState.selectedNeuron ? 6 : 4, 0, Math.PI * 2);

          if (neuron.flashing > 0.05) {
            // Amber glow when firing / trailing
            ctx.fillStyle = `rgba(184, 116, 0, ${0.4 + 0.6 * neuron.flashing})`;
            ctx.fill();
            ctx.strokeStyle = '#F2B33D';
            ctx.lineWidth = 2.0;
            ctx.stroke();
          } else {
            ctx.fillStyle = pop.excitatory ? '#D55E00' : '#0072B2';
            ctx.fill();
          }

          if (neuron.id === simState.selectedNeuron) {
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2.0;
            ctx.stroke();
          }
        }
      }
    });

    // 3. Draw traveling pulses along curves
    simState.pulses.forEach(p => {
      p.progress += p.speed;
      const mx = (p.x1 + p.x2) / 2;
      const my = (p.y1 + p.y2) / 2;
      // Quadratic bezier point calculation
      const t = p.progress;
      const px = (1 - t) * (1 - t) * p.x1 + 2 * (1 - t) * t * mx + t * t * p.x2;
      const py = (1 - t) * (1 - t) * p.y1 + 2 * (1 - t) * t * my + t * t * p.y2;

      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = p.isExc ? '#F2B33D' : '#5BA8E0';
      ctx.fill();
    });
    simState.pulses = simState.pulses.filter(p => p.progress < 1.0);
  }

  function renderOscilloscopes() {
    // 1. Raster Plot
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
        const x = (tIdx / 100) * w;
        spikes.forEach(id => {
          const y = (id / Math.max(1, nTotal)) * h;
          const neuron = simState.neurons[id];
          ctx.fillStyle = neuron && neuron.popId === 0 ? '#D55E00' : '#0072B2';
          ctx.fillRect(x, y, 2.5, Math.max(2, h / nTotal));
        });
      });
    }

    // 2. Voltage Trace
    const vCanvas = document.getElementById('voltage-canvas');
    if (vCanvas) {
      const ctx = vCanvas.getContext('2d');
      const w = vCanvas.width;
      const h = vCanvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#090A0A';
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = '#B87400';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      simState.voltages.forEach((v, idx) => {
        const x = (idx / 120) * w;
        const y = h - ((v - (-80)) / 120.0) * h;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // 3. Firing Rate
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
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        simState.rates.forEach((rates, idx) => {
          const x = (idx / 100) * w;
          const r = rates[pIdx] || 0;
          const y = h - Math.min(1.0, r / 120.0) * h;
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
    }

    // 4. Weight Heatmap & Histogram (when STDP is on, show matrix heatmap + histogram)
    const wCanvas = document.getElementById('weight-canvas');
    const wLabel = document.getElementById('weight-view-label');
    if (wCanvas) {
      const ctx = wCanvas.getContext('2d');
      const w = wCanvas.width;
      const h = wCanvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#090A0A';
      ctx.fillRect(0, 0, w, h);

      if (config.stdp_enabled && simState.synapses.length > 0) {
        if (wLabel) wLabel.textContent = 'STDP Weight Matrix Heatmap & Histogram';
        // Render small weight matrix heatmap on left half, histogram on right half
        const syn = simState.synapses[0];
        const rows = syn.weights.length;
        const cols = syn.weights[0].length;
        const cellW = (w / 2) / Math.max(1, cols);
        const cellH = h / Math.max(1, rows);

        syn.weights.forEach((row, i) => {
          row.forEach((conn, j) => {
            if (!conn) return;
            const intensity = Math.min(1.0, conn.w / 5.0);
            ctx.fillStyle = `rgba(184, 116, 0, ${intensity})`;
            ctx.fillRect(j * cellW, i * cellH, cellW - 0.5, cellH - 0.5);
          });
        });

        // Histogram on right half
        const histX = w / 2 + 10;
        const histW = w / 2 - 15;
        const bins = new Array(15).fill(0);
        simState.weights.forEach(wt => {
          const b = Math.min(14, Math.floor((wt / 5.0) * 15));
          bins[b]++;
        });
        const maxBin = Math.max(...bins, 1);
        const bw = histW / 15;
        bins.forEach((cnt, idx) => {
          const bh = (cnt / maxBin) * h;
          ctx.fillStyle = '#F2B33D';
          ctx.fillRect(histX + idx * bw, h - bh, bw - 1, bh);
        });
      } else {
        if (wLabel) wLabel.textContent = 'Synaptic Weight Histogram';
        if (simState.weights.length > 0) {
          const bins = new Array(25).fill(0);
          simState.weights.forEach(wt => {
            const b = Math.min(24, Math.floor((wt / 5.0) * 25));
            bins[b]++;
          });
          const maxBin = Math.max(...bins, 1);
          const bw = w / 25;
          bins.forEach((cnt, idx) => {
            const bh = (cnt / maxBin) * h;
            ctx.fillStyle = '#B87400';
            ctx.fillRect(idx * bw + 1, h - bh, bw - 1, bh);
          });
        }
      }
    }
  }

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

  function updateCodePreview() {
    const code = generatePythonCode();
    const codePreview = document.getElementById('code-preview');
    if (codePreview) codePreview.textContent = code;
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
    updateCodePreview();
  }

  document.addEventListener('DOMContentLoaded', () => {
    // Respect prefers-reduced-motion: if true, default isPaused = true (static frame)
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
      simState.isPlaying = false;
      const playBtn = document.getElementById('btn-play');
      if (playBtn) playBtn.textContent = 'Run';
    }

    loadPreset('balanced');
    initSimulation();
    renderPopulationsUI();
    updateCodePreview();

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
      playBtn.textContent = simState.isPlaying ? 'Pause' : 'Run';
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

    document.getElementById('btn-pulse')?.addEventListener('click', () => {
      simState.neurons.forEach(n => {
        if (n.popId === 0) n.v = 0.0;
      });
    });

    document.getElementById('studio-canvas')?.addEventListener('click', e => {
      const rect = e.target.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      let closestId = 0;
      let minDist = Infinity;
      simState.neurons.forEach(n => {
        const dist = Math.hypot(n.x - x, n.y - y);
        if (dist < minDist) {
          minDist = dist;
          closestId = n.id;
        }
      });
      if (minDist < 35) {
        simState.selectedNeuron = closestId;
        const selLabel = document.getElementById('selected-trace-label');
        if (selLabel) selLabel.textContent = `Neuron #${closestId} Voltage (mV)`;
      } else {
        simState.neurons.forEach(n => { n.v += 12.0; });
      }
    });

    document.getElementById('preset-select')?.addEventListener('change', e => {
      loadPreset(e.target.value);
    });

    document.getElementById('model-select')?.addEventListener('change', e => {
      config.model = e.target.value;
      updateCodePreview();
    });
    document.getElementById('param-tau')?.addEventListener('input', e => {
      config.tau = parseFloat(e.target.value) || 20.0;
      updateCodePreview();
    });
    document.getElementById('param-vth')?.addEventListener('input', e => {
      config.v_th = parseFloat(e.target.value) || -40.0;
      updateCodePreview();
    });
    document.getElementById('slider-poisson')?.addEventListener('input', e => {
      config.poisson_rate = parseFloat(e.target.value);
      document.getElementById('val-poisson-rate').textContent = config.poisson_rate;
      updateCodePreview();
    });
    document.getElementById('slider-current')?.addEventListener('input', e => {
      config.current = parseFloat(e.target.value);
      document.getElementById('val-current').textContent = config.current;
      updateCodePreview();
    });
    document.getElementById('stdp-enabled')?.addEventListener('change', e => {
      config.stdp_enabled = e.target.checked;
      updateCodePreview();
    });

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
        updateCodePreview();
      }
    });

    document.getElementById('btn-export')?.addEventListener('click', () => {
      currentStep = 5;
      updateStepView();
      const codePreview = document.getElementById('code-preview');
      if (codePreview) codePreview.scrollIntoView({ behavior: 'smooth' });
    });

    const copyHandler = () => {
      const code = generatePythonCode();
      navigator.clipboard.writeText(code).then(() => {
        const status = document.getElementById('copy-status');
        if (status) {
          status.textContent = 'Copied to clipboard!';
          setTimeout(() => { status.textContent = ''; }, 2000);
        }
      }).catch(() => {
        const ta = document.createElement('textarea');
        ta.value = code;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        const status = document.getElementById('copy-status');
        if (status) status.textContent = 'Copied (fallback)!';
      });
    };

    document.getElementById('btn-copy-code')?.addEventListener('click', copyHandler);
    document.getElementById('btn-copy-code-main')?.addEventListener('click', copyHandler);

    // Main Animation Loop
    function animationLoop() {
      if (simState.isPlaying && !prefersReducedMotion) {
        simulateStep();
      }
      renderCanvas();
      renderOscilloscopes();
      requestAnimationFrame(animationLoop);
    }
    requestAnimationFrame(animationLoop);
  });
})();
