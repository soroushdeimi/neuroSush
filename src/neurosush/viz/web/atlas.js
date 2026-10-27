/**
 * neuroSush Atlas - Plate Shell & Animation Engine
 * ES2020 vanilla library: clock, timeline, trace decoding, i18n, tokens & canvas tools.
 */
(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = factory();
  } else if (typeof globalThis !== 'undefined') {
    globalThis.Atlas = factory();
  } else {
    root.Atlas = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // Base64 decoding helper (cross-platform: browser + node)
  function decodeBase64(b64) {
    if (!b64 || b64.length === 0) return new Uint8Array(0);
    if (typeof atob === 'function') {
      const bin = atob(b64);
      const len = bin.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = bin.charCodeAt(i);
      }
      return bytes;
    }
    if (typeof Buffer !== 'undefined') {
      return new Uint8Array(Buffer.from(b64, 'base64'));
    }
    throw new Error('No base64 decoder available in current environment');
  }

  /**
   * Unpack a serialized packed array into a typed array.
   * @param {{ dtype: string, shape?: number[], data: string }} packed
   * @returns {Float32Array|Uint8Array|Int32Array}
   */
  function unpack(packed) {
    if (!packed || typeof packed !== 'object') {
      throw new Error(`packed must be an object, got ${typeof packed}`);
    }
    const { dtype, shape, data } = packed;
    if (!dtype) {
      throw new Error('dtype is required in packed array');
    }
    const rawBytes = data ? decodeBase64(data) : new Uint8Array(0);
    const buffer = rawBytes.buffer.slice(
      rawBytes.byteOffset,
      rawBytes.byteOffset + rawBytes.byteLength
    );

    let array;
    switch (dtype) {
      case 'float32':
        array = new Float32Array(buffer);
        break;
      case 'uint8':
        array = new Uint8Array(buffer);
        break;
      case 'int32':
        array = new Int32Array(buffer);
        break;
      default:
        throw new Error(`unsupported dtype: ${dtype}`);
    }
    return array;
  }

  /**
   * Trace class representing a multi-series recorded simulation trace.
   */
  class Trace {
    constructor(traceDict = {}) {
      this.meta = traceDict.meta || {};
      this.series = {};
      let inferredSteps;

      for (const [name, packed] of Object.entries(traceDict.series || {})) {
        const data = unpack(packed);
        const shape = packed.shape || [data.length];
        this.series[name] = { data, shape };
        if (inferredSteps === undefined && shape.length > 0) {
          inferredSteps = shape[0];
        }
      }
      // Python's step-index list is canonical; normalize older count-only inputs here.
      const steps = traceDict.steps;
      this.indices = Array.isArray(steps)
        ? [...steps]
        : Array.from({ length: steps ?? inferredSteps ?? 0 }, (_, i) => i);
      this.steps = this.indices.length;

      // Event steps (e.g. spikes)
      if (Array.isArray(traceDict.events)) {
        this.events = [...traceDict.events].sort((a, b) => a - b);
      } else if (this.series.spikes) {
        const spk = this.series.spikes.data;
        const shape = this.series.spikes.shape;
        const stride = shape.length > 1 ? shape.slice(1).reduce((a, b) => a * b, 1) : 1;
        const evs = [];
        for (let s = 0; s < this.steps; s++) {
          let hasSpike = false;
          for (let j = 0; j < stride; j++) {
            if (spk[s * stride + j] > 0) {
              hasSpike = true;
              break;
            }
          }
          if (hasSpike) evs.push(s);
        }
        this.events = evs;
      } else {
        this.events = [];
      }
    }

    /**
     * Return a TypedArray slice view for series at step i.
     * @param {string} name
     * @param {number} i
     * @returns {Float32Array|Uint8Array|Int32Array}
     */
    at(name, i) {
      const s = this.series[name];
      if (!s) {
        throw new Error(`unknown series: ${name}`);
      }
      if (i < 0 || i >= this.steps) {
        throw new Error(`step ${i} out of bounds for trace with ${this.steps} steps`);
      }
      const shape = s.shape;
      const stride = shape.length > 1 ? shape.slice(1).reduce((a, b) => a * b, 1) : 1;
      const start = i * stride;
      const end = start + stride;
      return s.data.subarray(start, end);
    }
  }

  /**
   * Deterministic Clock for simulation playhead.
   */
  class Clock {
    constructor(stepsOrOpts = 1, maybeOpts = {}) {
      const opts = typeof stepsOrOpts === 'object' ? stepsOrOpts : { steps: stepsOrOpts, ...maybeOpts };
      this.steps = Math.max(1, opts.steps ?? 1);
      this.t = Math.max(0, Math.min(this.steps - 1, opts.initialStep ?? 0));
      this.stepsPerSecond = opts.stepsPerSecond ?? 30;
      this.speed = opts.speed ?? 1;
      this.loop = opts.loop ?? true;
      this.events = Array.isArray(opts.events) ? [...opts.events].sort((a, b) => a - b) : [];
      this.isPlaying = false;
      this._listeners = new Set();
      this._fractional = 0;
      this._rafId = null;
      this._lastTimestamp = null;
    }

    play() {
      if (this.isPlaying) return;
      this.isPlaying = true;
      this._lastTimestamp = null;
      this._startRaf();
      this._notify();
    }

    pause() {
      if (!this.isPlaying) return;
      this.isPlaying = false;
      this._stopRaf();
      this._notify();
    }

    toggle() {
      if (this.isPlaying) {
        this.pause();
      } else {
        this.play();
      }
    }

    step(n = 1) {
      this.seek(this.t + n);
    }

    seek(target) {
      const clamped = Math.max(0, Math.min(this.steps - 1, Math.round(target)));
      if (this.t !== clamped) {
        this.t = clamped;
        this._fractional = 0;
        this._notify();
      }
    }

    jumpEvent(direction) {
      if (!this.events || this.events.length === 0) return;
      if (direction > 0) {
        for (const ev of this.events) {
          if (ev > this.t) {
            this.seek(ev);
            return;
          }
        }
        if (this.loop) {
          this.seek(this.events[0]);
        } else {
          this.seek(this.events[this.events.length - 1]);
        }
      } else if (direction < 0) {
        for (let i = this.events.length - 1; i >= 0; i--) {
          const ev = this.events[i];
          if (ev < this.t) {
            this.seek(ev);
            return;
          }
        }
        if (this.loop) {
          this.seek(this.events[this.events.length - 1]);
        } else {
          this.seek(this.events[0]);
        }
      }
    }

    setSpeed(speed) {
      this.speed = Math.max(0.01, Number(speed) || 1);
      this._notify();
    }

    onChange(fn) {
      this._listeners.add(fn);
      return () => this._listeners.delete(fn);
    }

    _notify() {
      for (const fn of this._listeners) {
        try {
          fn(this);
        } catch (err) {
          if (typeof console !== 'undefined' && console.error) {
            console.error(err);
          }
        }
      }
    }

    /**
     * Advance clock by dtMs milliseconds deterministically.
     * Steps advanced = (dtMs / 1000) * stepsPerSecond * speed.
     * @param {number} dtMs
     */
    tick(dtMs) {
      if (!this.isPlaying) return;
      const dtSec = dtMs / 1000;
      const delta = dtSec * this.stepsPerSecond * this.speed;
      this._fractional += delta;
      const stepsToAdvance = Math.floor(this._fractional);
      if (stepsToAdvance !== 0) {
        this._fractional -= stepsToAdvance;
        let nextT = this.t + stepsToAdvance;
        if (this.loop) {
          nextT = nextT % this.steps;
          if (nextT < 0) nextT += this.steps;
          this.t = nextT;
          this._notify();
        } else {
          if (nextT >= this.steps - 1) {
            this.t = this.steps - 1;
            this.pause();
          } else if (nextT < 0) {
            this.t = 0;
            this._notify();
          } else {
            this.t = nextT;
            this._notify();
          }
        }
      }
    }

    _startRaf() {
      if (typeof requestAnimationFrame === 'function' && !this._rafId) {
        const loop = (timestamp) => {
          if (!this.isPlaying) return;
          if (this._lastTimestamp !== null) {
            const dt = timestamp - this._lastTimestamp;
            this.tick(dt);
          }
          this._lastTimestamp = timestamp;
          this._rafId = requestAnimationFrame(loop);
        };
        this._rafId = requestAnimationFrame(loop);
      }
    }

    _stopRaf() {
      if (this._rafId && typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(this._rafId);
      }
      this._rafId = null;
      this._lastTimestamp = null;
    }
  }

  // Strings & localization module
  const strings = {};

  function setStrings(dict = {}) {
    if (dict.en) {
      Object.assign(strings, dict.en);
    } else {
      Object.assign(strings, dict);
    }
  }

  function t(key, vars = {}) {
    let str = strings[key] ?? key;
    if (typeof str === 'string' && vars && typeof vars === 'object') {
      for (const [k, v] of Object.entries(vars)) {
        str = str.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v));
      }
    }
    return str;
  }

  // Theme manager
  const Theme = {
    get() {
      try {
        if (typeof localStorage !== 'undefined') {
          return localStorage.getItem('neurosush-theme') || 'auto';
        }
      } catch {}
      return 'auto';
    },
    set(theme) {
      try {
        if (typeof localStorage !== 'undefined') {
          localStorage.setItem('neurosush-theme', theme);
        }
      } catch {}
      if (typeof document !== 'undefined') {
        const root = document.documentElement;
        if (theme === 'dark') {
          root.setAttribute('data-theme', 'dark');
        } else if (theme === 'light') {
          root.setAttribute('data-theme', 'light');
        } else {
          root.removeAttribute('data-theme');
        }
      }
    },
    cycle() {
      const cur = this.get();
      const next = cur === 'auto' ? 'light' : cur === 'light' ? 'dark' : 'auto';
      this.set(next);
      return next;
    },
    init() {
      this.set(this.get());
    }
  };

  /**
   * Calculate nice round numbers for axis ticks.
   * @param {number} min
   * @param {number} max
   * @param {number} n
   * @returns {number[]}
   */
  function niceTicks(min, max, n = 5) {
    if (min === max) return [min];
    let low = min;
    let high = max;
    if (low > high) {
      low = max;
      high = min;
    }
    const range = high - low;
    const roughStep = range / Math.max(1, n - 1);
    const p = Math.floor(Math.log10(roughStep));
    const f = roughStep / Math.pow(10, p);
    let niceF;
    if (f < 1.5) niceF = 1;
    else if (f < 3.5) niceF = 2;
    else if (f < 7.5) niceF = 5;
    else niceF = 10;
    const step = niceF * Math.pow(10, p);

    const start = Math.ceil(low / step - 1e-9) * step;
    const ticks = [];
    const decimals = Math.max(0, -p + 1);
    for (let val = start; val <= high + 1e-9; val += step) {
      ticks.push(Number(val.toFixed(decimals)));
    }
    return ticks;
  }

  /**
   * Format number for display with tabular numerals.
   * @param {number} x
   * @param {number} digits
   * @returns {string}
   */
  function formatNumber(x, digits = 2) {
    if (x === null || x === undefined || Number.isNaN(x)) return '-';
    const num = Number(x);
    if (!Number.isFinite(num)) return String(num);
    let res = num.toFixed(digits);
    const zeroStr = (0).toFixed(digits);
    if (res === `-${zeroStr}` || res === '-0') {
      res = zeroStr;
    }
    return res;
  }

  /**
   * Draw polyline series on canvas context.
   * @param {CanvasRenderingContext2D} ctx
   * @param {ArrayLike<number>} xs
   * @param {ArrayLike<number>} ys
   * @param {object} style
   */
  function lineSeries(ctx, xs, ys, style = {}) {
    if (!ctx || !xs || !ys || xs.length === 0 || ys.length === 0) return;
    const len = Math.min(xs.length, ys.length);
    ctx.save();
    ctx.strokeStyle = style.strokeStyle || '#1b1a17';
    ctx.lineWidth = style.lineWidth ?? 1.5;
    if (Array.isArray(style.lineDash)) {
      ctx.setLineDash(style.lineDash);
    }
    ctx.beginPath();
    let first = true;
    for (let i = 0; i < len; i++) {
      const x = xs[i];
      const y = ys[i];
      if (Number.isFinite(x) && Number.isFinite(y)) {
        if (first) {
          ctx.moveTo(x, y);
          first = false;
        } else {
          ctx.lineTo(x, y);
        }
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  /**
   * Adjust canvas internal resolution to match device pixel ratio.
   * @param {HTMLCanvasElement} canvas
   * @returns {{ width: number, height: number, dpr: number }}
   */
  function fitCanvas(canvas) {
    if (!canvas) return { width: 0, height: 0, dpr: 1 };
    const dpr = typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1;
    const rect = canvas.getBoundingClientRect ? canvas.getBoundingClientRect() : { width: canvas.width, height: canvas.height };
    const w = Math.round(rect.width || canvas.width || 300);
    const h = Math.round(rect.height || canvas.height || 150);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext ? canvas.getContext('2d') : null;
    if (ctx && ctx.resetTransform) {
      ctx.resetTransform();
      ctx.scale(dpr, dpr);
    }
    return { width: w, height: h, dpr };
  }

  /**
   * Read dynamic theme colors from computed styles.
   * @returns {object}
   */
  function colors() {
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return {
        volt: '#b45309',
        exc: '#0f766e',
        inh: '#be123c',
        mod: '#7e22ce',
        trace: '#1d4ed8',
        theory: '#1b1a17',
        paper: '#f6f2e9',
        paperCard: '#eeeae0',
        ink: '#1b1a17',
        inkMuted: '#57534e',
        border: '#d6cfbf'
      };
    }
    const style = window.getComputedStyle(document.documentElement);
    const get = (prop, fallback) => style.getPropertyValue(prop).trim() || fallback;
    return {
      volt: get('--volt', '#b45309'),
      exc: get('--exc', '#0f766e'),
      inh: get('--inh', '#be123c'),
      mod: get('--mod', '#7e22ce'),
      trace: get('--trace', '#1d4ed8'),
      theory: get('--theory', '#1b1a17'),
      paper: get('--paper', '#f6f2e9'),
      paperCard: get('--paper-card', '#eeeae0'),
      ink: get('--ink', '#1b1a17'),
      inkMuted: get('--ink-muted', '#57534e'),
      border: get('--border', '#d6cfbf')
    };
  }

  /**
   * Timeline scrubber component.
   */
  class Timeline {
    constructor(containerEl, clock, trace) {
      this.container = containerEl;
      this.clock = clock;
      this.trace = trace;
      this.canvas = containerEl.querySelector('canvas') || document.createElement('canvas');
      if (!this.canvas.parentElement) {
        this.container.appendChild(this.canvas);
      }
      this.canvas.className = 'atlas-scrubber-canvas';
      this.ctx = this.canvas.getContext('2d');
      this.isDragging = false;

      this._initInteraction();
      this.render();
      this.clock.onChange(() => this.render());
    }

    _initInteraction() {
      const seekFromEvent = (e) => {
        const rect = this.canvas.getBoundingClientRect();
        const clientX = e.clientX ?? (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
        const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
        const ratio = rect.width > 0 ? x / rect.width : 0;
        const targetStep = Math.round(ratio * (this.clock.steps - 1));
        this.clock.seek(targetStep);
      };

      this.container.addEventListener('pointerdown', (e) => {
        this.isDragging = true;
        this.container.setPointerCapture(e.pointerId);
        seekFromEvent(e);
      });

      this.container.addEventListener('pointermove', (e) => {
        if (this.isDragging) {
          seekFromEvent(e);
        }
      });

      const stopDrag = (e) => {
        if (this.isDragging) {
          this.isDragging = false;
          try {
            this.container.releasePointerCapture(e.pointerId);
          } catch {}
        }
      };

      this.container.addEventListener('pointerup', stopDrag);
      this.container.addEventListener('pointercancel', stopDrag);

      this.container.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') {
          e.preventDefault();
          this.clock.step(-1);
        } else if (e.key === 'ArrowRight') {
          e.preventDefault();
          this.clock.step(1);
        } else if (e.key === 'PageDown') {
          e.preventDefault();
          this.clock.step(-10);
        } else if (e.key === 'PageUp') {
          e.preventDefault();
          this.clock.step(10);
        } else if (e.key === 'Home') {
          e.preventDefault();
          this.clock.seek(0);
        } else if (e.key === 'End') {
          e.preventDefault();
          this.clock.seek(this.clock.steps - 1);
        }
      });
    }

    render() {
      if (!this.canvas || !this.ctx) return;
      const { width, height } = fitCanvas(this.canvas);
      if (width <= 0 || height <= 0) return;

      const c = colors();
      const ctx = this.ctx;
      ctx.direction = 'ltr';
      const steps = Math.max(1, this.clock.steps);
      const currentT = this.clock.t;
      const progress = steps > 1 ? currentT / (steps - 1) : 0;

      ctx.clearRect(0, 0, width, height);

      // Track center line
      const midY = height / 2;
      ctx.strokeStyle = c.border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(8, midY);
      ctx.lineTo(width - 8, midY);
      ctx.stroke();

      // Event ticks (e.g. spikes)
      if (this.clock.events && this.clock.events.length > 0) {
        ctx.fillStyle = c.volt;
        const availableW = width - 16;
        for (const ev of this.clock.events) {
          const ratio = steps > 1 ? ev / (steps - 1) : 0;
          const x = 8 + ratio * availableW;
          ctx.fillRect(Math.round(x) - 1, midY - 6, 2, 12);
        }
      }

      // Progress bar fill
      const availableW = width - 16;
      const playheadX = 8 + progress * availableW;
      ctx.strokeStyle = c.volt;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(8, midY);
      ctx.lineTo(playheadX, midY);
      ctx.stroke();

      // Playhead thumb
      ctx.fillStyle = c.volt;
      ctx.beginPath();
      ctx.arc(playheadX, midY, 5, 0, Math.PI * 2);
      ctx.fill();

      // Border on thumb
      ctx.strokeStyle = c.paper;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Update ARIA attributes
      this.container.setAttribute('aria-valuenow', String(currentT));
      this.container.setAttribute('aria-valuemin', '0');
      this.container.setAttribute('aria-valuemax', String(steps - 1));
      this.container.setAttribute('aria-valuetext', `${t('step')} ${currentT} ${t('of')} ${steps - 1}`);
    }
  }

  /**
   * Check if a container element (such as .atlas-margin) is empty.
   * Considers whitespace text nodes as empty.
   * @param {Element|object} el
   * @returns {boolean}
   */
  function isMarginEmpty(el) {
    if (!el) return true;
    if (el.children && el.children.length > 0) return false;
    const text = el.textContent || '';
    return text.trim().length === 0;
  }

  /**
   * Check whether a keyboard event should be ignored by the shell.
   * Ignored if default was already prevented by a plate handler or if focus is in an input.
   * @param {KeyboardEvent|object} e
   * @returns {boolean}
   */
  function shouldIgnoreKeyboardEvent(e) {
    if (!e) return true;
    if (e.defaultPrevented) return true;
    const target = e.target;
    if (!target) return false;
    if (target.isContentEditable) return true;
    const tag = target.tagName ? target.tagName.toUpperCase() : '';
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return true;
    if (typeof target.closest === 'function' && target.closest('input, select, textarea, [contenteditable]')) {
      return true;
    }
    return false;
  }

  /**
   * Helper to construct a margin block element.
   * @param {string|object} labelOrOpts
   * @param {string|Node} maybeContent
   * @returns {Element|null}
   */
  function createMarginBlock(labelOrOpts, maybeContent) {
    if (typeof document === 'undefined') return null;
    const sec = document.createElement('div');
    sec.className = 'atlas-margin-section';
    if (typeof labelOrOpts === 'string') {
      const lbl = document.createElement('div');
      lbl.className = 'atlas-margin-label';
      lbl.textContent = labelOrOpts;
      sec.appendChild(lbl);
      if (typeof maybeContent === 'string') {
        const val = document.createElement('div');
        val.className = 'atlas-margin-value';
        val.textContent = maybeContent;
        sec.appendChild(val);
      } else if (maybeContent && (maybeContent.nodeType || maybeContent instanceof Node)) {
        sec.appendChild(maybeContent);
      }
    } else if (labelOrOpts && typeof labelOrOpts === 'object') {
      if (labelOrOpts.label) {
        const lbl = document.createElement('div');
        lbl.className = 'atlas-margin-label';
        lbl.textContent = labelOrOpts.label;
        sec.appendChild(lbl);
      }
      if (labelOrOpts.value !== undefined) {
        const val = document.createElement('div');
        val.className = 'atlas-margin-value';
        val.textContent = String(labelOrOpts.value);
        sec.appendChild(val);
      }
      if (labelOrOpts.html) {
        sec.insertAdjacentHTML('beforeend', labelOrOpts.html);
      }
      if (labelOrOpts.content && (labelOrOpts.content.nodeType || labelOrOpts.content instanceof Node)) {
        sec.appendChild(labelOrOpts.content);
      }
    }
    return sec;
  }

  // Registry for plate definitions
  const plateRegistry = new Map();
  const Plate = {
    register(spec) {
      if (!spec || !spec.id) {
        throw new Error('Plate spec must define an id');
      }
      plateRegistry.set(spec.id, spec);
    },
    get(id) {
      return plateRegistry.get(id);
    },
    list() {
      return Array.from(plateRegistry.keys());
    }
  };

  /**
   * Bootstraps the plate shell into the DOM.
   * @param {object} spec
   */
  function start(spec = {}) {
    if (typeof document === 'undefined') return;

    if (spec.theme && spec.theme !== 'auto') {
      Theme.set(spec.theme);
    } else {
      Theme.init();
    }

    const rootEl = spec.root || document.getElementById('atlas-root') || document.body;
    const plateId = spec.plateId || 'demo';
    const plate = Plate.get(plateId);
    if (!plate) {
      console.error(`Plate ${plateId} is not registered`);
    }

    const trace = spec.trace instanceof Trace ? spec.trace : new Trace(spec.trace || {});

    const clock = new Clock({
      steps: trace.steps,
      events: trace.events,
      stepsPerSecond: 30,
      loop: true
    });

    const isLive = Boolean(
      spec.live ||
      (trace.meta && trace.meta.live) ||
      spec.source === 'live' ||
      (rootEl && typeof rootEl.getAttribute === 'function' && rootEl.getAttribute('data-live') === 'true')
    );
    if (isLive && rootEl && typeof rootEl.setAttribute === 'function') {
      rootEl.setAttribute('data-live', 'true');
    }

    // Check reduced-motion preference
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      clock.pause();
    } else {
      clock.play();
    }

    function getThemeLabel() {
      const mode = Theme.get();
      const modeName = t(`theme_${mode}`);
      return t('theme_label', { mode: modeName });
    }

    function getSourceBadge(src) {
      const s = (src || 'replay').toLowerCase();
      return t(`badge_${s}`);
    }

    const plateTitle = plate && plate.title
      ? (typeof plate.title === 'object' ? plate.title.en : plate.title)
      : plateId;

    // Build the shell HTML structure
    rootEl.innerHTML = `
      <div class="atlas-plate" lang="en" dir="ltr"${isLive ? ' data-live="true"' : ''}>
        <header class="atlas-header">
          <div class="atlas-header-top">
            <span class="atlas-brand" id="atlas-brand">neuroSush</span>
            <div class="atlas-header-controls">
              <button class="atlas-btn" id="atlas-btn-theme" title="${t('toggle_theme_title')}" aria-label="${t('toggle_theme_aria')}">
                <span id="atlas-theme-text" class="atlas-mono">${getThemeLabel()}</span>
              </button>
              <button class="atlas-btn" id="atlas-btn-help" title="${t('help_btn_title')}" aria-label="${t('help_btn_aria')}">
                <span class="atlas-btn-icon">?</span>
              </button>
            </div>
          </div>
          <div class="atlas-title-group">
            <span class="atlas-plate-id" id="atlas-plate-badge">${t('plate_badge', { id: plateId.toUpperCase() })}</span>
            <h1 class="atlas-title" id="atlas-title">${plateTitle}</h1>
          </div>
        </header>

        <main class="atlas-main" id="atlas-main">
          <aside class="atlas-margin" id="atlas-margin"></aside>
          <section class="atlas-figure" id="atlas-figure">
            <div class="atlas-canvas-container" id="atlas-canvas-container"></div>
          </section>
        </main>

        <nav class="atlas-timeline" id="atlas-timeline"${isLive ? ' style="display: none;" hidden' : ''}>
          <div class="atlas-scrubber-row">
            <div class="atlas-scrubber-track" id="atlas-scrubber" role="slider" tabindex="0" aria-label="${t('scrubber_aria')}">
              <canvas class="atlas-scrubber-canvas" id="atlas-scrubber-canvas"></canvas>
            </div>
          </div>
          <div class="atlas-controls-bar">
            <div class="atlas-button-cluster">
              <button class="atlas-btn" id="atlas-btn-play" title="${t('play_title')}" aria-label="${t('play_aria')}">
                <span class="atlas-btn-icon" id="atlas-play-icon">${clock.isPlaying ? '❚❚' : '▶'}</span>
                <span id="atlas-play-text">${t(clock.isPlaying ? 'pause' : 'play')}</span>
              </button>
              <button class="atlas-btn" id="atlas-btn-step-back" title="${t('step_back_title')}" aria-label="${t('step_back_aria')}">
                <span class="atlas-btn-icon">◀</span>
              </button>
              <button class="atlas-btn" id="atlas-btn-step-fwd" title="${t('step_fwd_title')}" aria-label="${t('step_fwd_aria')}">
                <span class="atlas-btn-icon">▶</span>
              </button>
              <button class="atlas-btn" id="atlas-btn-jump-prev" title="${t('jump_prev_title')}" aria-label="${t('jump_prev_aria')}">
                <span class="atlas-btn-icon">⇤</span>
              </button>
              <button class="atlas-btn" id="atlas-btn-jump-next" title="${t('jump_next_title')}" aria-label="${t('jump_next_aria')}">
                <span class="atlas-btn-icon">⇥</span>
              </button>
            </div>

            <div class="atlas-button-cluster">
              <select class="atlas-speed-select" id="atlas-select-speed" title="${t('speed_title')}" aria-label="${t('speed_aria')}">
                <option value="0.25">0.25x</option>
                <option value="0.5">0.5x</option>
                <option value="1" selected>1.0x</option>
                <option value="2">2.0x</option>
                <option value="4">4.0x</option>
              </select>
              <button class="atlas-btn active" id="atlas-btn-loop" title="${t('loop_title')}" aria-label="${t('loop_aria')}" aria-pressed="true">
                <span id="atlas-loop-text">${t('loop')}</span>
              </button>
              <button class="atlas-btn" id="atlas-btn-fork" title="${t('fork_title')}" aria-label="${t('fork_aria')}">
                <span id="atlas-fork-text">${t('fork')}</span>
              </button>
              <button class="atlas-btn" id="atlas-btn-theory" title="${t('theory_title')}" aria-label="${t('theory_aria')}" aria-pressed="false">
                <span id="atlas-theory-text">${t('theory')}</span>
              </button>
              <button class="atlas-btn" id="atlas-btn-mute" title="${t('mute_title')}" aria-label="${t('mute_aria')}" aria-pressed="false">
                <span id="atlas-mute-text">${t('mute')}</span>
              </button>
            </div>

            <div class="atlas-step-counter" id="atlas-step-counter">
              <span class="atlas-mono" id="atlas-curr-step">${clock.t}</span> / <span class="atlas-mono" id="atlas-total-steps">${Math.max(0, clock.steps - 1)}</span>
            </div>
          </div>
        </nav>

        <footer class="atlas-footer">
          <div class="atlas-footer-item">
            <span>${t('source')}</span>:
            <span class="atlas-badge" id="atlas-source-label">${getSourceBadge(spec.source || (plate && plate.source) || (isLive ? 'live' : 'replay'))}</span>
          </div>
          <div class="atlas-footer-item">
            <span>${t('library')} <span class="atlas-mono">${trace.meta.library ?? ''}</span></span>
          </div>
        </footer>

        <div class="atlas-modal-backdrop" id="atlas-help-modal" style="display: none;">
          <div class="atlas-modal" role="dialog" aria-labelledby="atlas-help-title">
            <div class="atlas-modal-header">
              <h2 class="atlas-modal-title" id="atlas-help-title">${t('help_title')}</h2>
              <button class="atlas-btn" id="atlas-modal-close">${t('close')}</button>
            </div>
            <table class="atlas-shortcut-table">
              <tbody>
                <tr><td><kbd class="atlas-kbd">Space</kbd></td><td>${t('key_space')}</td></tr>
                <tr><td><kbd class="atlas-kbd">←</kbd> / <kbd class="atlas-kbd">→</kbd></td><td>${t('key_arrows')}</td></tr>
                <tr><td><kbd class="atlas-kbd">Shift + ← / →</kbd></td><td>${t('key_shift_arrows')}</td></tr>
                <tr><td><kbd class="atlas-kbd">Home</kbd> / <kbd class="atlas-kbd">End</kbd></td><td>${t('key_home_end')}</td></tr>
                <tr><td><kbd class="atlas-kbd">F</kbd></td><td>${t('key_f')}</td></tr>
                <tr><td><kbd class="atlas-kbd">T</kbd></td><td>${t('key_t')}</td></tr>
                <tr><td><kbd class="atlas-kbd">M</kbd></td><td>${t('key_m')}</td></tr>
                <tr><td><kbd class="atlas-kbd">?</kbd></td><td>${t('key_help')}</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;

    // Initialize timeline scrubber
    const scrubberEl = document.getElementById('atlas-scrubber');
    const timeline = new Timeline(scrubberEl, clock, trace);

    // Context object passed to plate
    const frameListeners = [];
    const figureContainer = document.getElementById('atlas-canvas-container');
    const marginContainer = document.getElementById('atlas-margin');

    if (marginContainer) {
      marginContainer.block = function (labelOrOpts, maybeContent) {
        const sec = createMarginBlock(labelOrOpts, maybeContent);
        if (sec) marginContainer.appendChild(sec);
        return sec;
      };
      marginContainer.section = marginContainer.block;
    }

    const ctx = {
      trace,
      clock,
      get t() { return clock.t; },
      get lang() { return 'en'; },
      get theme() { return Theme.get(); },
      figure: figureContainer,
      margin: marginContainer,
      colors,
      canvas(w, h) {
        const c = document.createElement('canvas');
        c.className = 'atlas-canvas';
        if (w) c.width = w;
        if (h) c.height = h;
        figureContainer.appendChild(c);
        fitCanvas(c);
        if (w && h) {
          figureContainer.style.aspectRatio = `${w} / ${h}`;
        }
        return c;
      },
      onFrame(fn) {
        frameListeners.push(fn);
        return () => {
          const idx = frameListeners.indexOf(fn);
          if (idx !== -1) frameListeners.splice(idx, 1);
        };
      }
    };

    if (typeof ResizeObserver !== 'undefined' && figureContainer) {
      const ro = new ResizeObserver(() => {
        figureContainer.querySelectorAll('canvas.atlas-canvas').forEach(fitCanvas);
      });
      ro.observe(figureContainer);
    }

    // UI elements
    const btnPlay = document.getElementById('atlas-btn-play');
    const playIcon = document.getElementById('atlas-play-icon');
    const playText = document.getElementById('atlas-play-text');
    const btnStepBack = document.getElementById('atlas-btn-step-back');
    const btnStepFwd = document.getElementById('atlas-btn-step-fwd');
    const btnJumpPrev = document.getElementById('atlas-btn-jump-prev');
    const btnJumpNext = document.getElementById('atlas-btn-jump-next');
    const speedSelect = document.getElementById('atlas-select-speed');
    const btnLoop = document.getElementById('atlas-btn-loop');
    const btnFork = document.getElementById('atlas-btn-fork');
    const btnTheory = document.getElementById('atlas-btn-theory');
    const btnMute = document.getElementById('atlas-btn-mute');
    const muteText = document.getElementById('atlas-mute-text');
    const btnTheme = document.getElementById('atlas-btn-theme');
    const btnHelp = document.getElementById('atlas-btn-help');
    const modalHelp = document.getElementById('atlas-help-modal');
    const btnCloseModal = document.getElementById('atlas-modal-close');
    const currStepText = document.getElementById('atlas-curr-step');
    const themeText = document.getElementById('atlas-theme-text');

    // Sync play state
    clock.onChange(() => {
      currStepText.textContent = String(clock.t);
      if (clock.isPlaying) {
        playIcon.textContent = '❚❚';
        playText.textContent = t('pause');
      } else {
        playIcon.textContent = '▶';
        playText.textContent = t('play');
      }
      for (const fn of frameListeners) {
        try {
          fn(clock.t);
        } catch (err) {
          console.error(err);
        }
      }
    });

    btnPlay.addEventListener('click', () => clock.toggle());
    btnStepBack.addEventListener('click', () => clock.step(-1));
    btnStepFwd.addEventListener('click', () => clock.step(1));
    btnJumpPrev.addEventListener('click', () => clock.jumpEvent(-1));
    btnJumpNext.addEventListener('click', () => clock.jumpEvent(1));

    speedSelect.addEventListener('change', (e) => {
      clock.setSpeed(parseFloat(e.target.value));
    });

    btnLoop.addEventListener('click', () => {
      clock.loop = !clock.loop;
      btnLoop.classList.toggle('active', clock.loop);
      btnLoop.setAttribute('aria-pressed', String(clock.loop));
    });

    btnFork.addEventListener('click', () => {
      window.dispatchEvent(new CustomEvent('atlas:fork', { detail: { step: clock.t } }));
    });

    let theoryActive = false;
    btnTheory.addEventListener('click', () => {
      theoryActive = !theoryActive;
      btnTheory.classList.toggle('active', theoryActive);
      btnTheory.setAttribute('aria-pressed', String(theoryActive));
      window.dispatchEvent(new CustomEvent('atlas:theory', { detail: { active: theoryActive } }));
    });

    let muteActive = false;
    btnMute.addEventListener('click', () => {
      muteActive = !muteActive;
      btnMute.classList.toggle('active', muteActive);
      btnMute.setAttribute('aria-pressed', String(muteActive));
      muteText.textContent = t(muteActive ? 'unmute' : 'mute');
      window.dispatchEvent(new CustomEvent('atlas:mute', { detail: { muted: muteActive } }));
    });

    btnTheme.addEventListener('click', () => {
      Theme.cycle();
      themeText.textContent = getThemeLabel();
      timeline.render();
    });

    const toggleHelp = () => {
      const isVisible = modalHelp.style.display !== 'none';
      modalHelp.style.display = isVisible ? 'none' : 'flex';
    };

    btnHelp.addEventListener('click', toggleHelp);
    btnCloseModal.addEventListener('click', toggleHelp);
    modalHelp.addEventListener('click', (e) => {
      if (e.target === modalHelp) toggleHelp();
    });

    // Global keyboard listener
    window.addEventListener('keydown', (e) => {
      // Don't intercept if prevented by plate handler or if typing in input
      if (shouldIgnoreKeyboardEvent(e)) {
        return;
      }
      if (e.code === 'Space') {
        e.preventDefault();
        clock.toggle();
      } else if (e.key === 'ArrowLeft' && e.shiftKey) {
        e.preventDefault();
        clock.jumpEvent(-1);
      } else if (e.key === 'ArrowRight' && e.shiftKey) {
        e.preventDefault();
        clock.jumpEvent(1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        clock.step(-1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        clock.step(1);
      } else if (e.key === 'Home') {
        e.preventDefault();
        clock.seek(0);
      } else if (e.key === 'End') {
        e.preventDefault();
        clock.seek(clock.steps - 1);
      } else if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        if (btnFork) btnFork.click();
      } else if (e.key === 't' || e.key === 'T') {
        e.preventDefault();
        if (btnTheory) btnTheory.click();
      } else if (e.key === 'm' || e.key === 'M') {
        e.preventDefault();
        if (btnMute) btnMute.click();
      } else if (e.key === '?' || (e.shiftKey && e.code === 'Slash')) {
        e.preventDefault();
        toggleHelp();
      } else if (e.key === 'Escape' && modalHelp.style.display !== 'none') {
        e.preventDefault();
        toggleHelp();
      }
    });

    // Mount the plate content
    if (plate && typeof plate.mount === 'function') {
      const mountRoot = figureContainer;
      mountRoot.figure = figureContainer;
      mountRoot.margin = marginContainer;
      plate.mount(mountRoot, ctx);
    }

    // Check if margin is empty after mount; collapse if plate drew nothing into it
    if (isMarginEmpty(marginContainer)) {
      marginContainer.classList.add('atlas-margin-collapsed');
      const mainEl = document.getElementById('atlas-main') || marginContainer.closest('.atlas-main');
      if (mainEl) {
        mainEl.classList.add('atlas-no-margin');
      }
    }

    return { clock, timeline, ctx };
  }

  return {
    unpack,
    Trace,
    Clock,
    Timeline,
    setStrings,
    t,
    Theme,
    niceTicks,
    formatNumber,
    lineSeries,
    fitCanvas,
    colors,
    isMarginEmpty,
    shouldIgnoreKeyboardEvent,
    createMarginBlock,
    Plate,
    start
  };
});
