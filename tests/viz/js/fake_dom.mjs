/**
 * Minimal fake DOM for loading a plate and calling mount()/render() under node.
 * It is permissive on purpose: elements created through innerHTML are indexed
 * by their id="..." so getElementById finds them, and canvas 2D contexts record
 * calls instead of drawing. Anything the plate reads that does not exist raises
 * the same ReferenceError/TypeError it would raise in a browser.
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
export const WEB = path.resolve(here, '../../../src/neurosush/viz/web');

function noopProxy(record) {
  const fn = (...args) => undefined;
  return new Proxy(fn, {
    get(target, prop) {
      if (prop === 'measureText') return (text) => ({ width: String(text).length * 6 });
      if (prop === 'createLinearGradient' || prop === 'createRadialGradient') {
        return () => ({ addColorStop() {} });
      }
      if (prop === 'getImageData') return (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4) });
      if (prop === Symbol.toPrimitive) return () => 0;
      if (record && prop in record) return record[prop];
      return (...args) => undefined;
    },
    set(target, prop, value) {
      if (record) record[prop] = value;
      return true;
    },
  });
}

export function makeContext2d() {
  const state = {};
  const calls = { count: 0 };
  const proxy = new Proxy(state, {
    get(target, prop) {
      if (prop === '__calls') return calls;
      if (prop in target) return target[prop];
      if (prop === 'measureText') return (text) => ({ width: String(text).length * 6 });
      if (prop === 'createLinearGradient' || prop === 'createRadialGradient') {
        return () => ({ addColorStop() {} });
      }
      if (prop === 'getImageData') return (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4) });
      if (prop === 'canvas') return undefined;
      return (...args) => {
        calls.count++;
        return undefined;
      };
    },
    set(target, prop, value) {
      target[prop] = value;
      return true;
    },
  });
  return proxy;
}

class FakeClassList {
  constructor() { this.set = new Set(); }
  add(...n) { n.forEach((x) => this.set.add(x)); }
  remove(...n) { n.forEach((x) => this.set.delete(x)); }
  toggle(n, f) { const on = f === undefined ? !this.set.has(n) : !!f; on ? this.set.add(n) : this.set.delete(n); return on; }
  contains(n) { return this.set.has(n); }
}

export class FakeDocument {
  constructor() {
    this.byId = new Map();
    this.listeners = {};
    this.head = new FakeElement('head', this);
    this.body = new FakeElement('body', this);
    this.documentElement = new FakeElement('html', this);
    this.readyState = 'complete';
    this.defaultView = null;
  }
  createElement(tag) { return new FakeElement(tag, this); }
  createElementNS(ns, tag) { return new FakeElement(tag, this); }
  createTextNode(text) { const e = new FakeElement('#text', this); e.textContent = String(text); return e; }
  getElementById(id) { return this.byId.get(id) || null; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener() {}
}

export class FakeElement {
  constructor(tag, doc) {
    this.tagName = String(tag).toUpperCase();
    this.ownerDocument = doc;
    this.children = [];
    this.parentElement = null;
    this.parentNode = null;
    this.listeners = {};
    this.attrs = {};
    this.dataset = {};
    this.classList = new FakeClassList();
    this._style = {};
    this.style = new Proxy(this._style, {
      get: (t, p) => {
        if (p === 'setProperty') return (k, v) => { t[k] = v; };
        if (p === 'removeProperty') return (k) => { delete t[k]; };
        if (p === 'getPropertyValue') return (k) => t[k] || '';
        return t[p];
      },
      set: (t, p, v) => { t[p] = v; return true; },
    });
    this._text = '';
    this._html = '';
    this._ctx = null;
    this.width = 300;
    this.height = 150;
    this._value = '';
    this.checked = false;
    this.disabled = false;
    this.hidden = false;
    this.open = false;
    this.scrollTop = 0;
    this.clientWidth = 800;
    this.clientHeight = 500;
    this.offsetWidth = 800;
    this.offsetHeight = 500;
  }
  get value() {
    if (this.tagName === 'SELECT' && this._value === '') {
      const opts = this._all().filter((e) => e.tagName === 'OPTION');
      const pick = opts.find((o) => 'selected' in o.attrs) || opts[0];
      return pick ? (pick.attrs.value ?? pick._text) : '';
    }
    return this._value;
  }
  set value(v) { this._value = String(v); }
  get id() { return this.attrs.id || ''; }
  set id(v) { this.setAttribute('id', v); }
  get className() { return this.attrs.class || ''; }
  set className(v) { this.attrs.class = String(v); String(v).split(/\s+/).filter(Boolean).forEach((c) => this.classList.add(c)); }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get innerText() { return this._text; }
  set innerText(v) { this._text = String(v); }
  get innerHTML() { return this._html; }
  set innerHTML(v) {
    this._html = String(v);
    this.children = [];
    const VOID = new Set(['input', 'br', 'hr', 'img', 'meta', 'link', 'source']);
    const RAW = new Set(['style', 'script']);
    const tagRe = /<(\/?)([a-zA-Z][a-zA-Z0-9]*)\b((?:"[^"]*"|'[^']*'|[^>"'])*)>/g;
    const stack = [this];
    let last = 0;
    let m;
    const addText = (text, top) => {
      if (text && top.tagName === 'OPTION') top._text += text;
    };
    while ((m = tagRe.exec(this._html))) {
      const top = stack[stack.length - 1];
      addText(this._html.slice(last, m.index), top);
      last = tagRe.lastIndex;
      const name = m[2].toLowerCase();
      if (m[1]) {
        for (let i = stack.length - 1; i > 0; i--) {
          if (stack[i].tagName.toLowerCase() === name) { stack.length = i; break; }
        }
        continue;
      }
      const el = new FakeElement(name, this.ownerDocument);
      const attrRe = /([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g;
      let am;
      while ((am = attrRe.exec(m[3]))) {
        const val = am[2] ?? am[3] ?? am[4] ?? '';
        el.setAttribute(am[1], val);
        if (am[1] === 'value') el._value = val;
        if (am[1] === 'checked') el.checked = true;
        if (am[1] === 'disabled') el.disabled = true;
      }
      el.parentElement = el.parentNode = top;
      top.children.push(el);
      if (RAW.has(name)) {
        const end = this._html.indexOf('</' + name, last);
        last = tagRe.lastIndex = end < 0 ? this._html.length : this._html.indexOf('>', end) + 1;
      } else if (!VOID.has(name) && !m[3].trim().endsWith('/')) {
        stack.push(el);
      }
    }
  }
  setAttribute(k, v) {
    this.attrs[k] = String(v);
    if (k === 'id') this.ownerDocument.byId.set(String(v), this);
  }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  hasAttribute(k) { return k in this.attrs; }
  _adopt(child) {
    if (child.parentElement) child.parentElement.children = child.parentElement.children.filter((c) => c !== child);
    child.parentElement = child.parentNode = this;
    if (child.attrs.id) this.ownerDocument.byId.set(child.attrs.id, child);
    return child;
  }
  appendChild(c) { this.children.push(this._adopt(c)); return c; }
  append(...cs) { cs.forEach((c) => { if (typeof c === 'string') return; this.appendChild(c); }); }
  prepend(...cs) { cs.forEach((c) => { if (typeof c !== 'string') this.children.unshift(this._adopt(c)); }); }
  insertBefore(c, ref) {
    this._adopt(c);
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(c); else this.children.splice(i, 0, c);
    return c;
  }
  before() {}
  after() {}
  remove() {
    if (this.parentElement) this.parentElement.children = this.parentElement.children.filter((c) => c !== this);
    if (this.attrs.id) this.ownerDocument.byId.delete(this.attrs.id);
  }
  replaceChildren(...cs) { this.children = []; cs.forEach((c) => this.appendChild(c)); }
  contains(other) {
    if (other === this) return true;
    return this.children.some((c) => c.contains(other));
  }
  _matches(p) {
    const m = /^([a-zA-Z0-9]*)((?:[#.][-\w]+|\[[-\w]+(?:=[^\]]*)?\])*)$/.exec(p);
    if (!m) return false;
    if (m[1] && this.tagName !== m[1].toUpperCase()) return false;
    for (const part of m[2].match(/[#.][-\w]+|\[[^\]]+\]/g) || []) {
      if (part[0] === '#') { if (this.attrs.id !== part.slice(1)) return false; }
      else if (part[0] === '.') { if (!this.classList.contains(part.slice(1))) return false; }
      else {
        const [k, v] = part.slice(1, -1).split('=');
        if (!(k in this.attrs)) return false;
        if (v !== undefined && this.attrs[k] !== v.replace(/^["']|["']$/g, '')) return false;
      }
    }
    return true;
  }
  closest(sel) {
    for (let el = this; el; el = el.parentElement) if (el._matches(sel)) return el;
    return null;
  }
  _all() {
    const out = [];
    for (const c of this.children) { out.push(c, ...c._all()); }
    return out;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  querySelectorAll(sel) {
    const parts = String(sel).split(',').map((s) => s.trim());
    return this._all().filter((el) => parts.some((p) => el._matches(p)));
  }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener() {}
  dispatchEvent(ev) { (this.listeners[ev.type] || []).forEach((fn) => fn(ev)); return true; }
  click() { this.dispatchEvent({ type: 'click', target: this, preventDefault() {} }); }
  focus() {}
  blur() {}
  setPointerCapture() {}
  releasePointerCapture() {}
  getBoundingClientRect() {
    return { left: 0, top: 0, right: this.clientWidth, bottom: this.clientHeight, width: this.clientWidth, height: this.clientHeight, x: 0, y: 0 };
  }
  getContext(kind) {
    if (kind !== '2d') return null;
    if (!this._ctx) { this._ctx = makeContext2d(); this._ctx.canvas = this; }
    return this._ctx;
  }
  toDataURL() { return 'data:,' + this._ctx?.__calls.count; }
}

/**
 * Create a browser-like global scope, load Atlas + one plate into it with
 * vm so the plate's own `Atlas` / `window` free variables resolve, and return
 * the registered plate plus helpers to mount it.
 */
export function loadPlate(id, { width = 1000, height = 700 } = {}) {
  const document = new FakeDocument();
  const frames = [];
  const timers = [];
  const win = {
    document,
    innerWidth: width,
    innerHeight: height,
    devicePixelRatio: 1,
    addEventListener() {},
    removeEventListener() {},
    matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
    getComputedStyle: () => ({ getPropertyValue: () => '' }),
    requestAnimationFrame: (fn) => { frames.push(fn); return frames.length; },
    cancelAnimationFrame() {},
    setInterval: (fn) => { timers.push(fn); return timers.length; },
    clearInterval() {},
    setTimeout: () => 0,
    clearTimeout() {},
    performance: { now: () => 0 },
    console,
    Math, Date, JSON, Object, Array, Number, String, Float32Array, Float64Array, Uint8Array, Uint32Array, Int32Array,
    Map, Set, Promise, Symbol, Error, TypeError, RangeError, isFinite, isNaN, parseFloat, parseInt, Infinity, NaN,
    Intl, RegExp, Boolean, WeakMap, ArrayBuffer, DataView, Proxy, Reflect, BigInt,
    localStorage: { getItem: () => null, setItem() {} },
    ResizeObserver: undefined,
    MutationObserver: class { observe() {} disconnect() {} },
    IntersectionObserver: class { observe() {} disconnect() {} },
    module: undefined,
  };
  win.window = win;
  win.globalThis = win;
  win.self = win;
  document.defaultView = win;
  const context = vm.createContext(win);
  const run = (file) => {
    const full = path.join(WEB, file);
    vm.runInContext(fs.readFileSync(full, 'utf8'), context, { filename: full });
  };
  run('atlas.js');
  if (!win.Atlas) throw new Error('atlas.js did not expose Atlas');
  run(path.join('plates', `${id}.js`));
  const plate = win.Atlas.Plate.get(id);
  if (!plate) throw new Error(`plate ${id} did not register`);

  function mount() {
    const figure = document.createElement('div');
    figure.setAttribute('id', 'atlas-figure');
    const margin = document.createElement('div');
    margin.setAttribute('id', 'atlas-margin');
    const plateRoot = document.createElement('div');
    plateRoot.className = 'atlas-plate';
    plateRoot.append(figure, margin);
    document.body.append(plateRoot);
    const ctx = {
      figure,
      margin,
      trace: null,
      clock: { steps: 0, t: 0 },
      lang: 'en',
      theme: 'light',
      colors: () => ({
        paper: '#f6f2e8', paperCard: '#eeeae0', ink: '#111', inkMuted: '#57534e', muted: '#57534e',
        border: '#d6cfbf', rule: '#d6cfbf', exc: '#b45309', inh: '#1d4ed8', volt: '#0f766e', spike: '#b91c1c',
        accent: '#b45309', trace: '#0f766e',
      }),
      canvas: (w, h) => {
        const c = document.createElement('canvas');
        if (w) c.width = w;
        if (h) c.height = h;
        figure.appendChild(c);
        return c;
      },
      onFrame(fn) { frames.push(fn); return () => {}; },
    };
    plate.mount(figure, ctx);
    return { figure, margin, ctx, document };
  }

  // Run every requestAnimationFrame / setInterval callback once (a "frame").
  function tick(n = 1) {
    for (let i = 0; i < n; i++) {
      const fns = frames.splice(0).concat(timers);
      fns.forEach((fn) => fn(performance.now()));
    }
  }

  return { win, plate, mount, tick, document };
}
