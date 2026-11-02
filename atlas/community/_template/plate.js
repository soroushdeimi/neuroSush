(function (root) {
  'use strict';

  const DEFAULTS = Object.freeze({
    v_rest: -70.0,
    resistance: 10.0,
  });

  function lif_derivative(v, current, params = DEFAULTS) {
    const v_rest = params.v_rest ?? DEFAULTS.v_rest;
    const resistance = params.resistance ?? DEFAULTS.resistance;
    return (v_rest - v) + resistance * current;
  }

  const core = {
    lif_derivative,
  };

  const prefersReducedMotion = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)').matches : false;

  function mount(container, options = {}) {
    // template mount logic
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { core };
  }

  if (typeof Atlas !== 'undefined') {
    Atlas.Plate.register({
      id: 'c-template',
      title: { en: 'Template LIF', fa: 'الگوی LIF' },
      source: 'teaching-js',
      mount,
    });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
