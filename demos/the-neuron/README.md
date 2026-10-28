# The Neuron Demo

This demo features a leaky bucket neuron and a Hodgkin-Huxley gates model, computed live.

### Controls
- `H`: H model
- `N`: N noise
- `S`: S slow motion
- `Space`: Space pause
- `P`: P pump label
- `D`: D double pulse
- `T`: T spike take-off

### How to run locally
Open `index.html` in any modern web browser.

### URL Shortcuts
- Mode and input: `?mode=hh&input=12`
- Slow motion: `&slow=1`
- Noise: `&noise=1`
- Pump: `&pump=1`
- Double pulse: `&d=1`
- Spike take-off: `&t=1`
- Pause: `&pause=1`

The Gates model uses the same equations as the library's HodgkinHuxley neuron, and the bucket mode was rebuilt for the talk.
