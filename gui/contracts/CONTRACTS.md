## 1. Shared contract (G3 copies this verbatim to `gui/contracts/CONTRACTS.md`; changes go only through the manager)

### 1.1 Common shapes
- `Issue = {"path": "<JSON pointer>", "message": str, "fix": str}`. This is the existing shape.
- Error body for every non-2xx except FastAPI's own 422: `{"error": {"code": str, "message": str, "issues": [Issue]}}`.
  Codes: `not_implemented`(501), `not_found`(404), `conflict`(409), `invalid_spec`(422), `memory_budget`(422),
  `too_many_steps`(422), `bad_transition`(409).
- IDs: project `p_` + 16 hex characters, run `r_` + 16 hex characters (`secrets.token_hex(8)`).
  Times: ISO-8601 UTC strings with a `Z` suffix.
- Data root: `NS_DATA` (default `/data` in Docker, `./.studio-data` otherwise).

### 1.2 NetworkSpec v2 (library `neurosush.structure.network_spec`, built by C1)
```json
{"version": 2,
 "network": {"dt": 1.0, "dtype": "float32", "device": "cpu", "seed": 7, "batch_size": null,
             "independent": false, "behaviors": [BehaviorSpec]},
 "groups":   {"exc": {"shape": 100, "behaviors": [BehaviorSpec], "inhibitory": false, "tags": []}},
 "synapses": [{"name": "exc->inh", "src": "exc", "dst": "inh", "behaviors": [BehaviorSpec],
               "compartment": "proximal", "tags": []}],
 "columns":  {"c1": <v1 ColumnSpec JSON, unchanged>},
 "recordings": [{"target": "exc", "attributes": ["v", "spikes"], "interval": 1, "max_steps": 1000}],
 "assets":   {"frames1": {"sha256": "<64 hex>", "format": "idx|csv|pt|png"}}}
```
- `BehaviorSpec = {"kind": str, "params": {..}, "enabled": true}`. `enabled` is optional and defaults to true.
- `dtype` is one of `float32|float64|float16|bfloat16`. `compartment` is one of `proximal|distal|apical`.
- A synapse `src`/`dst` is a top-level group name or `column.layer.group`.
- A synapse name defaults to `"<src>-><dst>"`. Names must be unique.
- Recordings use `TraceRecorder(*attributes, interval=, max_steps=)`. `target` is a group name, a synapse name or `"network"`.
- Literals allowed inside `params`. Nothing else is decoded, and there is no eval or import of user strings:
  - `{"$tensor": {"dtype","shape","data"}}`: exactly the `viz.trace.pack_array` dict.
  - `{"$tensor": {"asset": "frames1", "shape": [..]}}`: resolved through `build_network(..., assets=resolver)`.
  - `{"$init": "uniform"|"normal"|"constant", ...kwargs}`: drawn from `net.generator`.
  - `{"$fn": "<name in literals.ALLOWED_FUNCTIONS>", "kwargs": {..}}`. A dotted path or unknown name raises `ValueError`
    naming the JSON pointer.
- SpikeInput takes `{"frames": {"$asset": "frames1"}}`. Payoff stays code-only. ScheduledPayoff waits on a human decision.
- Library API:
  - `load_spec(text) -> ColumnSpec | NetworkSpec` (dispatches on `"version"`; no version means v1).
  - `dump_spec(spec) -> str` (canonical: sorted keys).
  - `build_network(spec, assets=None) -> Built(net, groups: dict, synapses: dict, recorders: dict[target, TraceRecorder])`.

### 1.3 REST (all under `/api`; routers in `gui/server/studio/api/<name>.py`, each exporting `router`)
| method path | body | 2xx response | owner |
|---|---|---|---|
| GET `/health` | none | `{"status":"ok","studio":ver,"library":ver,"torch":ver,"cuda":bool,"devices":["cpu",..]}` | G7 |
| GET `/schema` | none | catalog `{version:1, behaviors:[Descriptor]}` + header `ETag: "<fingerprint>"`; 304 on `If-None-Match` | G7 |
| POST `/validate` | v1 ColumnSpec or v2 NetworkSpec | `{"valid":bool,"errors":[Issue]}` (v2 branch: C4) | existing, C4 |
| POST `/probe` | `{"model":BehaviorSpec,"current":float,"dt":1.0,"steps":500}` (steps ≤ 5000) | `{"t":[ms],"v":[mV],"spikes":[step],"provenance":{library,kind,params,dt,steps,current}}` | C4 |
| GET `/projects` | none | `[{"id","name","updated_at","revision"}]` | G6 |
| POST `/projects` | `{"name","spec","layout"?}` | 201 Project | G6 |
| GET `/projects/{id}` | none | Project | G6 |
| PUT `/projects/{id}` | `{"name"?,"spec","layout"?,"base_revision":int}` | Project (revision+1); 409 `conflict` if base_revision is stale | G6 |
| DELETE `/projects/{id}` | none | 204 | G6 |
| GET `/projects/{id}/revisions[/{n}]` | none | `[{"revision","saved_at"}]` / Project at n (last 50 kept) | G6 |
| POST `/runs` | RunRequest | 201 RunInfo | C2 |
| GET `/runs`, `/runs/{id}` | none | `[RunInfo]` / RunInfo | C2 |
| POST `/runs/{id}/pause\|resume\|cancel` | none | RunInfo; 409 `bad_transition` | C2 |
| GET `/runs/{id}/trace?target=exc` | none | `Trace.to_dict()` JSON for that recording target | C2 |
| GET `/runs/{id}/export/html?plate=P01&target=exc` | none | text/html (`render_plate(source="replay")`) | G17 |
| GET `/runs/{id}/export/atlas` | none | application/zip Atlas submission (§4) | A4 |
| WS `/runs/{id}/stream` | §1.5 | §1.5 | C3 |

- `Project = {"id","name","created_at","updated_at","revision":int,"catalog_fingerprint","spec":NetworkSpec,"layout":{node_id:{"x","y"}}}`.
- `Descriptor` gains `"meta": {"host":"neurons|synapse|network","category":str,"requires":[str],"params":{name:{"unit":str|null}}}`.
  It is `{}` when no overlay entry exists.
- `RunRequest = {"project_id"?, "spec": NetworkSpec, "steps": int (1..NS_MAX_STEPS), "chunk": 100, "mode": "watch"|"fast",
  "speed": 1.0, "stream": {"spikes": ["exc"]|"all", "vsample": {"exc": [0,5]}, "scalars": ["rate:exc"],
  "weights": {"exc->inh": 100}}}`.
  - `speed`: in watch mode simulated ms advance at `speed` × wall ms.
  - In fast mode there is no pacing, and display frames are limited to ≤ 30/s with flags bit0 set.
  - Weights are sent every K steps.
- `RunInfo = {"id","project_id","status","step","steps","created_at","started_at","finished_at",
  "error": {"message","traceback"}|null, "provenance": {"library","torch","device","dtype","seed","catalog_fingerprint","spec_sha256"}}`.
- Memory preflight refusal: 422 `memory_budget`, `issues[0].message` includes the estimate in MB and the limit.

### 1.4 Run lifecycle
`queued → building → running ⇄ paused → completed | failed | cancelled`.
- Cancel is allowed from queued, building, running and paused. It sends SIGTERM, then SIGKILL after 5 s.
- Pause and resume happen only between chunks. Terminal states never change.
- One child process per run (multiprocessing spawn). Up to `NS_MAX_JOBS` run at once; the rest queue FIFO.
- Artefacts go in `NS_DATA/runs/<id>/`:
  - `provenance.json`, `log.jsonl`, `spec.json`
  - `traces.json.gz` = `{"format":"neurosush.run-traces/1","traces":{target: Trace.to_dict()}}`
  - `checkpoint.pt` on completion
- Trace meta = `Trace.make_meta(net) + {"run_id","spec_sha256","catalog_fingerprint","target","attributes"}`.
  Recorded traces are never decimated or dropped.

### 1.5 WebSocket `/api/runs/{id}/stream` (trace event format)
**Binary frames** are little-endian. Header (16 B) = `type u8 | flags u8 | stream u16 | seq u32 | step0 u32 | nsteps u16 | reserved u16(=0)`.
Flags: bit0 = decimated or downsampled; bit1 = last frame of this stream. Only batch element 0 is streamed.

| type | stream = | payload |
|---|---|---|
| 1 SPIKES | group index | `nsteps × ceil(n/8)` bytes; bit k of byte j = neuron 8j+k fired (LSB first) |
| 2 VSAMPLE | group index | `count u16, reserved u16, indices u32[count], float32[nsteps][count]` (count ≤ 64) |
| 3 SCALAR | scalar id | `float32[nsteps]` |
| 4 WEIGHTS | synapse index | `rows u16, cols u16, row_stride u16, col_stride u16, float32[rows*cols]` (row = presynaptic; block mean when stride > 1; nsteps = 1; ≤ 256×256) |

Golden frames (hex). Both codecs must round-trip them exactly:
- SPIKES (n=10, seq 1, step0 100, 2 steps; step100 fires {0,9}, step101 fires {3}): `0100000001000000640000000200000001020800`
- VSAMPLE (stream 1, seq 2, step0 100, idx [0,5], [[-65,-64.5],[-60.25,30]]): `02000100020000006400000002000000020000000000000005000000000082c2000081c2000071c20000f041`
- SCALAR (seq 3, step0 0, [0, 0.5, 12.25]): `03000000030000000000000003000000000000000000003f00004441`
- WEIGHTS (flags 1, seq 4, step0 500, 2×3, strides 4,4, [0,.25,.5,.75,1,-1]): `0401000004000000f4010000010000000200030004000400000000000000803e0000003f0000403f0000803f000080bf`

**Text frames** are JSON. The server sends `{"type", "seq", ...}`:
- `hello`: `{"protocol":1,"run":RunInfo}`. Always sent first.
- `started`:
  - `{"dt","groups":[{"stream","name","size","inhibitory"}]`
  - `"synapses":[{"stream","name","src","dst","shape":[pre,post]}]`
  - `"scalars":[{"id","name","unit"}]`, `"vsample":[{"stream","group","indices"}]`
  - `"schedule": viz.schedule.schedule_table(net)}`
  - Replayed to clients that join late.
- `status`: `{"status","step"}` on every lifecycle transition.
- `progress`: `{"step","steps","steps_per_s"}` at most 4/s.
- `log`: `{"level":"info|warning|error","message"}`. `issue`: `{"issue":Issue}`.
- `dropped`: `{"frames":int,"from_seq":int}`.
- `finished`: `{"status","step","error"|null,"artifacts":[names]}`. The server then closes with code 1000.

Client → server: `{"type":"ack","seq"}` | `{"type":"pace","mode","speed"}` | `{"type":"select","group","indices":[≤64]}`.
- `seq` is one monotonic counter per connection, shared by binary and text frames.
- Backpressure: at most 8 binary frames may be unacked; past that, the oldest display frames are dropped and `dropped` is sent.
- History is not replayed over WS. Use `/trace`.
- `Origin` must match `NS_ALLOWED_ORIGINS` (default: same origin). If `NS_AUTH_TOKEN` is set, `?token=` is required.

Scalar names: `rate:<group>` (spikes per neuron per step, unit `1/step`), `dopamine`, `loss`, `accuracy`, `anomaly`.
