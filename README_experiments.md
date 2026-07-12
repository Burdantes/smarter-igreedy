# Audit-vs-task measurement experiments

These scripts explore one question for IP geolocation under a ping budget:

> Given a limited number of pings, should you spend them **locating the target**
> (task measurements) or **testing/improving the latency model** you use to
> locate it (audit measurements)? When does auditing the model pay?

This is the "recursive Bayesian experimental design" framing: a measurement
system holds uncertainty over both the **answer** (target location) and the
**model** that turns RTTs into locations, and must split its budget between the
two. The scripts here build that comparison on synthetic and **real RIPE Atlas**
data. For the underlying method (the geolocation problem, the information
boundary, the model ladder) see `SIMULATION_ENVIRONMENT.md` and `CLAUDE.md`.

---

## TL;DR of the findings

- **On real data, auditing the additive model never pays.** Across every regime
  tested (25–355 targets, dense 67-anchors/target and sparse 6/target, geodesic
  and fiber base), the optimal share of budget spent on VP→VP audit pings is
  **~0%**. See `audit_vs_task_real.pdf`, `audit_vs_task_all_n200.pdf`,
  `scenario_100x*.pdf`.
- **The lever that *does* work is the base model.** Swapping the geodesic
  `distance/100` base term for the **fiber-floor** atlas (`internet_gmaps`) cuts
  error ~20–30% and beats the model-free nearest-neighbor baseline. See
  `audit_vs_task_fiber.pdf`.
- **Why auditing loses on real data:** the per-node offsets that audits pin are
  (a) already well-identified by ordinary task pings (pooling across many
  targets), and (b) not where the reducible error lives — that error is
  *structural* (distance-dependent routing overhead), which a fixed per-node
  offset cannot absorb and only a better base model (fiber) fixes.
- **Auditing *does* pay in the synthetic world** (`audit_vs_task.pdf`, optimal
  ~30% audit) — because that world is built so the dominant error IS a large,
  heterogeneous, under-identified per-node offset. Real anchors don't satisfy
  that precondition.

---

## Setup

1. **Python env.** Needs `numpy`, `scipy`, `matplotlib`. Loqman ran these with
   the `smarter-igreedy` conda env:
   ```bash
   conda activate smarter-igreedy
   ```
   (Any env with the scientific stack + the repo importable works.)

2. **The fiber atlas graph is already committed** at
   `internet_gmaps/data/graph_2026-07-04.npz` — no extra download for the fiber
   experiments. Note: they use the plain `FloorEstimator` (not
   `PolicyFloorEstimator`) to avoid a `reverse_geocoder` dependency.

3. **The RIPE mesh is NOT committed** (it's large and gitignored). Pull it once
   before running any real-data experiment (see next section).

Run everything from the repo root.

---

## Step 0 — pull the real RIPE Atlas mesh (once)

```bash
python pull_minimal_mesh.py 2026-07-08 0 1     # date, then hour indices
```

- Downloads a **couple of ~2.1 GB hourly ping dumps** from RIPE's public data
  store, aggregates each to a tiny summary (raw deleted immediately), and builds
  `cache/real_mini_mesh.pkl` — a probe-to-probe latency mesh
  (`{address_to_loc, loc_loc_meas}`).
- ~5–10 min, ~4 GB transient download. Only hours `0 1` are pulled to keep it
  minimal; add more hour indices for denser min-RTTs.
- **Caveat:** this is a **single 2-hour snapshot**, so min-RTTs are less
  converged than a full-day mesh. The mesh is the RIPE *anchoring* mesh: ~880
  nodes densely ping each other; the rest of the ~11.5k nodes are sparse.

Everything except the two synthetic scripts (`experiment_audit_vs_task.py`,
`experiment_geoloc_animation.py`) needs this pickle.

---

## The scripts

Outputs land in `figures/` (PDF + PNG) unless noted. Fiber runs are slow
(graph shortest-path per query; ~40–100 s per estimate at scale) — see
"Performance" below.

### Estimation / audit-value experiments

| Script | Data | Run | Output | What it shows |
|---|---|---|---|---|
| `experiment_audit_vs_task.py` | synthetic | `python experiment_audit_vs_task.py` | `audit_vs_task.pdf` | **The mechanism working.** A controlled world where the dominant error is a per-node offset → audits break the plateau, overconfidence gap, optimal ~30% audit. |
| `experiment_audit_vs_task_real.py` | mesh | `python experiment_audit_vs_task_real.py` | `audit_vs_task_real.pdf` | Real data, geodesic base. Auditing does NOT pay (optimal 0%); NN is strong. Also the **module that defines the submesh + sweep functions the others import.** |
| `experiment_audit_vs_task_fiber.py` | mesh + fiber | `python experiment_audit_vs_task_fiber.py` | `audit_vs_task_fiber.pdf` | Geodesic vs fiber base. **Fiber breaks the plateau & beats NN**; audits still 0%. |
| `experiment_audit_value_200.py` | mesh | `python experiment_audit_value_200.py` | `audit_vs_task_real_n200.pdf` | Hardens the 0%-audit result at **200 targets**, geodesic. |
| `experiment_audit_value_all.py` | mesh + fiber | `python experiment_audit_value_all.py` | `audit_vs_task_all_n200.pdf` | **All five categories** (NN, task/audit × geodesic/fiber) at 200 targets. The definitive comparison. |
| `experiment_scenario_100x150.py` | mesh + fiber | `python experiment_scenario_100x150.py [BUDGET] [TARGETS]` | `scenario_100x<ntgt>_b<budget>.pdf` | Concrete "100 anchors, N targets, B pings" audit-fraction sweep. Defaults: `10000 150`. Examples run: `10000 150` (dense, ~67 anchors/tgt), `900 150` (sparse, ~6/tgt), `14000 700` (→355 targets selected). |

### Visual explainers

| Script | Data | Run | Output | What it shows |
|---|---|---|---|---|
| `experiment_geoloc_animation.py` | synthetic | `python experiment_geoloc_animation.py` | `geoloc_animation.gif` | Idealized trilateration: with a correct model, adding vantage points tightens the feasible region and the MAP estimate beats nearest-neighbor. |
| `experiment_polygon_animation.py` | mesh + fiber | `python experiment_polygon_animation.py` | `polygon_animation.gif` | Real target, 25 sources: classical **speed-of-light polygon (CBG) centroid**, geodesic circle vs **fiber isochrone**, next to NN and MAP, with error-vs-#sources. |
| `generate_model_report.py` | mesh + fiber | `python generate_model_report.py` | `model_report.html` | **Interactive HTML.** Browse each source/target node: fitted parameters (μ = per-node overhead, σ = noise), the measurements + residuals behind them, and (for targets) a map thumbnail of true vs estimated location. Toggle geodesic/fiber base. Open in a browser. |

---

## The model being fit (context for reading the report)

All estimators share the two-way **additive latency model**:

```
rtt(source → dst)  ≈  base(source, dst)  +  μ_source  +  μ_dst   (+ noise σ)
```

- `base` = propagation floor: `geodesic_km / 100` (geodesic) or the fiber-floor
  atlas value (fiber). This is the **distance-scaling** term.
- `μ_source`, `μ_dst` = each node's **fixed** access/routing overhead (ms).
- Fit by `additive_batch_em` (`probabilistic_helpers.py`): self-supervised EM
  that alternates (M) pooling residuals across all targets to fit the μ's and
  (E) MAP-relocating each target. **No ground truth is used in the fit** — truth
  is only touched to *score* error afterward.

**A task ping** is `source → target` (localizes a target). **An audit ping** is
`source → source`, between two known landmarks — its residual is a clean read of
`μ_source + μ_dst` with no location ambiguity, so it pins the model's per-node
offsets. In the code (`experiment_audit_vs_task_real.py::estimate`), audit
endpoints are "pinned" to their true location and excluded from the location
optimization; the audit fraction of the budget is swept to find the optimum.

Ground truth for scoring = the RIPE probe registry coordinates (self-reported,
/24-aggregated — imperfect, so a few hundred km of error is registry noise).

---

## Key knobs (in `experiment_audit_vs_task_real.py`)

The other real-data scripts import this module and override these globals:

- `N_SOURCES`, `N_TARGETS` — submesh size.
- `REGION` — lat/lon box (default Central/Western Europe; keeps distances
  overhead-dominated so the additive model is well-specified).
- `MIN_SRC_DIST_KM = 250` — exclude anchors within 250 km of a target, forcing
  genuine triangulation (benchmark construction only; uses truth to *build* the
  scenario, never during inference).
- `COVERAGE_CAP` — max distinct anchors per target (experiments use 6 to
  simulate scarcity; the scenario scripts set it to 100 = effectively uncapped).

---

## Performance notes

- Geodesic estimates are fast (seconds). **Fiber estimates are slow** — each
  base-RTT is a graph shortest-path query (~0.26 ms) and the MAP optimizer calls
  it ~hundreds of times per target. A single fiber estimate is ~40–100 s at
  200–355 targets.
- The scripts set `fiber._FLOOR_CACHE_MAX` high so the per-point floor cache
  stays warm across a whole sweep (cold ~100 s → warm ~2–5 s).
- **Not yet built (open speedup):** precompute floors on a lat/lon grid once and
  bilinearly interpolate (`GriddedFiberRtt`) → ~1000× faster per call + disk
  cache. Worth doing before any larger fiber sweep.

---

## Demonstrating the effect (feedback loop, drift, scale)

Later experiments probe *when auditing the model actually pays* and use a fast
fiber model.

- `gridded_fiber.py` — **GriddedFiberRtt**, a drop-in for FiberFloorRtt that
  precomputes the floor on a lat/lon grid and bilinearly interpolates
  (~1000x faster per call, mean error 0.012 ms vs exact). Use this for any
  fiber run at scale.
- `experiment_feedback_loop.py` → `feedback_loop.pdf` — model-guided GREEDY
  selection vs random, geodesic vs fiber base. Shows the feedback loop:
  under a **biased (geodesic)** base, greedy is *confidently wrong* and
  audits help; under a **good (fiber)** base, greedy is *beneficial* and
  audits are not needed. (Greedy is per-candidate expensive, so this runs on
  a modest submesh.)
- `experiment_drift.py` → `drift.pdf` — three real snapshots 26 days apart
  (`pull_minimal_mesh.py <date> 0 1` for each of 2026-06-12/-06-28/-07-08),
  frozen vs audit-refresh vs task-refresh anchor offsets. Result: geolocation
  is **robust to 26-day drift** — a frozen model does not degrade (drift
  averages out over many anchors + per-target recalibration).
- `experiment_scale_split.py` → `scale_split.pdf` — the full run: split the
  dense anchor mesh into 350 sources x 500 targets (global, all-pairs) and
  compare NN, task/audit x geodesic/fiber over a budget sweep + audit
  allocation.
- `experiment_greedy_scale.py` → `greedy_scale.pdf` — the **robust** greedy vs
  random comparison, using the project's real `Iterative_Greedy_Geolocator`
  (batch region-based, not the fragile incremental harness). 60 src x 40 dst,
  geodesic and fiber. Result: greedy has a startup cost (worse than NN/random
  at low budget), but GREEDY + fiber is the best strategy at sufficient budget
  (465 vs NN 626 km at b=1200); geodesic greedy ≈ random+additive. Must be run
  as a script (multiprocessing).

**Unified finding across all of these:** the base model (fiber) is the lever;
**auditing per-node offsets never pays on real anchor data** (optimal audit
share ~0%); model-based triangulation beats NN in the sparse/budget-limited
regime, NN wins once coverage is dense; auditing only helps when the base
model itself is biased/underspecified (the geodesic feedback-loop panel).

## How many targets are available?

The mesh has ~11,570 nodes, but it's the RIPE **anchoring mesh**: ~880 nodes are
densely pinged by ~all anchors, and the rest are sparse. So from 100 anchors,
**~880 densely-covered targets** are the practical ceiling (it's a cliff, not a
gradient). Under the dense-coverage selection (≥10 far anchors), ~355 qualify.
To go higher, use more anchors or accept sparse coverage from the probe tail.
