# Recursive Bayesian experimental design for IP geolocation — results

This folder holds a self-contained study of **task vs. audit measurement
selection** for IP geolocation under a ping budget: given a limited number of
pings, should you spend them *locating the target* (task measurements) or
*testing/improving the latency model* you use to locate it (audit
measurements)? Which is the better investment, and when?

Everything runs on **real RIPE Atlas** latency data (plus a controlled synthetic
world to prove the mechanism). See `README.md` for how to run each script; this
file is the summary of what we found.

---

## Headline verdict

> **The base model is the lever, not auditing.** Swapping the geodesic `d/100`
> base term for the fiber-floor atlas cuts geolocation error ~20–36% at every
> scale. **Auditing the latency model's per-node offsets does not robustly pay
> for geolocation in any regime we could construct** (optimal audit share ≈ 0%).
> Geolocation is therefore the *disciplining / boundary* case for recursive-BED:
> it shows precisely *when the framework should say "don't audit," and why.*
> The "auditing pays" story belongs to tasks whose model is **non-poolable and
> structurally uncertain** (routing inference, fault localization).

Auditing pays **iff** three conditions hold simultaneously: (1) model
uncertainty dominates the answer's error, (2) the *audited component* is the
uncertain part, and (3) the audit reduces it more cheaply than task pings.
Geolocation violates (2)+(3): the reducible error is *structural* (the
distance→latency map, fixed by external fiber data — not pings), and the
auditable per-node offsets are *already pinned* by task pings via pooling.

---

## Hierarchy of levers (impact, largest first)

1. **Base-model structure** — fiber floor ≫ geodesic, everywhere (~20–36%).
   The single biggest lever; comes from *external infrastructure data*, not pings.
2. **Coverage / a NN crossover** — nearest-neighbor wins when a VP sits close to
   the target (dense coverage); model-based triangulation wins in the sparse /
   budget-limited regime.
3. **Selection** — model-guided greedy helps *only* with a good base model and
   enough budget to amortize a startup cost; on a *biased* base it backfires
   ("confidently wrong").
4. **Auditing per-node offsets** — ≈ zero lever on real anchor data
   (optimal audit share ~0% across every regime).

---

## Experiments and results

### A. The mechanism exists (synthetic) — `experiment_audit_vs_task.py`
A controlled world where the dominant error is a large, under-identified
per-node offset. **Audits win: optimal audit share ~30%**, task-only plateaus,
overconfidence gap opens. Proves the tradeoff is real when its precondition holds.
→ `plots/mechanism/audit_vs_task.png`

### B. Real data, auditing doesn't pay — `experiment_audit_vs_task_real.py`, `_fiber.py`, `_value_200.py`, `_value_all.py`
- Geodesic base: audits never pay; **NN is strong**; optimal audit share 0%.
- **Fiber base breaks the plateau and beats NN** at high budget.
- Hardened at **200 targets** and across all categories (NN, task/audit ×
  geodesic/fiber): fiber best (task-only fiber 493 vs NN 626 km at b=1600);
  audit share 0% under both bases.
→ `plots/real_audit/audit_vs_task_real.png`, `_fiber.png`, `audit_vs_task_all_n200.png`

### C. Full anchor mesh, split src/dst — `experiment_scale_split.py`
350 src × 500 dst (global, all-pairs). Fiber > geodesic; audit 0%-optimal for
both bases; NN vs model is coverage-dependent (fiber model wins at moderate
coverage, NN at dense).
→ `plots/scale/scale_split_350x500.png` (and `scale_split_50x919.png`, the
sparse-source run)

### D. Concrete scenarios — `experiment_scenario_100x150.py`
100 anchors, 150 targets, budget sweeps (10,000 and 900 pings). 10% audit is a
wash-to-slightly-negative; fiber base −22% mean error. At 900 pings (~6
anchors/target) the geodesic model is *worse* than NN (misspecified
triangulation with few rings); fiber rescues it.
→ `plots/scale/scenario_100x150_b10000.png`, `scenario_100x150_b900.png`

### E. Robust greedy vs random — `experiment_greedy_scale.py`
Uses the project's real `Iterative_Greedy_Geolocator` (batch, region-based).
Greedy has a startup cost (worse than NN/random at low budget), but
**greedy + fiber is best at sufficient budget** (465 vs NN 626 km at b=1200);
geodesic greedy only ties random+additive.
→ `plots/scale/greedy_scale_60x40.png`

### F. The feedback loop (confidently wrong) — `experiment_feedback_loop.py`
Model-guided greedy on a **biased (geodesic)** base is *confidently wrong* —
widest overconfidence gap (true 1288 / reported 714 vs random 1257 / 800), and
**audits break the loop** (true error → 1116). The pathology is *bias-dependent*:
with a good base, greedy is beneficial and audits aren't needed. (Geodesic-only,
regional — the incremental greedy estimator is unreliable under the fiber base.)
→ `plots/dynamics/feedback_loop.png`

### G. Model drift over 26 days — `experiment_drift.py`
Three real snapshots (2026-06-12, -06-28, -07-08), ~5.9 ms mean |ΔRTT| at 26
days. **Geolocation is robust to drift**: a frozen model does *not* degrade
(945 → 967 km over 26 days) — redundant coverage + per-target recalibration wash
it out. So "keep auditing because the model goes stale" doesn't bite here.
→ `plots/dynamics/drift.png`

### H. Scarce VPs + sparse targets — `experiment_scarce_vp.py`
The regime where auditing *should* pay. Tested 10–12 VPs × 30–300 targets.
**Definitive (12 seeds): auditing does not robustly pay** — optimal audit share
0% (geo 1052→1236, fiber 669→890 as audit% rises). Apparent wins at few targets
with 3 seeds were **seed noise**. Structural reason: no regime has offsets
simultaneously (a) binding, (b) unpooled by task pings, (c) geometrically
resolvable. Fiber still ~36% better than geodesic.
→ `plots/dynamics/scarce_vp_12vp.png`

### Visual explainers
- `experiment_geoloc_animation.py` → `plots/animations/geoloc_animation.gif` — idealized
  trilateration (correct model): feasible region tightens, MAP beats NN.
- `experiment_polygon_animation.py` → `plots/animations/polygon_animation.gif` — classical
  speed-of-light (CBG) polygon centroid, geodesic circle vs fiber isochrone.
- `generate_model_report.py` → `model_report.html` — interactive per-node view
  of the fitted additive model (parameters + the measurements behind them +
  per-target map thumbnails).

---

## Methods notes

- **Model:** additive latency `rtt(a,b) ≈ base(a,b) + μ_a + μ_b`, base =
  geodesic `d/100` or the fiber floor. Offsets fit self-supervised (no ground
  truth). Audit = VP↔VP ping (both known → clean read of the offsets).
- **Fiber speedup:** `gridded_fiber.GriddedFiberRtt` precomputes the floor on a
  lat/lon grid and bilinearly interpolates (~1000× faster than the exact
  shortest-path query; unreachable/ocean cells filled with the geodesic floor so
  interpolation never leaks huge values).
- **Estimator speed:** vectorize the MAP objective (NumPy over constraints, not
  a Python loop) → ~100× on the batch/scarce experiments.
- **Ground truth:** RIPE probe-registry coordinates (self-reported,
  /24-aggregated — a noise floor on scoring). Used only for evaluation.

## Limitations

- Single 2-hour snapshots; drift over 3 snapshots spanning 26 days; the RIPE
  **anchor mesh** (dense, well-behaved — not the general edge-probe population).
- Additive per-node model; fiber floor v3.2 × 1.3.
- **Geolocation only.** The positive "auditing pays" case likely lives in the
  routing-inference / fault-localization case studies, where the model is
  non-poolable and structurally uncertain — not yet built here.
