# What this study establishes

The introduction sets out a general program: treat the models that guide
Internet measurement as first-class measurement targets, and choose each
measurement — *task* or *audit* — by its expected value across a hierarchy of
answer-uncertainty and model-uncertainty. **This study is a concrete, empirical
instantiation of that program in a single domain — IP geolocation — on real
RIPE Atlas latency data.** We chose geolocation deliberately as the *hard* case
for the thesis: it is the domain where measurements are cheapest and most
redundant, so if auditing the model pays anywhere it should be easy to show
here — and where it does not, we can say precisely why.

## What we built

We instantiate the task/audit distinction concretely. The *answer* is a
target's location. The *model* is a latency→distance map,

  `rtt(a, b) ≈ base(a, b) + μ_a + μ_b`,

where `base` is a propagation floor — a geodesic speed-of-light term, or a
fiber-infrastructure floor from an external atlas — and the per-node offsets
`μ` capture routing/access overhead. A **task measurement** pings a target from
a vantage point (it informs the location directly). An **audit measurement**
pings between two *known* landmarks, yielding a clean read of the model's
parameters with no location ambiguity. We fit the model self-supervised (no
ground truth enters inference), score against held-out probe locations, and
sweep how a fixed ping budget is divided between task and audit measurements —
across source counts (12–350), target counts (25–~950), coverage regimes, and
both base models.

## What we found

Four findings, each mapping to a claim in the introduction.

**1. The feedback loop is real.** Under a biased base model, model-guided
(greedy) selection is *confidently wrong*: it selects pings that confirm its
current estimate, so its reported uncertainty shrinks while its true error does
not — a wider overconfidence gap than random selection, and sometimes worse
absolute error. This is the introduction's "probe ever closer to the wrong
location," demonstrated on real pings. The pathology is a property of the
*biased model*: with a good base model the same greedy selection is beneficial.

**2. Auditing the latency model almost never pays for geolocation.** Across
25–950 targets, 12–350 sources, dense and sparse coverage, and both bases, the
budget-optimal audit share is ≈ 0%. The reason is structural and, we argue,
general to the task: the *reducible* error is not in the auditable per-node
offsets but in the base model's *structure* (the distance→latency map) — and
that is fixed not by spending measurement budget but by better external data
(swapping the geodesic floor for a fiber-atlas floor cuts error 20–36%).
Meanwhile the offsets an audit can pin are *already determined* by ordinary task
measurements, which pool them across targets. When we deliberately removed that
pooling (few sources, few targets), auditing still did not robustly help:
reducing targets either starves the one-sided geometry or inflates variance
faster than it exposes the model.

**3. Where auditing does pay, and by how little.** One real-data regime meets
the precondition: a small, clustered vantage-point fleet geolocating a *few far
targets*. There the geometry is one-sided (triangulation cannot resolve distance
along the source→target axis) and the binding error is a large, *shared*,
region-specific long-haul overhead (~63–142 ms, with little per-target scatter)
that few targets cannot pool. An audit to a known landmark in the target's
region measures that overhead directly and transfers it to the targets. Under a
biased base this yields a *small but statistically significant* improvement
(~4–11%; +175 to +822 km; 150-trial 95% confidence intervals). Even here a
better base model is 4–10× more effective and needs no audit.

**4. Geolocation is robust to model drift.** Over real snapshots 26 days apart
(mean per-path RTT drift ~6 ms), a model frozen at day 0 does not measurably
degrade — redundant coverage and per-target recalibration absorb the drift. So
the introduction's "models go stale, keep auditing them" motivation, sound in
general, does not bite for anchor-mesh geolocation at this timescale.

## What this means

Read together, these results **discipline** the thesis rather than simply
confirm it. Recursive-BED is correct and useful, but its value is *conditional*,
and we can now state the condition precisely:

> **Auditing pays iff the binding, reducible uncertainty lives in a model
> component that (a) task measurements cannot already determine by pooling, and
> (b) cannot be fixed more cheaply by an external structural prior.**

Geolocation usually violates both, so the framework's honest output is "spend on
the task, and improve the base model from external data — do not audit." This is
a *stronger* validation of an acquisition rule than a manufactured win: a rule
that only ever says "audit" is not deciding anything; ours correctly declines
where auditing does not help and identifies the narrow regime where it does.

We therefore position geolocation as the study's **boundary case** — the domain
that defines when *not* to audit, and why. The introduction's other two
domains — **routing inference** and **performance/fault localization** — are
where we expect the positive case to live, for a structural reason our
geolocation results make concrete: their models are *non-poolable* (a routing
policy cannot be averaged out of target measurements the way per-node latency
offsets can) and *structurally uncertain* (a missing topological dependency
changes the inference, with no external atlas to sidestep it). Extending the
same task/audit machinery to those domains is the natural next step. This study
contributes the framework, the acquisition-rule precondition, the empirical
method — and one methodological caution: these are small effects, and measuring
them reliably requires confidence intervals over many trials, not single runs;
early single-seed results here were misleading noise.

## The lever hierarchy (what actually moves geolocation accuracy)

1. **Base-model structure** (fiber floor ≫ geodesic; ~20–36%) — from external
   infrastructure data, not pings.
2. **Coverage** — nearest-neighbor wins when a vantage point is close; model
   triangulation wins in the sparse / budget-limited regime.
3. **Selection** — model-guided greedy helps only with a good base and enough
   budget; on a biased base it is confidently wrong.
4. **Auditing per-node offsets** — ≈ 0 for geolocation, except the modest
   far-target regime above.

## Scope and limitations

Real-data but bounded: RIPE Atlas anchors (a dense, well-behaved mesh, not the
general edge-host population); a 2-hour measurement snapshot with a 26-day drift
window; an additive latency model over geodesic or fiber-floor bases. The pool
of distinct, ground-truth-located targets is intrinsically limited (~1,600
nodes) because scoring requires known locations; broad edge-target coverage
would need probe-to-probe or external-landmark data. These bounds constrain the
generality of the numbers, not the structural argument.

*(Full experiment-by-experiment results, numbers, and figures: `RESULTS.md`.)*
