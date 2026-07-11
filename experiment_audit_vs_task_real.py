"""
Audit vs. task measurement on REAL RIPE Atlas latency data.

Same question as experiment_audit_vs_task.py, but every RTT here is a real
measured min-RTT from the anchor mesh (cache/real_mini_mesh.pkl, built by
pull_minimal_mesh.py). No synthetic pings, no known ground-truth model.

Setup
  sources  : well-connected mesh nodes, treated as vantage points (known loc)
  targets  : other nodes whose location is UNKNOWN during inference, scored
             against ground truth only afterward
  task ping : real RTT source -> target        (constrains src + tgt offsets)
  audit ping: real RTT source -> source' (both known landmarks) -> a clean
             observation that pins the shared per-node offsets

Because there is no true latency model on real data, the reference is NOT an
oracle but the project's standard baseline:
  task-only   additive model, task pings only  (the current system)
  audit+task  additive model, part of the budget spent on VP->VP audits
  random+NN   report the lowest-RTT source's location (no model at all)

Per-target coverage is capped (COVERAGE_CAP) — the "observe only a small
portion" premise — so task pings to a target saturate and any residual error
is model error, which only an audit can address.

Output: figures/audit_vs_task_real.pdf (+ .png), 3 panels mirroring the
synthetic version.  Whether audits help on REAL data — and by how much — is
the empirical question this figure answers.
"""
from __future__ import annotations

import os
import pickle
import numpy as np

from probabilistic_helpers import (fit_additive_params, additive_map_location,
                                    ADDITIVE_PRIOR_MU_MS, ADDITIVE_PRIOR_VAR_MS2)
from utils import get_distance

KM_PER_MS = 100.0
MESH_PATH = "cache/real_mini_mesh.pkl"

N_SOURCES = 45          # vantage points (known location)
N_TARGETS = 25          # unknown-location targets, scored post hoc
POOL = 220              # candidate high-degree nodes to select from
COVERAGE_CAP = 6        # distinct sources any one target may be pinged by
MIN_TGT_COV = 8         # a target must be reachable by >= this many far sources

# Constrain to a geographically bounded region so distances stay
# overhead-dominated (the additive model is well-specified over short hops)
# without being coverage-starved. Broad Europe: thousands of anchors.
REGION = (36.0, 60.0, -10.0, 28.0)   # lat0, lat1, lon0, lon1

# Realistic "distributed landmarks": no vantage point sits right next to a
# target, so localization genuinely requires triangulation (and therefore a
# latency model) rather than just reporting the nearest VP. Uses ground-truth
# location for SCENARIO CONSTRUCTION only — never during inference.
MIN_SRC_DIST_KM = 250.0


# ----------------------------------------------------------------------------
# Load real mesh and select a well-connected sub-mesh
# ----------------------------------------------------------------------------

def load_submesh():
    with open(MESH_PATH, "rb") as f:
        d = pickle.load(f)
    loc = d['address_to_loc']
    meas = d['loc_loc_meas']            # src -> dst -> min_rtt (bare float)

    la0, la1, lo0, lo1 = REGION
    loc = {n: (la, lo) for n, (la, lo) in loc.items()
           if la0 <= la <= la1 and lo0 <= lo <= lo1}
    print(f"region nodes with location: {len(loc)}")

    # Restrict to a dense pool of the highest-out-degree nodes with locations.
    out_deg = {s: sum(1 for t in meas[s] if t in loc)
               for s in meas if s in loc}
    pool = sorted(out_deg, key=out_deg.get, reverse=True)[:POOL]
    pool_set = set(pool)

    # In-pool degree (so chosen sources ping each other -> audits exist).
    in_pool_deg = {s: sum(1 for t in meas.get(s, {}) if t in pool_set)
                   for s in pool}
    sources = sorted(pool, key=in_pool_deg.get, reverse=True)[:N_SOURCES]
    src_set = set(sources)

    # A target's usable sources are only those FAR from it (>= MIN_SRC_DIST_KM):
    # the "no co-located landmark" scenario that forces triangulation.
    def far_sources(n):
        return [s for s in sources if n in meas.get(s, {})
                and get_distance(loc[s], loc[n]) >= MIN_SRC_DIST_KM]
    cand = [n for n in pool if n not in src_set and len(far_sources(n)) >= MIN_TGT_COV]
    targets = sorted(cand, key=lambda n: len(far_sources(n)), reverse=True)[:N_TARGETS]

    src_loc = {s: loc[s] for s in sources}
    tgt_loc = {t: loc[t] for t in targets}

    # Available real edges (task edges honor the min-distance construction).
    far = {t: set(far_sources(t)) for t in targets}
    task_edges = [(s, t) for t in targets for s in far[t]]
    audit_edges = [(s, sd) for s in sources for sd in sources
                   if s != sd and sd in meas.get(s, {})]

    def rtt(a, b):
        return float(meas[a][b])

    print(f"submesh: {len(sources)} sources, {len(targets)} targets")
    print(f"task edges: {len(task_edges)}  audit edges: {len(audit_edges)}")
    covs = [sum(1 for s in sources if t in meas.get(s, {})) for t in targets]
    print(f"per-target source coverage: min/median/max "
          f"{min(covs)}/{int(np.median(covs))}/{max(covs)}")
    return dict(sources=sources, targets=targets, src_loc=src_loc,
                tgt_loc=tgt_loc, task_edges=task_edges, audit_edges=audit_edges,
                rtt=rtt)


# ----------------------------------------------------------------------------
# Measurement ordering (dumb selection — this is about the model, not the
# acquisition rule), capped per target.
# ----------------------------------------------------------------------------

def task_order(mesh, seed):
    rng = np.random.default_rng(seed)
    by_tgt = {t: [] for t in mesh['targets']}
    for s, t in mesh['task_edges']:
        by_tgt[t].append(s)
    for t in by_tgt:
        rng.shuffle(by_tgt[t])
        by_tgt[t] = by_tgt[t][:COVERAGE_CAP]
    order = []
    for k in range(COVERAGE_CAP):
        for t in mesh['targets']:
            if k < len(by_tgt[t]):
                order.append((by_tgt[t][k], t))
    return order


def audit_order(mesh, seed):
    rng = np.random.default_rng(seed + 777)
    edges = list(mesh['audit_edges'])
    rng.shuffle(edges)
    return edges


# ----------------------------------------------------------------------------
# Estimator: the project's fitting core, with pinned (known-location) audit
# destinations.  base RTT term = geodesic/100 (offsets absorb overhead).
# ----------------------------------------------------------------------------

def _base_ms(rtt_model, a_loc, b_loc):
    """Base RTT term between two points: geodesic/100 or the injected model."""
    if rtt_model is None:
        return get_distance(a_loc, b_loc) / KM_PER_MS
    return rtt_model.base_ms(a_loc, b_loc)


def estimate(mesh, task_pairs, audit_pairs, n_iters=5, rtt_model=None):
    rtt, src_loc = mesh['rtt'], mesh['src_loc']
    rtts = {}
    for s, t in task_pairs:
        rtts.setdefault((s, t), []).append(rtt(s, t))
    for s, sd in audit_pairs:
        rtts.setdefault((s, sd), []).append(rtt(s, sd))

    pinned = {sd: src_loc[sd] for _, sd in audit_pairs}
    unknown = [t for t in mesh['targets'] if any(d == t for _, d in rtts)]

    nn = {}
    for (s, d), rs in rtts.items():
        if d in pinned:
            continue
        r = min(rs)
        if d not in nn or r < nn[d][0]:
            nn[d] = (r, s)
    nn_est = {t: src_loc[s] for t, (_, s) in nn.items()}
    est = dict(nn_est)
    est.update(pinned)

    ms = mt = vs = vt = {}
    for _ in range(n_iters):
        resid = {(s, d): [r - _base_ms(rtt_model, src_loc[s], est[d]) for r in rs]
                 for (s, d), rs in rtts.items() if d in est}
        ms, vs, mt, vt = fit_additive_params(resid)
        for t in unknown:
            rows = [(src_loc[s], r, ms[s] + mt[t], vs[s] + vt[t])
                    for (s, d), rs in rtts.items() if d == t for r in rs]
            if rows:
                est[t] = additive_map_location(rows, [est[t], nn_est[t]],
                                               rtt_model=rtt_model)
    return est, ms, mt, vs, vt, rtts, nn_est


def nn_estimate(mesh, task_pairs):
    """Baseline: report the lowest-RTT source's location. No model."""
    rtt, src_loc = mesh['rtt'], mesh['src_loc']
    best = {}
    for s, t in task_pairs:
        r = rtt(s, t)
        if t not in best or r < best[t][0]:
            best[t] = (r, s)
    return {t: src_loc[s] for t, (_, s) in best.items()}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def mean_err(mesh, est):
    e = [get_distance(est[t], mesh['tgt_loc'][t]) for t in mesh['targets']
         if t in est]
    return float(np.mean(e)) if e else np.nan


def reported_km(mesh, est, ms, mt, vs, vt, rtts, rtt_model=None):
    src_loc = mesh['src_loc']
    sizes = []
    for t in mesh['targets']:
        if t not in est:
            continue
        nw = nwr = 0.0
        for (s, d), rs in rtts.items():
            if d != t:
                continue
            vsum = vs.get(s, ADDITIVE_PRIOR_VAR_MS2) + vt.get(t, ADDITIVE_PRIOR_VAR_MS2)
            off = ms.get(s, ADDITIVE_PRIOR_MU_MS) + mt.get(t, ADDITIVE_PRIOR_MU_MS)
            base = _base_ms(rtt_model, src_loc[s], est[t])
            for r in rs:
                resid = r - base - off
                w = 1.0 / vsum
                nw += w
                nwr += w * resid * resid
        if nw > 0:
            sizes.append((np.sqrt(nwr / nw) + 1.0 / np.sqrt(nw)) * KM_PER_MS)
    return float(np.mean(sizes)) if sizes else np.nan


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------

def budget_sweep(mesh, seeds, budgets, audit_fraction=0.3, rtt_model=None):
    out = {a: {'err': [], 'rep': []} for a in ('task', 'audit', 'nn')}
    for B in budgets:
        acc = {a: {'err': [], 'rep': []} for a in ('task', 'audit', 'nn')}
        for seed in seeds:
            to = task_order(mesh, seed)
            ao = audit_order(mesh, seed)

            tp = [to[i % len(to)] for i in range(B)]
            est, ms, mt, vs, vt, rt, _ = estimate(mesh, tp, [], rtt_model=rtt_model)
            acc['task']['err'].append(mean_err(mesh, est))
            acc['task']['rep'].append(reported_km(mesh, est, ms, mt, vs, vt, rt, rtt_model))

            A = min(int(round(audit_fraction * B)), len(ao))
            Bt = B - A
            tp = [to[i % len(to)] for i in range(Bt)] if Bt > 0 else []
            ap = ao[:A]
            est, ms, mt, vs, vt, rt, _ = estimate(mesh, tp, ap, rtt_model=rtt_model)
            acc['audit']['err'].append(mean_err(mesh, est))
            acc['audit']['rep'].append(reported_km(mesh, est, ms, mt, vs, vt, rt, rtt_model))

            tp = [to[i % len(to)] for i in range(B)]
            acc['nn']['err'].append(mean_err(mesh, nn_estimate(mesh, tp)))
            acc['nn']['rep'].append(np.nan)
        for a in out:
            out[a]['err'].append(np.nanmean(acc[a]['err']))
            rep = acc[a]['rep']
            out[a]['rep'].append(np.nanmean(rep) if np.any(np.isfinite(rep)) else np.nan)
    for a in out:
        for k in out[a]:
            out[a][k] = np.array(out[a][k])
    return out


def allocation_sweep(mesh, seeds, budget, fractions, rtt_model=None):
    means = []
    for f in fractions:
        errs = []
        for seed in seeds:
            to = task_order(mesh, seed)
            ao = audit_order(mesh, seed)
            A = min(int(round(f * budget)), len(ao))
            Bt = budget - A
            tp = [to[i % len(to)] for i in range(Bt)] if Bt > 0 else []
            est, *_ = estimate(mesh, tp, ao[:A], rtt_model=rtt_model)
            errs.append(mean_err(mesh, est))
        means.append(np.nanmean(errs))
    return np.array(means)


def main():
    mesh = load_submesh()
    seeds = list(range(10))
    budgets = [20, 30, 40, 50, 60, 75, 90, 110, 130, 150, 175]
    print("budget sweep...")
    sweep = budget_sweep(mesh, seeds, budgets, audit_fraction=0.3)

    alloc_budget = 120
    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print("allocation sweep...")
    alloc = allocation_sweep(mesh, seeds, alloc_budget, fractions)
    print("allocation (audit% -> mean err km):",
          {int(f*100): round(float(e)) for f, e in zip(fractions, alloc)})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C_TASK, C_AUDIT, C_NN = '#d1495b', '#2e86ab', '#6a994e'
    b = np.array(budgets)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    axA.plot(b, sweep['task']['err'], '-o', color=C_TASK, lw=2,
             label='task-only additive (current system)')
    axA.plot(b, sweep['audit']['err'], '-s', color=C_AUDIT, lw=2,
             label='audit + task additive (30% audit)')
    axA.plot(b, sweep['nn']['err'], '--^', color=C_NN, lw=2,
             label='random + nearest-neighbor (no model)')
    sat = COVERAGE_CAP * N_TARGETS
    axA.axvline(sat, color=C_TASK, ls=':', alpha=0.6)
    axA.set_xlabel('total budget (pings)')
    axA.set_ylabel('mean localization error (km)')
    axA.set_title('A  NN wins; additive model plateaus above it')
    axA.legend(frameon=False, fontsize=8.5)
    axA.grid(alpha=0.25)

    axB.fill_between(b, sweep['task']['rep'], sweep['task']['err'],
                     color=C_TASK, alpha=0.12, label='task-only overconfidence gap')
    axB.plot(b, sweep['task']['err'], '-o', color=C_TASK, lw=2,
             label='task-only: TRUE error')
    axB.plot(b, sweep['task']['rep'], '--o', color=C_TASK, lw=1.6, mfc='white',
             label='task-only: REPORTED uncertainty')
    axB.plot(b, sweep['audit']['err'], '-s', color=C_AUDIT, lw=2,
             label='audit+task: TRUE error')
    axB.plot(b, sweep['audit']['rep'], '--s', color=C_AUDIT, lw=1.6, mfc='white',
             label='audit+task: REPORTED uncertainty')
    axB.set_xlabel('total budget (pings)')
    axB.set_ylabel('km')
    axB.set_title('B  "Confidently wrong": reports ~700 km, off by more')
    axB.legend(frameon=False, fontsize=8)
    axB.grid(alpha=0.25)

    axC.plot(np.array(fractions) * 100, alloc, '-o', color='#8338ec', lw=2)
    fbest = fractions[int(np.nanargmin(alloc))]
    axC.axvline(fbest * 100, color='#8338ec', ls=':', alpha=0.7)
    axC.annotate(f'min ≈ {int(fbest*100)}% audit',
                 xy=(fbest * 100, np.nanmin(alloc)),
                 xytext=(fbest * 100 + 6, np.nanmin(alloc) +
                         0.12 * (np.nanmax(alloc) - np.nanmin(alloc) + 1)),
                 fontsize=9, color='#8338ec')
    axC.set_xlabel('share of budget spent on AUDIT (%)')
    axC.set_ylabel('mean localization error (km)')
    axC.set_title(f'C  Optimal audit share here ≈ {int(fbest*100)}% (nothing to gain)')
    axC.grid(alpha=0.25)

    fig.suptitle('Audit vs. task on REAL RIPE Atlas data — when auditing THIS '
                 'model does not pay (error is structural, not per-node)',
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = 'figures/audit_vs_task_real.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")

    print("\nbudget |  task  | audit  |   nn   | task_rep | audit_rep")
    for i, B in enumerate(budgets):
        print(f"{B:6d} | {sweep['task']['err'][i]:6.0f} | {sweep['audit']['err'][i]:6.0f} | "
              f"{sweep['nn']['err'][i]:6.0f} | {sweep['task']['rep'][i]:8.0f} | "
              f"{sweep['audit']['rep'][i]:6.0f}")


if __name__ == '__main__':
    main()
