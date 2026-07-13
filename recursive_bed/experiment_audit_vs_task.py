"""
Audit vs. task measurement: a minimal demonstration of recursive Bayesian
experimental design in the geolocation setting.

Premise (from the recursive-BED framing): a model-guided measurement system
can become *self-consistent but wrong*. It represents uncertainty in the
ANSWER (target location) but treats uncertainty in the upstream MODEL (the
latency model relating RTT to distance) as fixed. Every ping it can make is a
TASK measurement (VP -> target). It never spends budget to AUDIT the model.

This script builds a controlled world where the dominant error lives in the
model — large, heterogeneous, *shared* per-source send-offsets a[s] — and the
per-target coverage is capped (the "observe only a small portion" premise), so
task pings to a target saturate and cannot fix a shared-model bias.

Three strategies compete for the same total budget:

  task-only        every ping is VP->target; the shared additive model is
                   refit from target-estimate residuals (the current system).
  audit+task       a fraction of the budget buys VP->VP pings between KNOWN
                   landmarks, which pin a[s] cleanly; the rest are tasks.
  oracle-model     offsets known exactly (a well-validated model); task pings
                   only. Upper bound / "no model uncertainty" reference.

Both honest estimators reuse the project's real fitting core
(`fit_additive_params` + `additive_map_location`). The ONLY thing the audit
arm adds is the capability the codebase currently lacks: a destination whose
location is known (a landmark), turning its residual into a clean, direct
observation of the source offset.

Output: figures/audit_vs_task.pdf  (three panels)
  A  mean localization error vs total budget  (task-only plateaus; audit+task
     breaks through after a crossover; oracle is the floor)
  B  the "confidently wrong" diagnostic: reported uncertainty vs true error.
     Task-only's reported uncertainty collapses while its true error stays
     high (the gap = overconfidence). audit+task stays calibrated.
  C  allocation sweep at a fixed budget: mean error vs audit fraction — a
     U-shape with an interior optimum (spend some, not all, on the model).
"""
from __future__ import annotations
# --- path bootstrap (moved into recursive_bed/): make repo-root modules,
# internet_gmaps, and relative data/figure paths resolve regardless of CWD ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'internet_gmaps'))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)
# --- end bootstrap ---

import os
import numpy as np

from probabilistic_helpers import fit_additive_params, additive_map_location
from utils import get_distance

KM_PER_MS = 100.0

# ----------------------------------------------------------------------------
# Synthetic world
# ----------------------------------------------------------------------------

N_SOURCES = 10
N_TARGETS = 12
COVERAGE_CAP = 4          # distinct sources any single target may be pinged by
NOISE_MS = 1.5            # per-sample measurement noise
N_SAMPLES = 3             # min-of-N per ping (mirrors min-RTT practice)


def make_world(seed: int) -> dict:
    """A world of known-location sources and unknown-location targets under
    the two-way additive model  rtt = d/100 + a[src] + b[dst] + noise.

    a[src] (send-offset) is LARGE, heterogeneous and shared across targets —
    the dominant *model* error. b[node] (receive-offset) is small. Sources
    carry both (they act as audit destinations for each other)."""
    rng = np.random.default_rng(seed)

    def rand_locs(n):
        lat = rng.uniform(-55, 65, n)
        lon = rng.uniform(-170, 170, n)
        return [(float(la), float(lo)) for la, lo in zip(lat, lon)]

    src_ids = [f"S{i}" for i in range(N_SOURCES)]
    tgt_ids = [f"T{i}" for i in range(N_TARGETS)]
    src_loc = dict(zip(src_ids, rand_locs(N_SOURCES)))
    tgt_loc = dict(zip(tgt_ids, rand_locs(N_TARGETS)))

    # Send-offsets: mean ~20 ms, exponential spread, two pathological sources.
    a = {s: float(8.0 + rng.exponential(12.0)) for s in src_ids}
    for s in rng.choice(src_ids, size=2, replace=False):
        a[s] = float(rng.uniform(38.0, 45.0))
    # Receive-offsets: small, on every node.
    b = {n: float(rng.uniform(0.0, 3.0)) for n in src_ids + tgt_ids}

    return dict(rng=rng, src_ids=src_ids, tgt_ids=tgt_ids, src_loc=src_loc,
                tgt_loc=tgt_loc, a=a, b=b)


def ping(world, src, dst, dst_loc) -> float:
    """One min-of-N RTT sample from a known source to a point."""
    base = get_distance(world['src_loc'][src], dst_loc) / KM_PER_MS
    mean = base + world['a'][src] + world['b'][dst]
    samples = mean + world['rng'].normal(0.0, NOISE_MS, N_SAMPLES)
    return float(np.min(samples))


# ----------------------------------------------------------------------------
# Measurement ordering (selection is deliberately dumb — this is about the
# model, not the acquisition rule)
# ----------------------------------------------------------------------------

def task_order(world) -> list:
    """Round-robin over targets; each target consumes distinct sources in a
    fixed random order, capped at COVERAGE_CAP. Returns [(src, tgt), ...]."""
    rng = world['rng']
    per_tgt = {t: list(rng.permutation(world['src_ids'])[:COVERAGE_CAP])
               for t in world['tgt_ids']}
    order = []
    for k in range(COVERAGE_CAP):
        for t in world['tgt_ids']:
            order.append((per_tgt[t][k], t))
    return order


def audit_order(world) -> list:
    """Round-robin VP->VP pairs among sources (both endpoints known)."""
    rng = world['rng']
    srcs = world['src_ids']
    dst_perm = {s: list(rng.permutation([x for x in srcs if x != s]))
                for s in srcs}
    order = []
    for k in range(len(srcs) - 1):
        for s in srcs:
            order.append((s, dst_perm[s][k]))
    return order


# ----------------------------------------------------------------------------
# Estimator: the project's fitting core (fit_additive_params +
# additive_map_location), with support for PINNED (known-location) destinations
# — the one capability the audit arm needs and the codebase lacks.
# ----------------------------------------------------------------------------

def batch_em(world, task_pairs, audit_pairs, n_iters=4):
    """Fresh, NN-anchored params-first alternation (mirrors additive_batch_em)
    over task pings (dst unknown) and audit pings (dst a known landmark, pinned
    to truth). Returns (estimates, mu_s, mu_t, var_s, var_t, rtts_by_pair)."""
    src_loc = world['src_loc']
    rtts = {}      # (src, dst_id) -> [rtt]
    for s, t in task_pairs:
        rtts.setdefault((s, t), []).append(ping(world, s, t, world['tgt_loc'][t]))
    for s, sd in audit_pairs:
        rtts.setdefault((s, sd), []).append(ping(world, s, sd, src_loc[sd]))

    # Known locations: sources always; audit destinations are known landmarks.
    pinned = {sd: src_loc[sd] for _, sd in audit_pairs}
    unknown = [t for t in world['tgt_ids']
               if any(dst == t for _, dst in rtts)]

    # NN init for unknown targets (lowest-RTT source seen).
    nn = {}
    for (s, dst), rs in rtts.items():
        if dst in pinned:
            continue
        r = min(rs)
        if dst not in nn or r < nn[dst][0]:
            nn[dst] = (r, s)
    nn_est = {t: src_loc[s] for t, (_, s) in nn.items()}
    est = dict(nn_est)
    est.update(pinned)

    mu_s = mu_t = var_s = var_t = {}
    for _ in range(n_iters):
        residuals = {(s, d): [r - get_distance(src_loc[s], est[d]) / KM_PER_MS
                              for r in rs]
                     for (s, d), rs in rtts.items() if d in est}
        mu_s, var_s, mu_t, var_t = fit_additive_params(residuals)
        for t in unknown:
            rows = [(src_loc[s], r, mu_s[s] + mu_t[t], var_s[s] + var_t[t])
                    for (s, d), rs in rtts.items() if d == t for r in rs]
            if rows:
                est[t] = additive_map_location(rows, [est[t], nn_est[t]])

    return est, mu_s, mu_t, var_s, var_t, rtts


def oracle_estimate(world, task_pairs):
    """Localize each target with the TRUE offsets known (no model uncertainty)."""
    src_loc = world['src_loc']
    rtts = {}
    for s, t in task_pairs:
        rtts.setdefault((s, t), []).append(ping(world, s, t, world['tgt_loc'][t]))
    nn = {}
    for (s, t), rs in rtts.items():
        r = min(rs)
        if t not in nn or r < nn[t][0]:
            nn[t] = (r, s)
    est = {}
    var = NOISE_MS ** 2
    for t in {t for _, t in task_pairs}:
        rows = [(src_loc[s], r, world['a'][s] + world['b'][t], var)
                for (s, tt), rs in rtts.items() if tt == t for r in rs]
        start = src_loc[nn[t][1]]
        est[t] = additive_map_location(rows, [start, start])
    return est


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def mean_error_km(world, est) -> float:
    errs = [get_distance(est[t], world['tgt_loc'][t]) for t in world['tgt_ids']
            if t in est]
    return float(np.mean(errs)) if errs else np.nan


def reported_uncertainty_km(world, est, mu_s, mu_t, var_s, var_t, rtts) -> float:
    """The uncertainty the SYSTEM reports for its answers — precision-aware
    weighted-RMS residual + statistical floor (the additive region-size idea).
    Blind to model bias by construction: a self-consistent wrong fit has clean
    residuals and reports low uncertainty."""
    from probabilistic_helpers import ADDITIVE_PRIOR_MU_MS, ADDITIVE_PRIOR_VAR_MS2
    src_loc = world['src_loc']
    sizes = []
    for t in world['tgt_ids']:
        if t not in est or est[t] is None:
            continue
        num_w = 0.0
        num_wr2 = 0.0
        for (s, d), rs in rtts.items():
            if d != t:
                continue
            vsum = (var_s.get(s, ADDITIVE_PRIOR_VAR_MS2)
                    + var_t.get(t, ADDITIVE_PRIOR_VAR_MS2))
            off = (mu_s.get(s, ADDITIVE_PRIOR_MU_MS)
                   + mu_t.get(t, ADDITIVE_PRIOR_MU_MS))
            base = get_distance(src_loc[s], est[t]) / KM_PER_MS
            for r in rs:
                resid = r - base - off
                w = 1.0 / vsum
                num_w += w
                num_wr2 += w * resid * resid
        if num_w <= 0:
            continue
        wrms = np.sqrt(num_wr2 / num_w)
        floor = 1.0 / np.sqrt(num_w)
        sizes.append((wrms + floor) * KM_PER_MS)
    return float(np.mean(sizes)) if sizes else np.nan


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------

def run_budget_sweep(seeds, budgets, audit_fraction=0.3):
    """For each budget: task-only, audit+task, oracle-model.
    Returns dict of arm -> (err_mean, err_std, reported_mean) arrays."""
    out = {a: {'err': [], 'err_sd': [], 'rep': []}
           for a in ('task', 'audit', 'oracle')}
    for B in budgets:
        acc = {a: {'err': [], 'rep': []} for a in ('task', 'audit', 'oracle')}
        for seed in seeds:
            w = make_world(seed)
            t_order = task_order(w)
            a_order = audit_order(w)

            # task-only: all B pings are tasks (recycle order for redundancy)
            tp = [t_order[i % len(t_order)] for i in range(B)]
            est, ms, mt, vs, vt, rt = batch_em(w, tp, [])
            acc['task']['err'].append(mean_error_km(w, est))
            acc['task']['rep'].append(
                reported_uncertainty_km(w, est, ms, mt, vs, vt, rt))

            # audit+task: split the SAME budget
            w = make_world(seed)  # reset RNG so pings are comparable
            t_order = task_order(w)
            a_order = audit_order(w)
            A = int(round(audit_fraction * B))
            Bt = B - A
            tp = [t_order[i % len(t_order)] for i in range(Bt)] if Bt else []
            ap = [a_order[i % len(a_order)] for i in range(A)] if A else []
            est, ms, mt, vs, vt, rt = batch_em(w, tp, ap)
            acc['audit']['err'].append(mean_error_km(w, est))
            acc['audit']['rep'].append(
                reported_uncertainty_km(w, est, ms, mt, vs, vt, rt))

            # oracle-model: B task pings, true offsets
            w = make_world(seed)
            t_order = task_order(w)
            tp = [t_order[i % len(t_order)] for i in range(B)]
            est = oracle_estimate(w, tp)
            acc['oracle']['err'].append(mean_error_km(w, est))
            acc['oracle']['rep'].append(np.nan)

        for a in ('task', 'audit', 'oracle'):
            out[a]['err'].append(np.nanmean(acc[a]['err']))
            out[a]['err_sd'].append(np.nanstd(acc[a]['err']))
            rep = acc[a]['rep']
            out[a]['rep'].append(np.nanmean(rep)
                                 if np.any(np.isfinite(rep)) else np.nan)
    for a in out:
        for k in out[a]:
            out[a][k] = np.array(out[a][k])
    return out


def run_allocation_sweep(seeds, budget, fractions):
    """At a fixed total budget, sweep the audit fraction 0..max."""
    means = []
    for f in fractions:
        errs = []
        for seed in seeds:
            w = make_world(seed)
            t_order = task_order(w)
            a_order = audit_order(w)
            A = int(round(f * budget))
            Bt = budget - A
            tp = [t_order[i % len(t_order)] for i in range(Bt)] if Bt else []
            ap = [a_order[i % len(a_order)] for i in range(A)] if A else []
            est, *_ = batch_em(w, tp, ap)
            errs.append(mean_error_km(w, est))
        means.append(np.nanmean(errs))
    return np.array(means)


def main():
    seeds = list(range(12))
    budgets = [24, 36, 48, 60, 72, 84, 96, 108, 120, 140, 160, 180]
    print("Running budget sweep...")
    sweep = run_budget_sweep(seeds, budgets, audit_fraction=0.3)

    alloc_budget = 100
    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print("Running allocation sweep...")
    alloc = run_allocation_sweep(seeds, alloc_budget, fractions)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C_TASK, C_AUDIT, C_ORACLE = '#d1495b', '#2e86ab', '#6a994e'
    budgets = np.array(budgets)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # --- Panel A: error vs budget ---
    axA.plot(budgets, sweep['task']['err'], '-o', color=C_TASK, lw=2,
             label='task-only (current system)')
    axA.plot(budgets, sweep['audit']['err'], '-s', color=C_AUDIT, lw=2,
             label='audit + task (30% audit)')
    axA.plot(budgets, sweep['oracle']['err'], '--', color=C_ORACLE, lw=2,
             label='oracle model (offsets known)')
    axA.set_xlabel('total budget (pings)')
    axA.set_ylabel('mean localization error (km)')
    axA.set_title('A  Task pings plateau; audits break through')
    axA.legend(frameon=False, fontsize=9)
    axA.grid(alpha=0.25)
    # mark where task-only stops improving (coverage saturates at CAP x N)
    sat = COVERAGE_CAP * N_TARGETS
    axA.axvline(sat, color=C_TASK, ls=':', alpha=0.6)
    axA.annotate('task coverage\nsaturates', xy=(sat, sweep['task']['err'].min()),
                 xytext=(sat + 6, sweep['task']['err'].min() +
                         0.32 * (sweep['task']['err'].max() -
                                 sweep['task']['err'].min())),
                 fontsize=8.5, color=C_TASK)

    # --- Panel B: confidently wrong ---
    axB.fill_between(budgets, sweep['task']['rep'], sweep['task']['err'],
                     color=C_TASK, alpha=0.12, label='task-only overconfidence gap')
    axB.plot(budgets, sweep['task']['err'], '-o', color=C_TASK, lw=2,
             label='task-only: TRUE error')
    axB.plot(budgets, sweep['task']['rep'], '--o', color=C_TASK, lw=1.6,
             mfc='white', label='task-only: REPORTED uncertainty')
    axB.plot(budgets, sweep['audit']['err'], '-s', color=C_AUDIT, lw=2,
             label='audit+task: TRUE error')
    axB.plot(budgets, sweep['audit']['rep'], '--s', color=C_AUDIT, lw=1.6,
             mfc='white', label='audit+task: REPORTED uncertainty')
    axB.set_xlabel('total budget (pings)')
    axB.set_ylabel('km')
    axB.set_title('B  "Confidently wrong": reported ≪ true (task-only)')
    axB.legend(frameon=False, fontsize=8)
    axB.grid(alpha=0.25)

    # --- Panel C: allocation U-curve ---
    axC.plot(np.array(fractions) * 100, alloc, '-o', color='#8338ec', lw=2)
    fbest = fractions[int(np.nanargmin(alloc))]
    axC.axvline(fbest * 100, color='#8338ec', ls=':', alpha=0.7)
    axC.annotate(f'optimum ≈ {int(fbest*100)}% audit',
                 xy=(fbest * 100, np.nanmin(alloc)),
                 xytext=(fbest * 100 + 8, np.nanmin(alloc) + 0.12 *
                         (np.nanmax(alloc) - np.nanmin(alloc))),
                 fontsize=9, color='#8338ec')
    axC.set_xlabel('share of budget spent on AUDIT (%)')
    axC.set_ylabel('mean localization error (km)')
    axC.set_title(f'C  Optimal split at fixed budget={alloc_budget}')
    axC.grid(alpha=0.25)

    fig.suptitle('Audit vs. task measurement — recursive Bayesian experimental '
                 'design in geolocation', fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = 'figures/audit_vs_task.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")

    # Console summary
    print("\nbudget |  task  | audit  | oracle | task_reported | audit_reported")
    for i, B in enumerate(budgets):
        print(f"{B:6d} | {sweep['task']['err'][i]:6.0f} | "
              f"{sweep['audit']['err'][i]:6.0f} | {sweep['oracle']['err'][i]:6.0f} | "
              f"{sweep['task']['rep'][i]:13.0f} | {sweep['audit']['rep'][i]:6.0f}")


if __name__ == '__main__':
    main()
