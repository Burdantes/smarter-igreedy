"""
Demo 1 — model drift over time (why the model can't be treated as fixed).

The paper's premise: the Internet changes (routes shift), so a latency model
calibrated once goes stale. A system that represents uncertainty in the ANSWER
but treats the MODEL as fixed becomes confidently wrong as drift accumulates.
Auditing (cheap landmark<->landmark pings) detects the drift and repairs the
model.

Real data: three RIPE snapshots 26 days apart (2026-06-12, -06-28, -07-08),
measured drift ~5.9 ms mean |dRTT| at 26 days (28% of pairs shift >5 ms). Fixed
submesh, FIBER base. The shared per-anchor offset model is what goes stale.

Three strategies, evaluated at each snapshot (days since t0 = 0, 16, 26):
  frozen        anchor offsets calibrated at t0, NEVER updated (model = fixed)
  audit-refresh anchor offsets recalibrated at tk from anchor<->anchor pings
  task-refresh  anchor offsets recalibrated at tk from the target pings (self-
                supervised control — stays fresh iff pooling can recover them)

Output: figures/drift.pdf (+ .png) — true error vs time for the three arms.
"""
from __future__ import annotations

import os
import pickle
import numpy as np

import experiment_audit_vs_task_real as E
from gridded_fiber import make_gridded_fiber
from probabilistic_helpers import (fit_additive_params, additive_map_location,
                                    ADDITIVE_PRIOR_MU_MS, ADDITIVE_PRIOR_VAR_MS2,
                                    ADDITIVE_PRIOR_STRENGTH)
from utils import get_distance

KM_PER_MS = 100.0
DATES = ['2026-06-12', '2026-06-28', '2026-07-08']
DAYS = {'2026-06-12': 0, '2026-06-28': 16, '2026-07-08': 26}

# Full anchor mesh, global, split src/dst (same methodology as scale_split).
E.REGION = (-90.0, 90.0, -180.0, 180.0)
E.MIN_SRC_DIST_KM = 0.0
E.N_SOURCES = 350
E.N_TARGETS = 500
E.POOL = 1000
E.MIN_TGT_COV = 20
E.COVERAGE_CAP = 10000


def snap_rtt(meas):
    def f(a, b):
        return meas.get(a, {}).get(b)
    return f


def fit_mu_t_fixed_a(resid, mu_a):
    """Shrunk per-target offset with anchor offsets held FIXED at mu_a."""
    mu_t = {}
    by_t = {}
    for (a, t), rs in resid.items():
        by_t.setdefault(t, []).append((a, rs))
    for t, items in by_t.items():
        num = ADDITIVE_PRIOR_STRENGTH * ADDITIVE_PRIOR_MU_MS
        den = ADDITIVE_PRIOR_STRENGTH
        for a, rs in items:
            for r in rs:
                num += r - mu_a.get(a, ADDITIVE_PRIOR_MU_MS)
                den += 1.0
        mu_t[t] = max(0.0, num / den)
    return mu_t


def localize(mesh, rtt_fn, task_pairs, fiber, fixed_mu_a=None, n_iters=5):
    """Fit target locations from task pings under the fiber base. If
    fixed_mu_a is given, anchor offsets are held fixed (only mu_t + locations
    are fit); else anchor offsets are fit jointly (self-supervised)."""
    src_loc = mesh['src_loc']
    rtts = {}
    for a, t in task_pairs:
        r = rtt_fn(a, t)
        if r is not None:
            rtts.setdefault((a, t), []).append(r)
    # NN init
    nn = {}
    for (a, t), rs in rtts.items():
        r = min(rs)
        if t not in nn or r < nn[t][0]:
            nn[t] = (r, a)
    est = {t: src_loc[a] for t, (_, a) in nn.items()}
    ms = dict(fixed_mu_a) if fixed_mu_a else {}
    mt, vs, vt = {}, {}, {}
    for _ in range(n_iters):
        resid = {(a, t): [r - fiber.base_ms(src_loc[a], est[t]) for r in rs]
                 for (a, t), rs in rtts.items() if t in est}
        if fixed_mu_a is None:
            ms, vs, mt, vt = fit_additive_params(resid)
        else:
            mt = fit_mu_t_fixed_a(resid, ms)
        for t in est:
            rows = [(src_loc[a], r, ms.get(a, ADDITIVE_PRIOR_MU_MS) + mt.get(t, 0),
                     vs.get(a, ADDITIVE_PRIOR_VAR_MS2) + vt.get(t, ADDITIVE_PRIOR_VAR_MS2))
                    for (a, t2), rs in rtts.items() if t2 == t for r in rs]
            if rows:
                est[t] = additive_map_location(rows, [est[t], src_loc[nn[t][1]]],
                                               rtt_model=fiber)
    return est, ms


def fit_anchor_offsets_from_audits(mesh, rtt_fn, audit_pairs, fiber):
    """Recalibrate per-anchor offsets from anchor<->anchor pings (both ends
    known) — a clean read of mu_a with no location ambiguity."""
    src_loc = mesh['src_loc']
    resid = {}
    for a, b in audit_pairs:
        r = rtt_fn(a, b)
        if r is not None:
            resid.setdefault((a, b), []).append(r - fiber.base_ms(src_loc[a], src_loc[b]))
    if not resid:
        return {}
    ms, _, _, _ = fit_additive_params(resid)
    return ms


def mean_err(mesh, est):
    return float(np.mean([get_distance(est[t], mesh['tgt_loc'][t])
                          for t in mesh['targets'] if t in est]))


def main():
    mesh = E.load_submesh()                 # anchors/targets from the 07-08 canonical mesh
    fiber = make_gridded_fiber(mesh, res_deg=0.5, slope=1.3)   # global grid
    snaps = {d: pickle.load(open(f'cache/real_mini_mesh_{d}.pkl', 'rb'))['loc_loc_meas']
             for d in DATES}
    task_pairs = mesh['task_edges']
    audit_pairs = mesh['audit_edges'][:400]

    # t0 calibration: anchor offsets fit from t0 task pings (self-supervised)
    print("calibrating model at t0 (2026-06-12)...")
    _, mu_a_t0 = localize(mesh, snap_rtt(snaps['2026-06-12']), task_pairs, fiber)

    rows = {'frozen': [], 'audit-refresh': [], 'task-refresh': []}
    for d in DATES:
        rtt = snap_rtt(snaps[d])
        # frozen: use the t0 anchor offsets forever
        est_f, _ = localize(mesh, rtt, task_pairs, fiber, fixed_mu_a=mu_a_t0)
        # audit-refresh: recalibrate anchor offsets from tk audits, then localize
        mu_a_k = fit_anchor_offsets_from_audits(mesh, rtt, audit_pairs, fiber)
        est_a, _ = localize(mesh, rtt, task_pairs, fiber, fixed_mu_a=mu_a_k)
        # task-refresh: recalibrate from tk task pings (self-supervised)
        est_t, _ = localize(mesh, rtt, task_pairs, fiber)
        rows['frozen'].append(mean_err(mesh, est_f))
        rows['audit-refresh'].append(mean_err(mesh, est_a))
        rows['task-refresh'].append(mean_err(mesh, est_t))
        print(f"{d} (day {DAYS[d]:>2}): frozen={rows['frozen'][-1]:.0f}  "
              f"audit={rows['audit-refresh'][-1]:.0f}  task={rows['task-refresh'][-1]:.0f} km")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [DAYS[d] for d in DATES]
    C = {'frozen': '#d1495b', 'audit-refresh': '#2e86ab', 'task-refresh': '#6a994e'}
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for name, ys in rows.items():
        ax.plot(x, ys, '-o', color=C[name], lw=2.3, ms=6, label=name)
    ax.set_xlabel('days since model calibration (t0 = 2026-06-12)')
    ax.set_ylabel('mean localization error (km)')
    ax.set_title('Geolocation is robust to 26-day latency drift (fiber base):\n'
                 'a frozen model does NOT degrade — auditing is unneeded here')
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = 'figures/drift.pdf'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
