"""
Concrete scenario: 100 anchors, 150 target IPs, 10,000-ping budget, cold start.
Compares 0% auditing vs 10% auditing (plus a few more fractions to show the
trend), geodesic AND fiber base, with the NN baseline. Reports the measurement
split and the resulting localization error (mean AND median km).

Output: figures/scenario_100x150.pdf (+ .png) and a console table.
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
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'internet_gmaps'))

import experiment_audit_vs_task_real as E
from experiment_audit_vs_task_fiber import make_fiber
from utils import get_distance

# scenario config (BUDGET and N_TARGETS overridable via argv: budget [targets])
E.N_SOURCES = 100
E.N_TARGETS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
E.POOL = E.N_SOURCES + E.N_TARGETS + 250     # enough well-connected candidates
E.MIN_TGT_COV = 10
E.COVERAGE_CAP = 100          # effectively uncapped: budget spreads over anchors

BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 10000


def errs(mesh, est):
    return [get_distance(est[t], mesh['tgt_loc'][t]) for t in mesh['targets'] if t in est]


def run_point(mesh, seeds, frac, rtt_model):
    """Return (mean, median, n_audit, n_task, anchors_per_target) at a given
    audit fraction and BUDGET, averaged over seeds."""
    ms, md = [], []
    A = int(round(frac * BUDGET))
    Bt = BUDGET - A
    for seed in seeds:
        to = E.task_order(mesh, seed)
        ao = E.audit_order(mesh, seed)
        a = min(A, len(ao))
        tp = [to[i % len(to)] for i in range(Bt)] if Bt > 0 else []
        est, *_ = E.estimate(mesh, tp, ao[:a], rtt_model=rtt_model)
        e = errs(mesh, est)
        ms.append(np.mean(e)); md.append(np.median(e))
    apt = round(Bt / len(mesh['targets']), 1)
    return float(np.mean(ms)), float(np.mean(md)), A, Bt, apt


def run_nn(mesh, seeds):
    ms, md = [], []
    for seed in seeds:
        to = E.task_order(mesh, seed)
        tp = [to[i % len(to)] for i in range(BUDGET)]
        est = E.nn_estimate(mesh, tp)
        e = errs(mesh, est)
        ms.append(np.mean(e)); md.append(np.median(e))
    return float(np.mean(ms)), float(np.mean(md))


def main():
    mesh = E.load_submesh()
    fiber = make_fiber(mesh)
    fiber._FLOOR_CACHE_MAX = 5_000_000
    seeds = [0, 1, 2, 3]

    nn_mean, nn_med = run_nn(mesh, seeds)

    geo_fracs = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
    fib_fracs = [0.0, 0.05, 0.10, 0.20, 0.30]
    print("running geodesic fractions...")
    geo = {f: run_point(mesh, seeds, f, None) for f in geo_fracs}
    print("running fiber fractions...")
    fib = {f: run_point(mesh, seeds, f, fiber) for f in fib_fracs}

    def line(tag, f, r):
        m, md, A, Bt, apt = r
        print(f"{tag:16} audit={int(f*100):>2}%  audit_pings={A:>4} task_pings={Bt:>5} "
              f"anchors/tgt={apt:>5}  mean={m:6.0f} km  median={md:6.0f} km")
    ntgt = len(mesh['targets'])          # actual selected count (may be < requested)
    apt_nn = round(BUDGET / ntgt, 1)
    print(f"\n=== 100 anchors, {ntgt} targets, budget {BUDGET}, cold start ===")
    print(f"{'NN (min-RTT)':16} audit= 0%  audit_pings=   0 task_pings={BUDGET:>5} "
          f"anchors/tgt={apt_nn:>5}  mean={nn_mean:6.0f} km  median={nn_med:6.0f} km")
    for f in geo_fracs:
        line("GEODESIC", f, geo[f])
    for f in fib_fracs:
        line("FIBER", f, fib[f])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    gx = [f * 100 for f in geo_fracs]
    fx = [f * 100 for f in fib_fracs]
    ax.plot(gx, [geo[f][0] for f in geo_fracs], '-o', color='#c9a227', lw=2, label='geodesic — mean')
    ax.plot(gx, [geo[f][1] for f in geo_fracs], '--o', color='#c9a227', lw=1.5, mfc='white', label='geodesic — median')
    ax.plot(fx, [fib[f][0] for f in fib_fracs], '-s', color='#2e86ab', lw=2, label='fiber — mean')
    ax.plot(fx, [fib[f][1] for f in fib_fracs], '--s', color='#2e86ab', lw=1.5, mfc='white', label='fiber — median')
    ax.axhline(nn_mean, color='#6a994e', ls='-', lw=1.4, alpha=.8, label=f'NN mean ({nn_mean:.0f})')
    ax.axhline(nn_med, color='#6a994e', ls=':', lw=1.4, alpha=.8, label=f'NN median ({nn_med:.0f})')
    ax.axvline(10, color='#888', ls=':', alpha=.6)
    ax.annotate('your 10%', xy=(10, ax.get_ylim()[1]*0.96), fontsize=9, color='#555')
    ax.set_xlabel(f'share of {BUDGET:,}-ping budget spent on AUDIT (%)')
    ax.set_ylabel('localization error (km)')
    ax.set_title(f'100 anchors · {ntgt} targets · {BUDGET:,} pings '
                 f'(~{apt_nn:g} anchors/target) — 0% vs auditing')
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = f'figures/scenario_100x{ntgt}_b{BUDGET}.pdf'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
