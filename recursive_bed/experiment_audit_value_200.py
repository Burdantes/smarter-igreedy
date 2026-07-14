"""
Audit-value experiment at scale: 200 target IPs (was 25), real RIPE data.

Hardens the "does auditing the additive model pay?" result with an order of
magnitude more targets. Reuses experiment_audit_vs_task_real's submesh
selection + sweeps, only enlarging N_TARGETS (and the candidate POOL) and
scaling the budget grid so per-target coverage is comparable to the n=25 run
(200 targets x 6-ping cap => meaningful budgets run to ~1200 pings).

Output: figures/audit_vs_task_real_n200.pdf (+ .png), same 3 panels as the
n=25 figure: (A) error vs budget, (B) confidently-wrong, (C) allocation sweep
(optimal audit share).
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

import experiment_audit_vs_task_real as E

# scale up the target set
E.N_TARGETS = 200
E.N_SOURCES = 45
E.POOL = 650          # need >= N_SOURCES + N_TARGETS well-connected candidates
E.MIN_TGT_COV = 8
# region / MIN_SRC_DIST_KM / COVERAGE_CAP unchanged from the n=25 run


def main():
    mesh = E.load_submesh()
    seeds = list(range(6))
    # 200 targets x cap 6 = 1200 distinct task pairs; sweep across that range
    budgets = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    print("budget sweep (n=200)...")
    sweep = E.budget_sweep(mesh, seeds, budgets, audit_fraction=0.3)

    alloc_budget = 900
    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print("allocation sweep (n=200)...")
    alloc = E.allocation_sweep(mesh, seeds, alloc_budget, fractions)
    print("allocation (audit% -> mean err km):",
          {int(f * 100): round(float(e)) for f, e in zip(fractions, alloc)})

    print("\nbudget |  task  | audit  |   nn   | task_rep | audit_rep")
    for i, B in enumerate(budgets):
        print(f"{B:6d} | {sweep['task']['err'][i]:6.0f} | {sweep['audit']['err'][i]:6.0f} | "
              f"{sweep['nn']['err'][i]:6.0f} | {sweep['task']['rep'][i]:8.0f} | "
              f"{sweep['audit']['rep'][i]:6.0f}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C_TASK, C_AUDIT, C_NN = '#d1495b', '#2e86ab', '#6a994e'
    b = np.array(budgets)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    axA.plot(b, sweep['task']['err'], '-o', color=C_TASK, lw=2,
             label='task-only additive')
    axA.plot(b, sweep['audit']['err'], '-s', color=C_AUDIT, lw=2,
             label='audit + task (30% audit)')
    axA.plot(b, sweep['nn']['err'], '--^', color=C_NN, lw=2,
             label='random + nearest-neighbor')
    axA.set_xlabel('total budget (pings)')
    axA.set_ylabel('mean localization error (km)')
    axA.set_title(f'A  n=200 targets: error vs budget')
    axA.legend(frameon=False, fontsize=8.5)
    axA.grid(alpha=0.25)

    axB.fill_between(b, sweep['task']['rep'], sweep['task']['err'],
                     color=C_TASK, alpha=0.12, label='task-only overconfidence gap')
    axB.plot(b, sweep['task']['err'], '-o', color=C_TASK, lw=2, label='task-only: TRUE')
    axB.plot(b, sweep['task']['rep'], '--o', color=C_TASK, lw=1.6, mfc='white',
             label='task-only: REPORTED')
    axB.plot(b, sweep['audit']['err'], '-s', color=C_AUDIT, lw=2, label='audit+task: TRUE')
    axB.plot(b, sweep['audit']['rep'], '--s', color=C_AUDIT, lw=1.6, mfc='white',
             label='audit+task: REPORTED')
    axB.set_xlabel('total budget (pings)')
    axB.set_ylabel('km')
    axB.set_title('B  "Confidently wrong"')
    axB.legend(frameon=False, fontsize=8)
    axB.grid(alpha=0.25)

    fbest = fractions[int(np.nanargmin(alloc))]
    axC.plot(np.array(fractions) * 100, alloc, '-o', color='#8338ec', lw=2)
    axC.axvline(fbest * 100, color='#8338ec', ls=':', alpha=0.7)
    axC.set_xlabel('share of budget spent on AUDIT (%)')
    axC.set_ylabel('mean localization error (km)')
    axC.set_title(f'C  Optimal audit share ≈ {int(fbest*100)}%  (budget={alloc_budget})')
    axC.grid(alpha=0.25)

    fig.suptitle('Audit value at scale — 200 real target IPs', fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs('figures/real_audit', exist_ok=True)
    out = 'figures/real_audit/audit_vs_task_real_n200.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
