"""
Audit value at scale, ALL categories: 200 real target IPs, geodesic AND fiber
base models, task-only vs audit+task, plus the NN baseline.

Combines experiment_audit_value_200 (scale) with experiment_audit_vs_task_fiber
(fiber base) so every run compares the full matrix:
  NN (min-RTT, no model)
  task-only  additive, geodesic base   |  audit+task additive, geodesic base
  task-only  additive, fiber base      |  audit+task additive, fiber base

Panels:
  A  error vs budget — all five categories
  B  allocation sweep — optimal audit share, geodesic vs fiber
  C  "confidently wrong" under the fiber base (true vs reported uncertainty)

Output: figures/audit_vs_task_all_n200.pdf (+ .png)
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'internet_gmaps'))

import experiment_audit_vs_task_real as E
from experiment_audit_vs_task_fiber import make_fiber

# scale up the target set (shared submesh for every category)
E.N_TARGETS = 200
E.N_SOURCES = 45
E.POOL = 650
E.MIN_TGT_COV = 8


def main():
    mesh = E.load_submesh()
    fiber = make_fiber(mesh)
    fiber._FLOOR_CACHE_MAX = 5_000_000     # keep the floor cache warm all run

    seeds = list(range(4))
    budgets = [200, 400, 700, 1000, 1300, 1600]
    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    alloc_budget = 900

    print("geodesic budget sweep...")
    geo = E.budget_sweep(mesh, seeds, budgets, audit_fraction=0.3, rtt_model=None)
    print("fiber budget sweep...")
    fib = E.budget_sweep(mesh, seeds, budgets, audit_fraction=0.3, rtt_model=fiber)
    print("geodesic allocation sweep...")
    alloc_geo = E.allocation_sweep(mesh, seeds, alloc_budget, fractions, rtt_model=None)
    print("fiber allocation sweep...")
    alloc_fib = E.allocation_sweep(mesh, seeds, alloc_budget, fractions, rtt_model=fiber)

    print("\nbudget |  NN  | task_geo audit_geo | task_fib audit_fib")
    for i, B in enumerate(budgets):
        print(f"{B:6d} | {geo['nn']['err'][i]:4.0f} | {geo['task']['err'][i]:8.0f} "
              f"{geo['audit']['err'][i]:9.0f} | {fib['task']['err'][i]:8.0f} "
              f"{fib['audit']['err'][i]:9.0f}")
    print("geodesic alloc (audit%->km):",
          {int(f*100): round(float(e)) for f, e in zip(fractions, alloc_geo)})
    print("fiber    alloc (audit%->km):",
          {int(f*100): round(float(e)) for f, e in zip(fractions, alloc_fib)})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C_NN, C_TG, C_AG, C_TF, C_AF = '#6a994e', '#c9a227', '#e9c46a', '#d1495b', '#2e86ab'
    b = np.array(budgets)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.7))

    # A: all categories
    axA.plot(b, geo['nn']['err'], '--^', color=C_NN, lw=2, label='random + NN (no model)')
    axA.plot(b, geo['task']['err'], '-o', color=C_TG, lw=2, label='task-only · geodesic')
    axA.plot(b, geo['audit']['err'], '-s', color=C_AG, lw=1.8, label='audit+task · geodesic')
    axA.plot(b, fib['task']['err'], '-o', color=C_TF, lw=2, label='task-only · FIBER')
    axA.plot(b, fib['audit']['err'], '-s', color=C_AF, lw=1.8, label='audit+task · FIBER')
    axA.set_xlabel('total budget (pings)'); axA.set_ylabel('mean localization error (km)')
    axA.set_title('A  n=200: all categories'); axA.legend(frameon=False, fontsize=8)
    axA.grid(alpha=0.25)

    # B: allocation, geodesic vs fiber
    fp = np.array(fractions) * 100
    axB.plot(fp, alloc_geo, '-o', color=C_TG, lw=2, label='geodesic base')
    axB.plot(fp, alloc_fib, '-o', color=C_AF, lw=2, label='fiber base')
    for arr, col in ((alloc_geo, C_TG), (alloc_fib, C_AF)):
        axB.axvline(fractions[int(np.nanargmin(arr))] * 100, color=col, ls=':', alpha=0.7)
    axB.set_xlabel('share of budget spent on AUDIT (%)')
    axB.set_ylabel('mean localization error (km)')
    axB.set_title(f'B  Optimal audit share (budget={alloc_budget})')
    axB.legend(frameon=False, fontsize=9); axB.grid(alpha=0.25)

    # C: confidently-wrong under fiber
    axC.fill_between(b, fib['task']['rep'], fib['task']['err'], color=C_TF,
                     alpha=0.12, label='task-only gap (fiber)')
    axC.plot(b, fib['task']['err'], '-o', color=C_TF, lw=2, label='task-only fiber: TRUE')
    axC.plot(b, fib['task']['rep'], '--o', color=C_TF, lw=1.6, mfc='white',
             label='task-only fiber: REPORTED')
    axC.plot(b, fib['audit']['err'], '-s', color=C_AF, lw=2, label='audit+task fiber: TRUE')
    axC.plot(b, fib['audit']['rep'], '--s', color=C_AF, lw=1.6, mfc='white',
             label='audit+task fiber: REPORTED')
    axC.set_xlabel('total budget (pings)'); axC.set_ylabel('km')
    axC.set_title('C  "Confidently wrong" (fiber base)')
    axC.legend(frameon=False, fontsize=8); axC.grid(alpha=0.25)

    fig.suptitle('Audit value at scale — 200 real IPs, all categories (geodesic & fiber)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = 'figures/audit_vs_task_all_n200.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
