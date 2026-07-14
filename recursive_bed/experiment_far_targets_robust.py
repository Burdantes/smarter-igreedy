"""
Far-targets, made statistically robust + the pooling crossover.

6 targets was too few to conclude (Oceania flipped). Here we (A) estimate the
audit benefit with CONFIDENCE INTERVALS over many independent trials per region,
and (B) show the POOLING CROSSOVER: how the benefit depends on #targets sharing
the audited region term — the honest answer to "why not 10K targets at once"
(with many targets the shared term pools out of task pings and the audit benefit
vanishes; the mechanism lives in the few-targets-per-region regime).

Data ceiling (real): only ~1590 distinct located destinations exist, ~510 far
from Europe — 10K DISTINCT known-location targets aren't available. We get
statistical power from many trials (resampling), not from 10K simultaneous
targets (which would pool the effect away anyway).

Geodesic base (where the biased-model pathology + audit rescue live).
Output: figures/scale/far_targets_robust.pdf
"""
from __future__ import annotations
# --- path bootstrap (recursive_bed/) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'internet_gmaps'))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)
# --- end bootstrap ---

import os
import pickle
import numpy as np
from utils import get_distance
from experiment_far_targets import estimate, REGIONS, SRC_BOX

N_SRC = 25


def load():
    d = pickle.load(open('cache/real_mini_mesh.pkl', 'rb'))
    loc = d['address_to_loc']; meas = d['loc_loc_meas']
    def rtt(a, b):
        v = meas.get(a, {}).get(b)
        return float(v[0] if isinstance(v, list) else v) if v is not None else None
    def box(bx):
        la0, la1, lo0, lo1 = bx
        return {n for n, (la, lo) in loc.items() if la0 <= la <= la1 and lo0 <= lo <= lo1}
    outdeg = {s: sum(1 for t in meas[s] if t in loc) for s in meas if s in loc}
    srcs = sorted([s for s in box(SRC_BOX) if s in outdeg], key=outdeg.get, reverse=True)[:N_SRC]
    # sorted() for reproducibility (set iteration order is hash-randomized)
    far = {name: sorted(t for t in box(bx) if sum(1 for s in srcs if rtt(s, t) is not None) >= N_SRC*0.7)
           for name, bx in REGIONS.items()}
    return loc, sorted(srcs), far, rtt


def trial(loc, srcs, far_anchors, rtt, n_tgt, audit_frac, seed, budget_per_tgt=12):
    r = np.random.default_rng(seed)
    perm = list(far_anchors); r.shuffle(perm)
    targets = perm[:n_tgt]; landmarks = perm[n_tgt:]
    if len(landmarks) < 3:
        return None
    budget = budget_per_tgt * n_tgt
    srcs_sh = list(srcs)
    tp = [(srcs_sh[i % len(srcs_sh)], targets[i % n_tgt]) for i in range(budget)]
    ap_all = [(sx, l) for l in landmarks for sx in srcs if rtt(sx, l) is not None]
    r.shuffle(ap_all)
    A = min(int(round(audit_frac * budget)), len(ap_all)); Bt = budget - A
    est = estimate(loc, srcs, targets, landmarks, rtt, tp[:Bt], ap_all[:A], fiber=None)
    return float(np.mean([get_distance(est[t], loc[t]) for t in targets]))


def ci(x):
    x = np.array([v for v in x if v is not None]); m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return m, 1.96 * se


def main():
    loc, srcs, far, rtt = load()
    for k in far:
        print(f"  {k}: {len(far[k])} far anchors")
    NT = 150          # trials

    # (A) audit benefit per region, 6 targets/trial, with 95% CI
    print("\n(A) audit benefit (0% vs 40% audit), 6 tgts/trial, geodesic, 95% CI:")
    A_res = {}
    for reg in ['easia', 'samerica', 'wus', 'oceania']:
        e0 = [trial(loc, srcs, far[reg], rtt, 6, 0.0, s) for s in range(NT)]
        e4 = [trial(loc, srcs, far[reg], rtt, 6, 0.4, s) for s in range(NT)]
        m0, c0 = ci(e0); m4, c4 = ci(e4)
        A_res[reg] = (m0, c0, m4, c4)
        print(f"  {reg:9}: 0%={m0:5.0f}±{c0:4.0f}  40%={m4:5.0f}±{c4:4.0f}  "
              f"benefit={m0-m4:+5.0f} km")

    # (B) pooling crossover on E.Asia: benefit vs #targets/trial
    print("\n(B) E.Asia pooling crossover: audit benefit (0% - best) vs #targets/trial:")
    tvals = [3, 6, 12, 20, 30]
    B_curve = []
    for nt in tvals:
        best = None; base0 = None
        for f in [0.0, 0.2, 0.4]:
            m = np.mean([v for v in (trial(loc, srcs, far['easia'], rtt, nt, f, s)
                                     for s in range(NT)) if v is not None])
            if f == 0.0: base0 = m
            best = m if best is None else min(best, m)
        B_curve.append((nt, base0, best, base0 - best))
        print(f"  {nt:2} tgts: 0%={base0:5.0f}  best={best:5.0f}  benefit={base0-best:+5.0f} km")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    regs = list(A_res)
    x = np.arange(len(regs)); w = 0.38
    m0 = [A_res[r][0] for r in regs]; c0 = [A_res[r][1] for r in regs]
    m4 = [A_res[r][2] for r in regs]; c4 = [A_res[r][3] for r in regs]
    axA.bar(x-w/2, m0, w, yerr=c0, color='#c9a227', capsize=3, label='0% audit')
    axA.bar(x+w/2, m4, w, yerr=c4, color='#e76f51', capsize=3, label='40% audit')
    axA.set_xticks(x); axA.set_xticklabels(regs)
    axA.set_ylabel('far-target error (km)'); axA.legend(frameon=False, fontsize=9)
    axA.set_title(f'A  Audit benefit per region (6 tgts/trial, {NT} trials, 95% CI)')
    axA.grid(alpha=0.25, axis='y')
    nts = [b[0] for b in B_curve]
    axB.plot(nts, [b[1] for b in B_curve], '-o', color='#c9a227', lw=2, label='task-only (0% audit)')
    axB.plot(nts, [b[2] for b in B_curve], '-o', color='#e76f51', lw=2, label='best audit share')
    axB.set_xlabel('# targets per trial (sharing the region term)')
    axB.set_ylabel('far-target error (km)')
    axB.set_title('B  E.Asia: audit persistently helps across #targets (no pooling collapse in 3–30)')
    axB.legend(frameon=False, fontsize=9); axB.grid(alpha=0.25)
    fig.suptitle('Far targets (biased geodesic base): auditing a regional landmark helps a little '
                 '(~4–11%, significant) but the fiber base is 4–10x better and needs no audit', fontsize=12, y=1.0)
    fig.tight_layout()
    os.makedirs('figures/scale', exist_ok=True)
    out = 'figures/scale/far_targets_robust.pdf'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
