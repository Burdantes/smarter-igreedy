"""
Demo 2 — the model-guided feedback loop ("confidently wrong").

The paper's opening danger: when the MODEL decides which evidence to collect
(model-guided/greedy selection) and that evidence is then used to refine the
answer, a biased model makes the system "probe ever closer to the wrong
location" while its reported uncertainty shrinks. Our earlier experiments used
RANDOM selection, which cannot create this loop. This one does.

Setup: real RIPE mesh, FIBER base (gridded fiber floor). Even with a good base,
model-guided greedy SELECTION can confirm its own estimate — it keeps picking
pings the model predicts are consistent, so it plateaus above what diverse
sampling would reach. Audits inject anchor-anchor calibration the greedy would
never choose. Shared additive model on top.

Three strategies, tracked over a shared ping budget:
  random          round-robin anchor selection (no model guidance)
  greedy          model-guided: each ping picks the anchor the current (biased)
                  model predicts will most shrink that target's feasible region
  greedy + audit  greedy, but a fraction of pings are anchor->anchor audits that
                  pin the shared per-anchor offsets

For each we record mean TRUE error and mean REPORTED uncertainty. The loop shows
up as: greedy's REPORTED uncertainty drops below random's while its TRUE error
does NOT — a wider overconfidence gap — and audits closing that gap.

Output: figures/feedback_loop.pdf (+ .png)
"""
from __future__ import annotations

import os
import numpy as np

import experiment_audit_vs_task_real as E
from probabilistic_helpers import (fit_additive_params, additive_map_location,
                                    ADDITIVE_PRIOR_MU_MS, ADDITIVE_PRIOR_VAR_MS2)
from gridded_fiber import make_gridded_fiber
from utils import get_distance

KM_PER_MS = 100.0

E.N_SOURCES = 60
E.N_TARGETS = 15
E.POOL = 400
E.MIN_TGT_COV = 8
E.COVERAGE_CAP = 100

RTT_MODEL = None       # set to the gridded fiber model in main()


def base_ms(a_loc, b_loc):
    if RTT_MODEL is None:
        return get_distance(a_loc, b_loc) / KM_PER_MS
    return RTT_MODEL.base_ms(a_loc, b_loc)


def refit(mesh, rtts, pinned, est):
    """One shared-model refit + relocation of unknown targets. Returns
    (mu_s, var_s, mu_t, var_t)."""
    src_loc = mesh['src_loc']
    resid = {(s, d): [r - base_ms(src_loc[s], est[d]) for r in rs]
             for (s, d), rs in rtts.items() if d in est}
    if not resid:
        return {}, {}, {}, {}
    ms, vs, mt, vt = fit_additive_params(resid)
    # nearest-neighbor anchor per target (lowest-RTT) — a robust MAP start
    nn = {}
    for (s, d), rs in rtts.items():
        if d in pinned:
            continue
        r = min(rs)
        if d not in nn or r < nn[d][0]:
            nn[d] = (r, s)
    for t in [d for d in est if d not in pinned]:
        rows = [(src_loc[s], r, ms.get(s, 0) + mt.get(t, 0),
                 vs.get(s, ADDITIVE_PRIOR_VAR_MS2) + vt.get(t, ADDITIVE_PRIOR_VAR_MS2))
                for (s, d), rs in rtts.items() if d == t for r in rs]
        if len(rows) >= 1:
            starts = [est[t]] + ([src_loc[nn[t][1]]] if t in nn else [])
            est[t] = additive_map_location(rows, starts, rtt_model=RTT_MODEL)
    return ms, vs, mt, vt


def region_km(mesh, t, est, rtts, ms, mt, vs, vt):
    """Reported uncertainty for target t: precision-weighted rms residual +
    statistical floor, km."""
    src_loc = mesh['src_loc']
    nw = nwr = 0.0
    for (s, d), rs in rtts.items():
        if d != t:
            continue
        vsum = vs.get(s, ADDITIVE_PRIOR_VAR_MS2) + vt.get(t, ADDITIVE_PRIOR_VAR_MS2)
        off = ms.get(s, ADDITIVE_PRIOR_MU_MS) + mt.get(t, ADDITIVE_PRIOR_MU_MS)
        base = base_ms(src_loc[s], est[t])
        for r in rs:
            resid = r - base - off
            w = 1.0 / vsum
            nw += w; nwr += w * resid * resid
    if nw <= 0:
        return 20037.0
    return (np.sqrt(nwr / nw) + 1.0 / np.sqrt(nw)) * KM_PER_MS


def greedy_pick(mesh, t, est, rtts, ms, mt, vs, vt, avail):
    """Model-guided choice: the anchor whose PREDICTED reading (RTT from the
    current biased model at the current estimate) most shrinks t's reported
    region. This is the feedback loop — selection optimises the model's belief."""
    src_loc = mesh['src_loc']
    cur = [(s, r) for (s, d), rs in rtts.items() if d == t for r in rs]
    cur_rows = [(src_loc[s], r, ms.get(s, 0) + mt.get(t, 0),
                 vs.get(s, ADDITIVE_PRIOR_VAR_MS2) + vt.get(t, ADDITIVE_PRIOR_VAR_MS2))
                for s, r in cur]
    nn_loc = src_loc[min(cur, key=lambda x: x[1])[0]] if cur else est[t]
    best, best_sz = None, np.inf
    for a in avail:
        pred = base_ms(src_loc[a], est[t]) + ms.get(a, 0) + mt.get(t, 0)  # model prediction
        rows = cur_rows + [(src_loc[a], pred, ms.get(a, 0) + mt.get(t, 0),
                            vs.get(a, ADDITIVE_PRIOR_VAR_MS2) + vt.get(t, ADDITIVE_PRIOR_VAR_MS2))]
        loc = additive_map_location(rows, [est[t], nn_loc], rtt_model=RTT_MODEL)
        # predicted region size if the ping behaves as the model expects
        nw = nwr = 0.0
        for (aa, rr, off, vsum) in rows:
            resid = rr - base_ms(aa, loc) - off
            w = 1.0 / vsum; nw += w; nwr += w * resid * resid
        sz = (np.sqrt(nwr / nw) + 1.0 / np.sqrt(nw)) * KM_PER_MS
        if sz < best_sz:
            best_sz, best = sz, a
    return best


def run(mesh, seed, budget, mode, audit_frac=0.0, refit_every=10):
    rng = np.random.default_rng(seed)
    src_loc = mesh['src_loc']
    tgts = mesh['targets']
    far = {t: [s for (s, t2) in mesh['task_edges'] if t2 == t] for t in tgts}
    rtts, pinned = {}, {}
    est = {}
    pinged = {t: set() for t in tgts}
    ms = mt = vs = vt = {}

    audit_pairs = list(mesh['audit_edges']); rng.shuffle(audit_pairs)
    n_audit = int(round(audit_frac * budget)); ai = 0

    # seed each target with one random ping so it has a location
    for t in tgts:
        a = far[t][rng.integers(len(far[t]))]
        rtts.setdefault((a, t), []).append(mesh['rtt'](a, t)); pinged[t].add(a)
        est[t] = src_loc[a]
    ms, vs, mt, vt = refit(mesh, rtts, pinned, est)

    spent = len(tgts)
    curve = []
    ti = 0
    while spent < budget:
        # audit step?
        if audit_frac > 0 and ai < n_audit and (spent % max(1, int(1/audit_frac)) == 0) and ai < len(audit_pairs):
            a, b = audit_pairs[ai]; ai += 1
            rtts.setdefault((a, b), []).append(mesh['rtt'](a, b))
            pinned[b] = src_loc[b]; est[b] = src_loc[b]
        else:
            t = tgts[ti % len(tgts)]; ti += 1
            avail = [a for a in far[t] if a not in pinged[t]]
            if not avail:
                spent += 1; continue
            if mode == 'random':
                a = avail[rng.integers(len(avail))]
            else:
                a = greedy_pick(mesh, t, est, rtts, ms, mt, vs, vt, avail)
            rtts.setdefault((a, t), []).append(mesh['rtt'](a, t)); pinged[t].add(a)
        spent += 1
        if spent % refit_every == 0:
            ms, vs, mt, vt = refit(mesh, rtts, pinned, est)
            true_err = np.mean([get_distance(est[t], mesh['tgt_loc'][t]) for t in tgts])
            rep = np.mean([region_km(mesh, t, est, rtts, ms, mt, vs, vt) for t in tgts])
            curve.append((spent, float(true_err), float(rep)))
    return curve


def run_arms(mesh, seeds, budget, refit_every):
    arms = {'random': dict(mode='random'),
            'greedy': dict(mode='greedy'),
            'greedy+audit': dict(mode='greedy', audit_frac=0.25)}
    results = {}
    for name, kw in arms.items():
        print(f"  {name}...")
        allc = [run(mesh, s, budget, refit_every=refit_every, **kw) for s in seeds]
        steps = [c[0] for c in allc[0]]
        true = np.mean([[p[1] for p in c] for c in allc], axis=0)
        rep = np.mean([[p[2] for p in c] for c in allc], axis=0)
        results[name] = (steps, true, rep)
        print(f"    final true={true[-1]:.0f} reported={rep[-1]:.0f} (gap {true[-1]-rep[-1]:.0f})")
    return results


def main():
    global RTT_MODEL
    mesh = E.load_submesh()
    fiber = make_gridded_fiber(mesh, res_deg=0.25, slope=1.3)
    seeds = [0, 1]
    budget = 500
    refit_every = 20

    print("=== GEODESIC (biased) base ===")
    RTT_MODEL = None
    geo = run_arms(mesh, seeds, budget, refit_every)
    print("=== FIBER (good) base ===")
    RTT_MODEL = fiber
    fib = run_arms(mesh, seeds, budget, refit_every)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C = {'random': '#6a994e', 'greedy': '#d1495b', 'greedy+audit': '#2e86ab'}
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.6), sharex=True)

    for col, (label, res) in enumerate([('GEODESIC (biased) base', geo),
                                        ('FIBER (good) base', fib)]):
        aT, aG = ax[0][col], ax[1][col]
        for name, (steps, true, rep) in res.items():
            aT.plot(steps, true, '-o', color=C[name], lw=2.2, ms=3, label=f'{name} TRUE')
            aT.plot(steps, rep, '--', color=C[name], lw=1.3, alpha=.8, label=f'{name} REPORTED')
            aG.plot(steps, np.array(true) - np.array(rep), '-o', color=C[name], lw=2.2, ms=3, label=name)
        aT.set_title(label); aT.grid(alpha=0.25); aT.legend(frameon=False, fontsize=7.5, ncol=2)
        aG.axhline(0, color='k', lw=0.6); aG.grid(alpha=0.25); aG.legend(frameon=False, fontsize=8.5)
        aG.set_xlabel('total pings')
        if col == 0:
            aT.set_ylabel('TRUE error (solid) /\nREPORTED unc. (dashed), km')
            aG.set_ylabel('overconfidence gap\nTRUE − REPORTED (km)')

    fig.suptitle('Model-guided feedback loop — the pathology depends on model bias.  '
                 'Biased base: greedy confidently wrong, audit helps.  '
                 'Good base: greedy beneficial, audit not needed.', fontsize=11.5, y=1.0)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = 'figures/feedback_loop.pdf'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
