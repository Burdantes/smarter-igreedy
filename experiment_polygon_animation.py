"""
Animated geolocation on REAL data, three estimator families, geodesic vs fiber.

Adds the CLASSICAL constraint-based polygon (CBG-style) to the picture:
  * Fix propagation at the speed of light -> each RTT gives a circle that is a
    GUARANTEED upper bound on distance, so the truth is ALWAYS inside it.
  * Intersect the circles -> a valid feasible polygon.
  * Estimate = CENTROID of that polygon.
Two geometries for the "circle":
  - GEODESIC : great-circle disk of radius rtt * 100 km (straight-line SOL).
  - FIBER    : the fiber-isochrone region { p : fiber_floor(src, p) <= rtt }
               (follows real cable paths; a tighter valid region).

For reference we also show:
  - NN  : nearest-neighbor / min-RTT (report the lowest-RTT source's location).
  - MAP : the calibrated additive estimator (geodesic and fiber base).

Panels: [geodesic polygon map] [fiber polygon map] [error vs #sources].
Output: figures/polygon_animation.gif
"""
from __future__ import annotations

import os
import sys
import math
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'internet_gmaps'))

from experiment_audit_vs_task_real import load_submesh
from experiment_audit_vs_task_fiber import make_fiber
from probabilistic_helpers import (additive_batch_em, additive_map_location,
                                    haversine_grid, ADDITIVE_PRIOR_VAR_MS2)
from utils import get_distance

KM_PER_MS = 100.0
GRID = 130
K_MAX = 25


def centroid(mask, LAT, LON, fallback):
    if mask.any():
        return (float(LAT[mask].mean()), float(LON[mask].mean()))
    return fallback


def main():
    mesh = load_submesh()
    src_loc, rtt = mesh['src_loc'], mesh['rtt']
    fiber = make_fiber(mesh)
    src_index = {s: i for i, s in enumerate(mesh['sources'])}

    # Calibrate additive offsets once, per base model (for the MAP reference).
    pairs = {(s, t): [rtt(s, t)] for (s, t) in mesh['task_edges']}
    _, mus_g, vs_g, mut_g, vt_g = additive_batch_em(pairs, src_loc, n_iters=6)
    _, mus_f, vs_f, mut_f, vt_f = additive_batch_em(pairs, src_loc, n_iters=6,
                                                    rtt_model=fiber)
    mtg = float(np.median(list(mut_g.values()))); vtg = float(np.median(list(vt_g.values())))
    mtf = float(np.median(list(mut_f.values()))); vtf = float(np.median(list(vt_f.values())))

    # Target with the most sources; order sources by angular diversity.
    cov = {t: [s for (s, t2) in mesh['task_edges'] if t2 == t] for t in mesh['targets']}
    t_star = max(cov, key=lambda t: len(cov[t]))
    true = mesh['tgt_loc'][t_star]
    srcs = cov[t_star]

    def bearing(fr, to):
        la1, lo1 = map(math.radians, fr); la2, lo2 = map(math.radians, to)
        d = lo2 - lo1
        return math.degrees(math.atan2(
            math.sin(d) * math.cos(la2),
            math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(d))) % 360
    brg = {s: bearing(true, src_loc[s]) for s in srcs}
    order = [min(srcs, key=lambda s: rtt(s, t_star))]
    rest = [s for s in srcs if s not in order]
    while rest and len(order) < K_MAX:
        order.append(max(rest, key=lambda s: min(
            min(abs(brg[s] - brg[p]) % 360, 360 - abs(brg[s] - brg[p]) % 360) for p in order)))
        rest.remove(order[-1])
    K = len(order)

    # Grid: must contain the feasible region. The region lies inside the
    # tightest SOL disk (min-RTT source). Union that disk's bbox with the
    # sources and the truth, plus margin.
    s0 = order[0]; r0 = rtt(s0, t_star) * KM_PER_MS
    lat0, lon0 = src_loc[s0]
    dlat = r0 / 111.0; dlon = r0 / (111.0 * math.cos(math.radians(lat0)))
    plats = [lat0 - dlat, lat0 + dlat, true[0]] + [src_loc[s][0] for s in order]
    plons = [lon0 - dlon, lon0 + dlon, true[1]] + [src_loc[s][1] for s in order]
    mlat = (min(plats) - 1, max(plats) + 1)
    mlon = (min(plons) - 1, max(plons) + 1)
    gl = np.linspace(*mlat, GRID); go = np.linspace(*mlon, GRID)
    LON, LAT = np.meshgrid(go, gl)

    print("precomputing geodesic + fiber fields over grid...")
    geo_field = {s: haversine_grid(src_loc[s][0], src_loc[s][1], LAT, LON) for s in order}
    fib_field = {s: np.empty_like(LAT) for s in order}
    for r in range(GRID):
        for c in range(GRID):
            floors = fiber.estimator.floor_ms(LAT[r, c], LON[r, c])
            for s in order:
                f = floors[src_index[s]]
                fib_field[s][r, c] = f if math.isfinite(f) else 1e9

    def sol_radius(s): return rtt(s, t_star) * KM_PER_MS

    frames = []
    prev_g = prev_f = None
    for k in range(1, K + 1):
        picked = order[:k]
        nn_src = min(picked, key=lambda s: rtt(s, t_star)); nn = src_loc[nn_src]
        mask_g = np.ones_like(LAT, bool); mask_f = np.ones_like(LAT, bool)
        for s in picked:
            mask_g &= geo_field[s] <= sol_radius(s)
            mask_f &= fib_field[s] <= rtt(s, t_star)
        cen_g = centroid(mask_g, LAT, LON, prev_g or nn)
        cen_f = centroid(mask_f, LAT, LON, prev_f or nn)
        prev_g, prev_f = cen_g, cen_f
        rows_g = [(src_loc[s], rtt(s, t_star), mus_g.get(s, 0) + mtg,
                   vs_g.get(s, ADDITIVE_PRIOR_VAR_MS2) + vtg) for s in picked]
        rows_f = [(src_loc[s], rtt(s, t_star), mus_f.get(s, 0) + mtf,
                   vs_f.get(s, ADDITIVE_PRIOR_VAR_MS2) + vtf) for s in picked]
        map_g = additive_map_location(rows_g, [nn, cen_g])
        map_f = additive_map_location(rows_f, [nn, cen_f], rtt_model=fiber)
        frames.append(dict(
            k=k, picked=list(picked), mask_g=mask_g.copy(), mask_f=mask_f.copy(),
            nn=nn, cen_g=cen_g, cen_f=cen_f, map_g=map_g, map_f=map_f,
            e_nn=get_distance(nn, true), e_cg=get_distance(cen_g, true),
            e_cf=get_distance(cen_f, true), e_mg=get_distance(map_g, true),
            e_mf=get_distance(map_f, true)))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, (axG, axF, axE) = plt.subplots(1, 3, figsize=(18, 6),
                                        gridspec_kw={'width_ratios': [1, 1, 1]})
    aspect = 1.0 / math.cos(math.radians(np.mean(mlat)))
    ymax = max(max(f['e_nn'], f['e_cg'], f['e_cf'], f['e_mg'], f['e_mf'])
               for f in frames) * 1.08

    def draw_map(ax, f, mask, field, radius_of, centroid_pt, map_pt, title,
                 cen_label, use_rtt_levels=False):
        ax.clear()
        ax.contourf(LON, LAT, mask.astype(float), levels=[0.5, 1.5],
                    colors=['#8ecae6'], alpha=0.55)
        for s in f['picked']:
            lvl = rtt(s, t_star) if use_rtt_levels else radius_of(s)
            ax.contour(LON, LAT, field[s], levels=[lvl], colors='#2e86ab',
                       linewidths=0.5, alpha=0.25)
            ax.plot(*src_loc[s][::-1], 'o', color='#1b3b5f', ms=3.5, zorder=5)
        ax.plot(*src_loc[f['picked'][-1]][::-1], 'o', color='#f4a259', ms=9,
                mec='k', zorder=6)
        ax.plot(*true[::-1], '*', color='#e63946', ms=20, mec='k', zorder=9,
                label='true IP')
        ax.plot(*f['nn'][::-1], 'P', color='#6a994e', ms=11, mec='k', zorder=7,
                label=f"NN {f['e_nn']:.0f} km")
        ax.plot(*centroid_pt[::-1], 'D', color='#d62828', ms=11, mec='k',
                zorder=8, label=cen_label)
        ax.plot(*map_pt[::-1], 'X', color='#8338ec', ms=11, mec='k', zorder=8,
                label='MAP')
        ax.set_xlim(mlon); ax.set_ylim(mlat); ax.set_aspect(aspect)
        ax.set_title(title, fontsize=11)
        ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
        ax.grid(alpha=0.2)

    def draw(fi):
        f = frames[fi]
        draw_map(axG, f, f['mask_g'], geo_field, sol_radius, f['cen_g'], f['map_g'],
                 f"GEODESIC SOL polygon — {f['k']} src   "
                 f"(centroid {f['e_cg']:.0f} km)", f"centroid {f['e_cg']:.0f} km")
        draw_map(axF, f, f['mask_f'], fib_field, None, f['cen_f'], f['map_f'],
                 f"FIBER isochrone polygon — {f['k']} src   "
                 f"(centroid {f['e_cf']:.0f} km)", f"centroid {f['e_cf']:.0f} km",
                 use_rtt_levels=True)
        axE.clear()
        ks = [fr['k'] for fr in frames[:fi + 1]]
        axE.plot(ks, [fr['e_nn'] for fr in frames[:fi + 1]], '-P', color='#6a994e',
                 lw=2, label='NN (min-RTT)')
        axE.plot(ks, [fr['e_cg'] for fr in frames[:fi + 1]], '-D', color='#d62828',
                 lw=2, label='polygon centroid — geodesic')
        axE.plot(ks, [fr['e_cf'] for fr in frames[:fi + 1]], '-D', color='#f77f00',
                 lw=2, label='polygon centroid — fiber')
        axE.plot(ks, [fr['e_mg'] for fr in frames[:fi + 1]], '--X', color='#8338ec',
                 lw=1.5, alpha=0.8, label='MAP — geodesic')
        axE.plot(ks, [fr['e_mf'] for fr in frames[:fi + 1]], '-X', color='#5a189a',
                 lw=1.5, label='MAP — fiber')
        axE.set_xlim(0.5, K + 0.5); axE.set_ylim(0, ymax)
        axE.set_xlabel('number of sources picked')
        axE.set_ylabel('localization error (km)')
        axE.set_title('Error vs. number of sources')
        axE.legend(frameon=False, fontsize=8.5)
        axE.grid(alpha=0.25)
        fig.tight_layout()

    seq = list(range(K)) + [K - 1] * 4
    anim = FuncAnimation(fig, draw, frames=seq, interval=600)
    os.makedirs('figures', exist_ok=True)
    out = 'figures/polygon_animation.gif'
    anim.save(out, writer=PillowWriter(fps=2))
    print(f"wrote {out}  (target {t_star}, {K} sources)")
    print("k |  NN  | cen_geo | cen_fib | MAP_geo | MAP_fib")
    for f in frames:
        print(f"{f['k']} | {f['e_nn']:4.0f} | {f['e_cg']:7.0f} | {f['e_cf']:7.0f} | "
              f"{f['e_mg']:7.0f} | {f['e_mf']:7.0f}")


if __name__ == '__main__':
    main()
