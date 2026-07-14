"""
Animated geolocation of one IP: watch the feasible region (overlap of
per-source rings) tighten as vantage points are added, next to the min-RTT /
nearest-neighbor estimate, with localization error as a function of the number
of sources.

This is the IDEALIZED mechanism: a correctly calibrated latency model (we know
each source's overhead), so the rings are physically sized and trilateration
works as intended — the feasible region collapses onto the target and the
polygon/MAP estimate drops BELOW the min-RTT baseline. It is the constructive
complement to the real-data figures (audit_vs_task_real / _fiber), where a
misspecified model inflates the rings and NN wins instead. Here you see WHY the
polygon method is powerful when the model is trustworthy.

Left panel  : map — sources (added / newest), their rings, the overlap heatmap
              (how many rings cover each point = the feasible polygon), the true
              IP (star), the min-RTT/NN estimate, the MAP/polygon estimate.
Right panel : localization error (km) vs number of sources, for NN and MAP,
              plus the current min-RTT readout.

Output: figures/geoloc_animation.gif
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
import math
import numpy as np

from probabilistic_helpers import additive_map_location, haversine_grid
from utils import get_distance

KM_PER_MS = 100.0
RADIUS_MULT = 1.06
GRID = 240
NOISE_MS = 1.0


def destination(lat, lon, bearing_deg, dist_km):
    R = 6371.0
    br = math.radians(bearing_deg)
    d = dist_km / R
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la1) * math.cos(d)
                    + math.cos(la1) * math.sin(d) * math.cos(br))
    lo2 = lo1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(la1),
                           math.cos(d) - math.sin(la1) * math.sin(la2))
    return (math.degrees(la2), math.degrees(lo2))


def make_scene(seed=7, n=25):
    """A target with n vantage points around it at known overheads. The model
    is calibrated: we know each source's overhead a[s], so ring radii are
    physical distances. Sources are spread over all bearings at varied
    distances (some near -> a competitive NN baseline)."""
    rng = np.random.default_rng(seed)
    target = (48.5, 8.5)                      # central Europe
    bearings = (np.linspace(0, 360, n, endpoint=False) + rng.uniform(-8, 8, n)) % 360
    dists = rng.uniform(300, 1400, n)
    srcs = {}
    for i in range(n):
        loc = destination(*target, float(bearings[i]), float(dists[i]))
        a = float(rng.uniform(4.0, 22.0))     # per-source overhead (known)
        true_d = get_distance(loc, target)
        rtt = true_d / KM_PER_MS + a + float(np.min(rng.normal(0, NOISE_MS, 3)))
        srcs[f'V{i}'] = dict(loc=loc, a=a, rtt=rtt, dist=true_d)
    return target, srcs


def main():
    target, srcs = make_scene()
    ids = list(srcs)

    def rtt(s): return srcs[s]['rtt']
    def a(s): return srcs[s]['a']
    def loc(s): return srcs[s]['loc']

    # order sources for angular diversity around the target
    def bearing(fr, to):
        la1, lo1 = map(math.radians, fr)
        la2, lo2 = map(math.radians, to)
        dlon = lo2 - lo1
        x = math.sin(dlon) * math.cos(la2)
        y = (math.cos(la1) * math.sin(la2)
             - math.sin(la1) * math.cos(la2) * math.cos(dlon))
        return math.degrees(math.atan2(x, y)) % 360.0
    brg = {s: bearing(target, loc(s)) for s in ids}
    order = [ids[3]]                          # start off to one side
    rest = [s for s in ids if s not in order]
    while rest:
        def gap(s):
            return min(min(abs(brg[s] - brg[p]) % 360,
                           360 - abs(brg[s] - brg[p]) % 360) for p in order)
        nxt = max(rest, key=gap)
        order.append(nxt); rest.remove(nxt)
    K = len(order)

    def radius_km(s):
        return max(60.0, (rtt(s) - a(s)) * KM_PER_MS) * RADIUS_MULT

    lats = [target[0]] + [loc(s)[0] for s in order]
    lons = [target[1]] + [loc(s)[1] for s in order]
    mlat = (min(lats) - 2, max(lats) + 2)
    mlon = (min(lons) - 3, max(lons) + 3)
    gl = np.linspace(*mlat, GRID)
    go = np.linspace(*mlon, GRID)
    LON, LAT = np.meshgrid(go, gl)
    field = {s: haversine_grid(loc(s)[0], loc(s)[1], LAT, LON) for s in order}

    frames = []
    prev = None
    for k in range(1, K + 1):
        picked = order[:k]
        cover = np.zeros_like(LAT)
        for s in picked:
            cover += (field[s] <= radius_km(s)).astype(float)
        nn_src = min(picked, key=rtt)
        nn_est = loc(nn_src)
        rows = [(loc(s), rtt(s), a(s), NOISE_MS ** 2) for s in picked]
        # legitimate starts (no ground truth): NN anchor + a neutral interior
        # point (previous estimate, or the centroid of picked sources at k=1)
        centroid = (float(np.mean([loc(s)[0] for s in picked])),
                    float(np.mean([loc(s)[1] for s in picked])))
        map_est = additive_map_location(rows, [nn_est, prev or centroid])
        prev = map_est
        frames.append(dict(k=k, picked=list(picked), cover=cover, nn_est=nn_est,
                           map_est=map_est, min_rtt=rtt(nn_src),
                           err_nn=get_distance(nn_est, target),
                           err_map=get_distance(map_est, target)))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, (axM, axE) = plt.subplots(1, 2, figsize=(13.5, 6),
                                   gridspec_kw={'width_ratios': [1.5, 1]})
    aspect = 1.0 / math.cos(math.radians(np.mean(mlat)))
    ymax = max(max(f['err_nn'], f['err_map']) for f in frames) * 1.1

    def draw(fi):
        f = frames[fi]
        axM.clear(); axE.clear()
        axM.contourf(LON, LAT, f['cover'], levels=np.arange(0, f['k'] + 2) - 0.5,
                     cmap='YlGnBu', alpha=0.55)
        for s in f['picked']:
            axM.contour(LON, LAT, field[s], levels=[radius_km(s)],
                        colors='#2e86ab', linewidths=0.5, alpha=0.25)
            axM.plot(*loc(s)[::-1], 'o', color='#1b3b5f', ms=3.5, zorder=5)
        axM.plot(*loc(f['picked'][-1])[::-1], 'o', color='#f4a259', ms=9,
                 mec='k', zorder=6, label='newest source')
        axM.plot(*target[::-1], '*', color='#e63946', ms=22, mec='k',
                 zorder=8, label='true IP location')
        axM.plot(*f['nn_est'][::-1], 'P', color='#6a994e', ms=13, mec='k',
                 zorder=7, label=f"NN (min-RTT) — {f['err_nn']:.0f} km")
        axM.plot(*f['map_est'][::-1], 'X', color='#8338ec', ms=13, mec='k',
                 zorder=7, label=f"polygon / MAP — {f['err_map']:.0f} km")
        axM.set_xlim(mlon); axM.set_ylim(mlat); axM.set_aspect(aspect)
        axM.set_xlabel('longitude'); axM.set_ylabel('latitude')
        axM.set_title(f"Geolocating one IP — {f['k']} source(s) picked   "
                      f"(min-RTT {f['min_rtt']:.1f} ms)")
        axM.legend(loc='upper left', fontsize=8, framealpha=0.9)
        axM.grid(alpha=0.2)

        ks = [fr['k'] for fr in frames[:fi + 1]]
        axE.plot(ks, [fr['err_nn'] for fr in frames[:fi + 1]], '-P',
                 color='#6a994e', lw=2, label='NN (min-RTT)')
        axE.plot(ks, [fr['err_map'] for fr in frames[:fi + 1]], '-X',
                 color='#8338ec', lw=2, label='polygon / MAP')
        axE.set_xlim(0.5, K + 0.5); axE.set_ylim(0, ymax)
        axE.set_xlabel('number of sources picked')
        axE.set_ylabel('localization error (km)')
        axE.set_title('Error vs. number of sources')
        axE.legend(frameon=False, fontsize=10)
        axE.grid(alpha=0.25)
        fig.tight_layout()

    seq = list(range(K)) + [K - 1] * 4
    anim = FuncAnimation(fig, draw, frames=seq, interval=600)
    os.makedirs('figures/animations', exist_ok=True)
    out = 'figures/animations/geoloc_animation.gif'
    anim.save(out, writer=PillowWriter(fps=2))
    print(f"wrote {out}  ({K} sources)")
    print("k | min-RTT |  NN err | MAP err")
    for f in frames:
        print(f"{f['k']} | {f['min_rtt']:7.1f} | {f['err_nn']:7.0f} | {f['err_map']:7.0f}")


if __name__ == '__main__':
    main()
