"""
GriddedFiberRtt — a fast drop-in for FiberFloorRtt.

The fiber floor (internet_gmaps FloorEstimator) is accurate but each query is a
graph shortest-path expansion (~0.26 ms), and the MAP/greedy optimizers call it
hundreds of times per target. This precomputes the floor on a lat/lon grid ONCE
(one floor_ms per grid cell, all VPs at once) and then answers base_ms /
base_ms_rows by **bilinear interpolation** — ~microseconds per call, ~1000x
faster — at the cost of a small (grid-resolution) approximation of the smooth
isochrone surface.

Interface matches what the estimators use: base_ms(vp_loc, loc) and
base_ms_rows(vp_locs, loc), returning slope * floor_ms.
"""
from __future__ import annotations

import os
import sys
from glob import glob
import numpy as np

from utils import get_distance

KM_PER_MS = 100.0


class GriddedFiberRtt:
    def __init__(self, estimator, vp_locs, bounds, res_deg=0.25, slope=1.3):
        self.slope = float(slope)
        self.vp_locs = [(float(a), float(b)) for a, b in vp_locs]
        self.vp_idx = {(round(a, 6), round(b, 6)): i
                       for i, (a, b) in enumerate(self.vp_locs)}
        la0, la1, lo0, lo1 = bounds
        self.lat0, self.lon0, self.res = float(la0), float(lo0), float(res_deg)
        self.lats = np.arange(la0, la1 + res_deg, res_deg)
        self.lons = np.arange(lo0, lo1 + res_deg, res_deg)
        nla, nlo, nvp = len(self.lats), len(self.lons), len(self.vp_locs)
        g = np.empty((nla, nlo, nvp), dtype=float)
        for i in range(nla):
            for j in range(nlo):
                f = np.asarray(estimator.floor_ms(float(self.lats[i]),
                                                  float(self.lons[j])), dtype=float)
                g[i, j] = np.where(np.isfinite(f), f, self.BIG)
        self.grid = g

    BIG = 1e9

    def _interp_all(self, lat, lon):
        """Bilinearly interpolated floor to all VPs at (lat, lon) -> (nvp,)."""
        fi = (lat - self.lat0) / self.res
        fj = (lon - self.lon0) / self.res
        i = int(np.clip(np.floor(fi), 0, len(self.lats) - 2))
        j = int(np.clip(np.floor(fj), 0, len(self.lons) - 2))
        fy = float(np.clip(fi - i, 0.0, 1.0))
        fx = float(np.clip(fj - j, 0.0, 1.0))
        g = self.grid
        return (g[i, j] * (1 - fx) * (1 - fy) + g[i, j + 1] * fx * (1 - fy)
                + g[i + 1, j] * (1 - fx) * fy + g[i + 1, j + 1] * fx * fy)

    def base_ms(self, vp_loc, loc):
        v = self.vp_idx[(round(vp_loc[0], 6), round(vp_loc[1], 6))]
        f = self._interp_all(loc[0], loc[1])[v]
        if not np.isfinite(f) or f >= 1e8:   # unreachable cell -> OPEN geodesic fallback
            f = get_distance(vp_loc, loc) / KM_PER_MS
        return self.slope * f

    def base_ms_rows(self, vp_locs, loc):
        allf = self._interp_all(loc[0], loc[1])
        out = []
        for vp in vp_locs:
            f = allf[self.vp_idx[(round(vp[0], 6), round(vp[1], 6))]]
            out.append(self.slope * f if (np.isfinite(f) and f < 1e8)
                       else get_distance(vp, loc) / KM_PER_MS)
        return out


def make_gridded_fiber(mesh, res_deg=0.25, slope=1.3, margin=2.0):
    """Build a GriddedFiberRtt over the mesh's sources (as VPs), covering the
    bounding box of sources+targets plus a margin."""
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'internet_gmaps'))
    from fiber_graph import FiberGraph
    from floor_query import FloorEstimator
    npz = np.load(sorted(glob('internet_gmaps/data/graph_*.npz'))[-1])
    graph = FiberGraph(
        npz['node_lat'], npz['node_lon'], npz['edge_src'], npz['edge_dst'],
        npz['edge_rtt_ms'],
        edge_feature=npz['edge_feature'] if 'edge_feature' in npz else None,
        feature_names=tuple(npz['feature_names']) if 'feature_names' in npz else (),
    )
    srcs = mesh['sources']
    vlat = np.array([mesh['src_loc'][s][0] for s in srcs])
    vlon = np.array([mesh['src_loc'][s][1] for s in srcs])
    est = FloorEstimator(graph, vlat, vlon)
    vp_locs = [mesh['src_loc'][s] for s in srcs]
    lats = [l[0] for l in vp_locs] + [mesh['tgt_loc'][t][0] for t in mesh['targets']]
    lons = [l[1] for l in vp_locs] + [mesh['tgt_loc'][t][1] for t in mesh['targets']]
    bounds = (min(lats) - margin, max(lats) + margin,
              min(lons) - margin, max(lons) + margin)
    return GriddedFiberRtt(est, vp_locs, bounds, res_deg, slope)
