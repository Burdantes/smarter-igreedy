"""
Minimal RIPE Atlas mesh puller for the audit-vs-task real-data experiment.

Pulls only a handful of hourly ping dumps (each ~2.1 GB, streamed and
immediately aggregated to a tiny summary, raw deleted), then builds a
loc_loc mesh WITHOUT the strict 80%-bidirectional prune — we keep every
node with a known location and let the experiment pick a well-connected
subgraph. Output: cache/real_mini_mesh.pkl  {'address_to_loc', 'loc_loc_meas'}.

Usage:  python pull_minimal_mesh.py 2026-07-08 0 1 2   # date, then hours
"""
import os, sys, glob, json, pickle
from datetime import datetime
import numpy as np
import tqdm

from pull_ripe_atlas_measurement_data import RipeAtlasPipeline
from utils import get_distance, convert_32_to_24


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-07-08"
    hours = [int(h) for h in sys.argv[2:]] or [0, 1]
    print(f"Pulling {date} hours {hours}")

    pipe = RipeAtlasPipeline(start_date=date, end_date=date, max_workers=2)
    d = datetime.strptime(date, "%Y-%m-%d")

    # Download + parse only the requested hours (not all 24).
    for h in hours:
        raw = pipe.download_dump((d, h))
        print(f"  hour {h:02d}: downloaded -> {raw}")
        parsed = pipe.process_dump(raw)
        print(f"  hour {h:02d}: parsed    -> {parsed}")

    # Build the mesh from all parsed summaries, attaching locations and the
    # physical-floor filter (rtt >= geodesic/100). Mirrors load_parsed_target_data
    # lines 211-253 but with NO 10-file cap and NO dense prune.
    address_to_loc = {
        convert_32_to_24(p['address_v4']): (p['latitude'], p['longitude'])
        for p in pipe.probe_metadata.values()
        if p.get('address_v4') and p.get('latitude') is not None
        and p.get('longitude') is not None
    }
    print(f"probes with location: {len(address_to_loc)}")

    meas, floor_cache = {}, {}
    for fn in tqdm.tqdm(glob.glob(os.path.join(pipe.parsed_dir, "*_summary.json")),
                        desc="merging summaries"):
        with open(fn) as f:
            hourly = json.load(f)
        for src, dsts in hourly.items():
            if src not in address_to_loc:
                continue
            for dst, rtts in dsts.items():
                if dst not in address_to_loc:
                    continue
                key = (src, dst)
                if key not in floor_cache:
                    floor_cache[key] = get_distance(address_to_loc[src],
                                                    address_to_loc[dst]) / 100.0
                valid = [r for r in rtts if r >= floor_cache[key]]
                if not valid:
                    continue
                m = float(np.min(valid))
                meas.setdefault(src, {})
                meas[src][dst] = min(m, meas[src][dst]) if dst in meas[src] else m

    nodes = set(meas) | {d for ds in meas.values() for d in ds}
    edges = sum(len(v) for v in meas.values())
    print(f"mesh: {len(meas)} sources, {len(nodes)} nodes, {edges} directed edges")

    os.makedirs("cache", exist_ok=True)
    out = "cache/real_mini_mesh.pkl"
    with open(out, "wb") as f:
        pickle.dump({'address_to_loc': {k: v for k, v in address_to_loc.items()
                                        if k in nodes},
                     'loc_loc_meas': meas}, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
