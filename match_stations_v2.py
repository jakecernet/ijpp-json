import json
import numpy as np

IJPP_PATH = "/mnt/user-data/uploads/output.json"
LPP_PATH = "/mnt/user-data/uploads/stations-in-range.json"

DEDUP_RADIUS_M = 20      # LPP: / IJPP: pair within this = same physical stop (mirrored data)
MATCH_RADIUS_M = 150     # max distance to consider an LPP-station <-> IJPP-record match
AMBIGUOUS_GAP_M = 15     # flag for review if best & runner-up are this close in score


def haversine_matrix(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (meters). Returns shape (len1, len2)."""
    R = 6371000.0
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    dlat = lat2r[None, :] - lat1r[:, None]
    dlon = lon2r[None, :] - lon1r[:, None]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1r[:, None]) * np.cos(lat2r[None, :]) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def greedy_match(dist, radius):
    """Greedy nearest-distance matching with mutual exclusivity.
    dist: (n_rows, n_cols) matrix. Returns list of (row_idx, col_idx, distance)."""
    pairs = []
    n_rows, n_cols = dist.shape
    for i in range(n_rows):
        row = dist[i]
        within = np.where(row <= radius)[0]
        for j in within:
            pairs.append((row[j], i, j))
    pairs.sort(key=lambda x: x[0])

    used_rows, used_cols = set(), set()
    matches = []
    for d, i, j in pairs:
        if i in used_rows or j in used_cols:
            continue
        matches.append((i, j, d))
        used_rows.add(i)
        used_cols.add(j)
    return matches


def dedup_ijpp(ijpp):
    """Merge LPP: prefixed entries with their nearby IJPP: prefixed twin
    (same physical stop mirrored into the national feed). Returns a new
    list of records with keys: lat, lon, name, gtfs_id, ijpp_id (optional)."""
    lpp_pref = [s for s in ijpp if s["gtfs_id"].startswith("LPP:")]
    ijpp_pref = [s for s in ijpp if s["gtfs_id"].startswith("IJPP:")]
    other = [s for s in ijpp if not s["gtfs_id"].startswith(("LPP:", "IJPP:"))]

    lat1 = np.array([s["lat"] for s in lpp_pref])
    lon1 = np.array([s["lon"] for s in lpp_pref])
    lat2 = np.array([s["lat"] for s in ijpp_pref])
    lon2 = np.array([s["lon"] for s in ijpp_pref])

    dist = haversine_matrix(lat1, lon1, lat2, lon2)
    pair_matches = greedy_match(dist, DEDUP_RADIUS_M)

    merged_lpp_idx = set()
    merged_ijpp_idx = set()
    combined = []

    for i, j, d in pair_matches:
        lpp_rec = lpp_pref[i]
        ijpp_rec = ijpp_pref[j]
        combined.append({
            "lat": lpp_rec["lat"],
            "lon": lpp_rec["lon"],
            "name": lpp_rec["name"],
            "gtfs_id": lpp_rec["gtfs_id"],
            "ijpp_id": ijpp_rec["gtfs_id"],
        })
        merged_lpp_idx.add(i)
        merged_ijpp_idx.add(j)

    # unpaired LPP: prefixed entries -> standalone (no ijpp_id)
    for i, s in enumerate(lpp_pref):
        if i not in merged_lpp_idx:
            combined.append({
                "lat": s["lat"], "lon": s["lon"], "name": s["name"],
                "gtfs_id": s["gtfs_id"],
            })

    # unpaired IJPP: prefixed entries -> standalone
    for j, s in enumerate(ijpp_pref):
        if j not in merged_ijpp_idx:
            combined.append({
                "lat": s["lat"], "lon": s["lon"], "name": s["name"],
                "gtfs_id": s["gtfs_id"],
            })

    # other agencies, untouched
    for s in other:
        combined.append({
            "lat": s["lat"], "lon": s["lon"], "name": s["name"],
            "gtfs_id": s["gtfs_id"],
        })

    return combined, len(pair_matches)


def main():
    ijpp_raw = json.load(open(IJPP_PATH, encoding="utf-8"))
    lpp = json.load(open(LPP_PATH, encoding="utf-8"))

    combined, n_dedup_pairs = dedup_ijpp(ijpp_raw)
    print(f"Deduped {n_dedup_pairs} LPP:/IJPP: pairs representing the same physical stop "
          f"(within {DEDUP_RADIUS_M}m).")
    print(f"Combined IJPP-side pool size after dedup: {len(combined)} "
          f"(was {len(ijpp_raw)})")

    lpp_lat = np.array([s["latitude"] for s in lpp])
    lpp_lon = np.array([s["longitude"] for s in lpp])
    c_lat = np.array([s["lat"] for s in combined])
    c_lon = np.array([s["lon"] for s in combined])

    dist = haversine_matrix(lpp_lat, lpp_lon, c_lat, c_lon)
    n_lpp, n_c = dist.shape

    # candidate pairs within match radius, for greedy matching + ambiguity check
    pairs = []
    for i in range(n_lpp):
        row = dist[i]
        within = np.where(row <= MATCH_RADIUS_M)[0]
        for j in within:
            pairs.append((row[j], i, j))
    pairs.sort(key=lambda x: x[0])

    candidates_per_lpp = {}
    for d, i, j in pairs:
        candidates_per_lpp.setdefault(i, []).append((d, j))

    used_lpp, used_c = set(), set()
    matches = []
    ambiguous = []

    for d, i, j in pairs:
        if i in used_lpp or j in used_c:
            continue
        matches.append((i, j, d))
        used_lpp.add(i)
        used_c.add(j)

        others = [(dd, jj) for dd, jj in candidates_per_lpp.get(i, [])
                  if jj != j and jj not in used_c]
        if others:
            runner_d, runner_j = min(others, key=lambda x: x[0])
            # Flag only genuinely tight calls: winner isn't a near-exact (<1m)
            # coordinate match, and the runner-up isn't clearly farther away.
            if d > 1.0 and runner_d <= d * 1.5:
                ambiguous.append((i, j, d, runner_j, runner_d))

    output_lpp = []
    for s in lpp:
        rec = dict(s)
        output_lpp.append(rec)

    for i, j, d in matches:
        rec = output_lpp[i]
        c = combined[j]
        rec["gtfs_id"] = c["gtfs_id"]
        if "ijpp_id" in c:
            rec["ijpp_id"] = c["ijpp_id"]

    # For unmatched LPP stations, find the nearest candidate regardless of
    # radius/exclusivity, and report whether it was "stolen" by a closer
    # LPP station, or genuinely absent nearby.
    unmatched_report = []
    for i in range(n_lpp):
        if i in used_lpp:
            continue
        row = dist[i]
        j_nearest = int(np.argmin(row))
        d_nearest = float(row[j_nearest])
        stolen_by = None
        if j_nearest in used_c:
            # find which lpp station claimed it
            for mi, mj, md in matches:
                if mj == j_nearest:
                    stolen_by = {"lpp_name": lpp[mi]["name"], "lpp_ref_id": lpp[mi]["ref_id"],
                                 "distance_m": round(md, 1)}
                    break
        unmatched_report.append({
            "lpp_name": lpp[i]["name"],
            "lpp_ref_id": lpp[i]["ref_id"],
            "nearest_candidate_name": combined[j_nearest]["name"],
            "nearest_candidate_gtfs_id": combined[j_nearest]["gtfs_id"],
            "nearest_candidate_distance_m": round(d_nearest, 1),
            "candidate_claimed_by_other_lpp_station": stolen_by,
        })

    unmatched_report.sort(key=lambda r: r["nearest_candidate_distance_m"])

    # Full unified stop list: every LPP station (matched or not) plus every
    # IJPP-side record (post-dedup) that wasn't claimed by an LPP station.
    # Matched stops appear once (as the LPP record, carrying gtfs_id/ijpp_id).
    full_stops = []
    for rec in output_lpp:
        r = dict(rec)
        full_stops.append(r)

    for j, c in enumerate(combined):
        if j in used_c:
            continue
        r = {
            "latitude": c["lat"],
            "longitude": c["lon"],
            "name": c["name"],
            "gtfs_id": c["gtfs_id"],
        }
        if "ijpp_id" in c:
            r["ijpp_id"] = c["ijpp_id"]
        full_stops.append(r)

    with open("/mnt/user-data/outputs/all-stops-merged.json", "w", encoding="utf-8") as f:
        json.dump(full_stops, f, ensure_ascii=False, indent=2)

    with open("/mnt/user-data/outputs/stations-in-range-matched.json", "w", encoding="utf-8") as f:
        json.dump(output_lpp, f, ensure_ascii=False, indent=2)

    with open("/mnt/user-data/outputs/unmatched-lpp-stations.json", "w", encoding="utf-8") as f:
        json.dump(unmatched_report, f, ensure_ascii=False, indent=2)

    review = []
    for i, j, d, rj, rd in ambiguous:
        review.append({
            "lpp_name": lpp[i]["name"],
            "lpp_ref_id": lpp[i]["ref_id"],
            "matched_name": combined[j]["name"],
            "matched_gtfs_id": combined[j]["gtfs_id"],
            "matched_ijpp_id": combined[j].get("ijpp_id"),
            "matched_distance_m": round(d, 1),
            "runner_up_name": combined[rj]["name"],
            "runner_up_gtfs_id": combined[rj]["gtfs_id"],
            "runner_up_ijpp_id": combined[rj].get("ijpp_id"),
            "runner_up_distance_m": round(rd, 1),
        })
    with open("/mnt/user-data/outputs/ambiguous-matches-review.json", "w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2)

    stolen_count = sum(1 for r in unmatched_report if r["candidate_claimed_by_other_lpp_station"])
    print(f"\nTotal LPP stations: {n_lpp}")
    print(f"Matched: {len(matches)}")
    print(f"Unmatched: {len(unmatched_report)} "
          f"({stolen_count} had a nearby candidate claimed by another LPP station, "
          f"{len(unmatched_report) - stolen_count} had nothing close by)")
    print(f"Flagged ambiguous (tight distance race between two candidates): {len(review)}")
    print(f"\nFull merged stop list: {len(full_stops)} total "
          f"({len(output_lpp)} LPP-side + {len(full_stops) - len(output_lpp)} IJPP-only)")


if __name__ == "__main__":
    main()
