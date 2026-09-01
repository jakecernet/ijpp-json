import json
import math
from copy import deepcopy

LPP_FILE = "lpp_stops.json"
IJPP_FILE = "ijpp_stops.json"
OUT_FILE = "unified_stops.json"

MATCH_DISTANCE_M = 10
NAME_DISTANCE_M = 20


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def norm_name(name):
    return name.lower().strip()


with open(LPP_FILE, encoding="utf-8") as f:
    lpp = json.load(f)

with open(IJPP_FILE, encoding="utf-8") as f:
    ijpp = json.load(f)

used_ijpp = set()
result = deepcopy(lpp)

for lpp_stop in result:
    best = None
    best_dist = float("inf")

    for ijpp_stop in ijpp:
        if ijpp_stop["gtfs_id"] in used_ijpp:
            continue

        d = haversine(
            lpp_stop["latitude"],
            lpp_stop["longitude"],
            ijpp_stop["lat"],
            ijpp_stop["lon"],
        )

        if d < best_dist:
            best = ijpp_stop
            best_dist = d

    if best is None:
        continue

    name_match = norm_name(lpp_stop["name"]) == norm_name(best["name"])

    if best_dist <= MATCH_DISTANCE_M or (
        best_dist <= NAME_DISTANCE_M and name_match
    ):
        lpp_stop["gtfs_id"] = best["gtfs_id"]
        used_ijpp.add(best["gtfs_id"])


# add remaining IJPP stops
for ijpp_stop in ijpp:
    if ijpp_stop["gtfs_id"] in used_ijpp:
        continue

    result.append(
        {
            "latitude": ijpp_stop["lat"],
            "longitude": ijpp_stop["lon"],
            "name": ijpp_stop["name"],
            "gtfs_id": ijpp_stop["gtfs_id"],
        }
    )


with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"✅ Done. Unified file written to {OUT_FILE}")
