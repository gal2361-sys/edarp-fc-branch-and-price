"""Parser for the published solution files (Molenbruch et al. 2023).

Format:
    VEHICLE v:
    Lap l[b]: node[time,load,station] node[time,load,station] ...

node < n is the pickup of request node; node >= n is the dropoff of request node-n.
b is the charging duration before that lap. A lap with no nodes and b = 0 is not
performed. For a dropoff node, time is the dropoff time tau.
"""

from __future__ import annotations

import re

_VEHICLE_RE = re.compile(r"VEHICLE\s+(\d+):")
_LAP_RE = re.compile(r"Lap\s+(\d+)\[([\d.]+)\]:(.*)")
_NODE_RE = re.compile(r"(\d+)\[([\d.]+),([\d.]+),(\d+)\]")


def parse_solution(path: str) -> dict[int, list[dict]]:
    """vehicle -> list of laps; a lap is {"l", "b", "nodes": [(node, time, load, station)]}."""
    vehicles: dict[int, list[dict]] = {}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = _VEHICLE_RE.match(line)
            if m:
                cur = int(m.group(1))
                vehicles[cur] = []
                continue
            m = _LAP_RE.match(line)
            if m and cur is not None:
                nodes = [
                    (int(a), float(b), float(c), int(d))
                    for a, b, c, d in _NODE_RE.findall(m.group(3))
                ]
                vehicles[cur].append(
                    {"l": int(m.group(1)), "b": float(m.group(2)), "nodes": nodes}
                )
    return vehicles
