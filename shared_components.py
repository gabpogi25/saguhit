"""
shared_components.py — shared grid/demand structures, electrical helpers, and result utilities
for the SaGUHiT framework.

This file provides:
- create_sample_grid_structure(): GEN -> TX -> S -> SS -> EP edges (your original content kept)
- create_sample_demand_data(): node demand dict (your original content kept)
- GridEdge dataclass (kept)
- SharedGridData: loader/holder for edges & demand (NEW)
- FlowResult: standardized results container (NEW)
- make_flow_header(): column headers for per-edge flow CSV (NEW)
- compute_I_from_power_kw(): 3-phase RMS current from power & voltage (NEW)
- compute_I2R_loss_kw(): per-edge I²R loss (NEW)
- aggregate_flows_to_result(): summarize flows to metrics (NEW)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional
import math, time, json, os



class SharedGridData:
    def __init__(self, edges, demand):
        self.edges = edges
        self.demand = demand

    @classmethod
    def load_from_csv(cls, filepath: str):
        import pandas as pd
        df = pd.read_csv(filepath)
        
        edges = []
        demand = {}
        for _, row in df.iterrows():
            edges.append({
                "from_node": row["from_node"],
                "to_node": row["to_node"],
                "distance_km": row["distance_km"],
                "from_coordinates": (row["from_x"], row["from_y"]),
                "to_coordinates": (row["to_x"], row["to_y"]),
                "conductor_type": row["conductor_type"],
                "resistance_ohm_per_km": row["resistance_ohm_per_km"],
                "line_capacity_kw": row["line_capacity_kw"],
            })
            demand[row["to_node"]] = row["demand_kw"]

        return cls(edges, demand)

# ------------------------------------------------------------------------------------
# conductor properties (kept)
# ------------------------------------------------------------------------------------
CONDUCTOR_PROPERTIES = {
    "acsr": {"resistance_ohm_per_km": 0.10, "capacity_kw": 20000, "voltage_kv": 69.0},
    "aluminum": {"resistance_ohm_per_km": 0.15, "capacity_kw": 10000, "voltage_kv": 13.8},
}

# ------------------------------------------------------------------------------------
# small helpers (kept)
# ------------------------------------------------------------------------------------
def parse_coordinates(coord_str):
    if isinstance(coord_str, (tuple, list)):
        return float(coord_str[0]), float(coord_str[1])
    s = str(coord_str).strip()
    s = s.strip("()")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) >= 2:
        try:
            return float(parts[0]), float(parts[1])
        except:
            pass
    return 0.0, 0.0

# ------------------------------------------------------------------------------------
# sample grid (kept)
# ------------------------------------------------------------------------------------
def create_sample_grid_structure():
    edges = [
        # Layer 1: Generation to Transmission Lines
        {'from_node':'GEN-1','to_node':'TX-1','distance_km':120.0,'from_coordinates':(0,0),'to_coordinates':(5,12),'conductor_type':'acsr'},
        {'from_node':'GEN-1','to_node':'TX-2','distance_km':95.0,'from_coordinates':(0,0),'to_coordinates':(10,5),'conductor_type':'acsr'},
        {'from_node':'GEN-1','to_node':'TX-3','distance_km':150.0,'from_coordinates':(0,0),'to_coordinates':(3,-15),'conductor_type':'acsr'},
        # Layer 2: Transmission to Primary Substations
        {'from_node':'TX-1','to_node':'S1','distance_km':20.0,'from_coordinates':(5,12),'to_coordinates':(7,14),'conductor_type':'aluminum'},
        {'from_node':'TX-1','to_node':'S2','distance_km':25.0,'from_coordinates':(5,12),'to_coordinates':(3,16),'conductor_type':'aluminum'},
        {'from_node':'TX-2','to_node':'S3','distance_km':15.0,'from_coordinates':(10,5),'to_coordinates':(12,6),'conductor_type':'aluminum'},
        {'from_node':'TX-2','to_node':'S4','distance_km':18.0,'from_coordinates':(10,5),'to_coordinates':(8,3),'conductor_type':'aluminum'},
        {'from_node':'TX-3','to_node':'S5','distance_km':22.0,'from_coordinates':(3,-15),'to_coordinates':(5,-17),'conductor_type':'aluminum'},
        # Layer 3: Primary to Secondary Substations
        {'from_node':'S1','to_node':'SS1','distance_km':5.0,'from_coordinates':(7,14),'to_coordinates':(8,15),'conductor_type':'aluminum'},
        {'from_node':'S1','to_node':'SS2','distance_km':6.0,'from_coordinates':(7,14),'to_coordinates':(6,16),'conductor_type':'aluminum'},
        {'from_node':'S2','to_node':'SS3','distance_km':8.0,'from_coordinates':(3,16),'to_coordinates':(4,18),'conductor_type':'aluminum'},
        {'from_node':'S2','to_node':'SS4','distance_km':10.0,'from_coordinates':(3,16),'to_coordinates':(2,14),'conductor_type':'aluminum'},
    ]

    # Layer 4: SS → 100 EPs
    ss_mapping = {
        'SS1': [1,6,11,16,21,26,31,36,41,46,51,56,61,66,71,76,81,86,91,96],
        'SS2': [2,7,12,17,22,27,32,37,42,47,52,57,62,67,72,77,82,87,92,97],
        'SS3': [3,8,13,18,23,28,33,38,43,48,53,58,63,68,73,78,83,88,93,98],
        'SS4': [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
        'SS5': [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    }

    def ep_coords(ss_coord, idx):
        base_x, base_y = ss_coord
        dx = (idx % 5) - 2
        dy = ((idx // 5) % 4) - 2
        return (base_x + dx*0.5, base_y + dy*0.5)

    ss_coords = {
        'SS1': (8,15),
        'SS2': (6,16),
        'SS3': (4,18),
        'SS4': (2,14),
        'SS5': (5,-17)
    }

    for ss, ep_list in ss_mapping.items():
        for ep in ep_list:
            coord = ep_coords(ss_coords[ss], ep)
            edges.append({
                'from_node': ss,
                'to_node': f'EP{ep}',
                'distance_km': round(0.5 + (ep % 3)*0.3, 3),
                'from_coordinates': ss_coords[ss],
                'to_coordinates': coord,
                'conductor_type': 'aluminum'
            })

    # attach conductor props
    for e in edges:
        props = CONDUCTOR_PROPERTIES.get(
            e.get('conductor_type','aluminum'),
            CONDUCTOR_PROPERTIES['aluminum']
        )
        e['resistance_ohm_per_km'] = props['resistance_ohm_per_km']
        e['line_capacity_kw'] = props['capacity_kw']
        e['voltage_kv'] = props.get('voltage_kv', 13.8)

    return edges

# ------------------------------------------------------------------------------------
# sample demand (kept)
# ------------------------------------------------------------------------------------
def create_sample_demand_data():
    d = {}
    d['GEN-1'] = 0.0
    d['TX-1'] = 15.0; d['TX-2'] = 12.0; d['TX-3'] = 10.0
    d['S1']=25.0; d['S2']=20.0; d['S3']=18.0; d['S4']=16.0; d['S5']=22.0
    d['SS1']=40.0; d['SS2']=35.0; d['SS3']=30.0; d['SS4']=32.0; d['SS5']=28.0

    residential = [1,6,11,16,21,26,31,36,41,46,51,56,61,66,71,76,81,86,91,96]
    for i,ep in enumerate(residential):
        d[f'EP{ep}'] = 25.0 if (i%2==0) else 50.0

    commercial = [2,7,12,17,22,27,32,37,42,47,52,57,62,67,72,77,82,87,92,97]
    for i,ep in enumerate(commercial):
        d[f'EP{ep}'] = 30.0 if (i%2==0) else 55.0

    barangay = [3,8,13,18,23,28,33,38,43,48,53,58,63,68,73,78,83,88,93,98]
    for i,ep in enumerate(barangay):
        d[f'EP{ep}'] = 35.0 if (i%2==0) else 60.0

    rural = [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99]
    for i,ep in enumerate(rural):
        d[f'EP{ep}'] = 40.0 if (i%2==0) else 65.0

    scattered = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    for i,ep in enumerate(scattered):
        d[f'EP{ep}'] = 45.0 if (i%2==0) else 20.0

    return d

# ------------------------------------------------------------------------------------
# dataclasses (kept + new)
# ------------------------------------------------------------------------------------
@dataclass
class GridEdge:
    from_node: str
    to_node: str
    distance_km: float
    resistance_ohm_per_km: float
    line_capacity_kw: float
    from_coordinates: Tuple[float,float]
    to_coordinates: Tuple[float,float]
    conductor_type: str = "aluminum"
    voltage_kv: float = 13.8
    metadata: dict = field(default_factory=dict)

@dataclass
class FlowResult:
    total_power_kw: float
    total_losses_kw: float
    total_distance_km: float
    peak_loading_pct: float
    per_edge: List[Dict[str, Any]]
    per_node: Dict[str, Any]
    elapsed_s: float

# ------------------------------------------------------------------------------------
# electrical helpers (NEW — full equations)
# ------------------------------------------------------------------------------------
def compute_I_from_power_kw(power_kw: float, voltage_kv: float, pf: float = 1.0) -> float:
    """
    3-phase RMS current (A) from real power (kW), line-line voltage (kV), power factor.
    I = P / (sqrt(3) * V * pf)
    Returns amps.
    """
    v_volts = voltage_kv * 1_000
    p_watts = power_kw * 1_000
    denom = math.sqrt(3.0) * v_volts * max(pf, 1e-6)
    return 0.0 if denom <= 0 else p_watts / denom

def compute_I2R_loss_kw(current_a: float, resistance_ohm_per_km: float, distance_km: float) -> float:
    """
    3-phase I²R copper losses along the line section.
    Approximates per-phase R_total = R_per_km * length; 3 phases → multiply by 3.
    P_loss (W) = 3 * I^2 * R_total  → convert to kW.
    """
    r_total = resistance_ohm_per_km * max(distance_km, 0.0)
    loss_w = 3.0 * (current_a ** 2) * r_total
    return loss_w / 1_000.0

# ------------------------------------------------------------------------------------
# results utilities (NEW)
# ------------------------------------------------------------------------------------
def make_flow_header() -> List[str]:
    return [
        "from_node", "to_node", "distance_km",
        "voltage_kv", "resistance_ohm_per_km",
        "line_capacity_kw", "power_kw",
        "current_a", "i2r_loss_kw", "loading_pct"
    ]

def aggregate_flows_to_result(
    edges: List[GridEdge],
    flows: List[Dict[str, Any]],
    per_node: Optional[Dict[str, Any]] = None
) -> FlowResult:
    """
    Combine per-edge flow dicts into summary metrics.
    Each flow dict should include at least:
      {'from_node','to_node','power_kw'}.
    We'll map it to the corresponding GridEdge to compute current, losses, loading.
    """
    t0 = time.time()
    edge_index = {(e.from_node, e.to_node): e for e in edges}

    rows = []
    total_p = 0.0
    total_loss = 0.0
    total_dist = 0.0
    peak_loading = 0.0

    for f in flows:
        key = (f["from_node"], f["to_node"])
        pkw = float(f.get("power_kw", 0.0))
        e = edge_index.get(key)
        if e is None:
            # unknown edge: skip safely
            continue

        I = compute_I_from_power_kw(pkw, e.voltage_kv, pf=f.get("pf", 1.0))
        loss_kw = compute_I2R_loss_kw(I, e.resistance_ohm_per_km, e.distance_km)
        loading = 0.0 if e.line_capacity_kw <= 0 else (abs(pkw) / e.line_capacity_kw) * 100.0

        rows.append({
            "from_node": e.from_node,
            "to_node": e.to_node,
            "distance_km": e.distance_km,
            "voltage_kv": e.voltage_kv,
            "resistance_ohm_per_km": e.resistance_ohm_per_km,
            "line_capacity_kw": e.line_capacity_kw,
            "power_kw": pkw,
            "current_a": I,
            "i2r_loss_kw": loss_kw,
            "loading_pct": loading
        })

        total_p += max(pkw, 0.0)
        total_loss += loss_kw
        total_dist += e.distance_km
        peak_loading = max(peak_loading, loading)

    elapsed = time.time() - t0
    return FlowResult(
        total_power_kw=total_p,
        total_losses_kw=total_loss,
        total_distance_km=total_dist,
        peak_loading_pct=peak_loading,
        per_edge=rows,
        per_node=per_node or {},
        elapsed_s=elapsed
    )

# ------------------------------------------------------------------------------------
# container for edges + demand, with CSV loader (NEW)
# ------------------------------------------------------------------------------------
class SharedGridData:
    """
    Central container the algorithms import.
    Use:
        g = SharedGridData.load("path/to/edges.csv")   # if CSV exists
        # or
        g = SharedGridData.load(None)                  # falls back to sample builders
    CSV schema expected (columns, case-insensitive where sensible):
        from_node,to_node,distance_km,conductor_type,resistance_ohm_per_km,
        line_capacity_kw,voltage_kv,from_coordinates,to_coordinates
    Optional demand CSV is NOT required; if missing, we use create_sample_demand_data().
    If the edges CSV has a column `demand_kw` on any row, we aggregate it by `to_node`.
    """
    def __init__(self, edges: List[GridEdge], demand: Dict[str, float]):
        self.edges: List[GridEdge] = edges
        self.demand: Dict[str, float] = demand
        self.nodes: List[str] = self._collect_nodes()

    @classmethod
    def load(cls, input_path: Optional[str] = None) -> "SharedGridData":
        edges: List[GridEdge]
        demand: Dict[str, float]

        if input_path and os.path.exists(input_path) and input_path.lower().endswith(".csv"):
            import csv
            rows = []
            demand_tmp: Dict[str, float] = {}
            with open(input_path, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    rows.append(row)

                    # if demand_kw is present, collect by node
                    if "demand_kw" in row and row["demand_kw"] not in (None, "", "NA"):
                        try:
                            node = (row.get("to_node") or row.get("node_id") or "").strip()
                            val = float(row["demand_kw"])
                            if node:
                                demand_tmp[node] = demand_tmp.get(node, 0.0) + val
                        except:
                            pass

            edges = []
            for row in rows:
                try:
                    ct = (row.get("conductor_type") or "aluminum").strip().lower()
                    props = CONDUCTOR_PROPERTIES.get(ct, CONDUCTOR_PROPERTIES["aluminum"])

                    edges.append(
                        GridEdge(
                            from_node=row["from_node"].strip(),
                            to_node=row["to_node"].strip(),
                            distance_km=float(row.get("distance_km", 0.0)),
                            resistance_ohm_per_km=float(row.get("resistance_ohm_per_km", props["resistance_ohm_per_km"])),
                            line_capacity_kw=float(row.get("line_capacity_kw", props["capacity_kw"])),
                            from_coordinates=parse_coordinates(row.get("from_coordinates", "(0,0)")),
                            to_coordinates=parse_coordinates(row.get("to_coordinates", "(0,0)")),
                            conductor_type=ct,
                            voltage_kv=float(row.get("voltage_kv", props.get("voltage_kv", 13.8))),
                            metadata={k:v for k,v in row.items()}
                        )
                    )
                except Exception as ex:
                    # skip malformed lines but don't crash
                    continue

            demand = demand_tmp if demand_tmp else create_sample_demand_data()
        else:
            # fall back to built-in synthetic set (your originals)
            _edges = create_sample_grid_structure()
            edges = [
                GridEdge(
                    from_node=e["from_node"],
                    to_node=e["to_node"],
                    distance_km=float(e["distance_km"]),
                    resistance_ohm_per_km=float(e["resistance_ohm_per_km"]),
                    line_capacity_kw=float(e["line_capacity_kw"]),
                    from_coordinates=parse_coordinates(e["from_coordinates"]),
                    to_coordinates=parse_coordinates(e["to_coordinates"]),
                    conductor_type=e.get("conductor_type","aluminum"),
                    voltage_kv=float(e.get("voltage_kv", 13.8)),
                    metadata={}
                )
                for e in _edges
            ]
            demand = create_sample_demand_data()

        return cls(edges, demand)

    def _collect_nodes(self) -> List[str]:
        s = set()
        for e in self.edges:
            s.add(e.from_node); s.add(e.to_node)
        return sorted(s)

    def adjacency(self) -> Dict[str, List[str]]:
        adj: Dict[str, List[str]] = {}
        for e in self.edges:
            adj.setdefault(e.from_node, []).append(e.to_node)
        return adj

    def to_json(self) -> str:
        return json.dumps({
            "edges": [asdict(e) for e in self.edges],
            "demand": self.demand
        }, indent=2)

# ------------------------------------------------------------------------------------
# legacy helper to mirror your original API (kept)
# ------------------------------------------------------------------------------------
def load_grid_and_demand():
    edges_raw = create_sample_grid_structure()
    demand = create_sample_demand_data()
    edges = [
        GridEdge(
            from_node=e["from_node"],
            to_node=e["to_node"],
            distance_km=float(e["distance_km"]),
            resistance_ohm_per_km=float(e["resistance_ohm_per_km"]),
            line_capacity_kw=float(e["line_capacity_kw"]),
            from_coordinates=parse_coordinates(e["from_coordinates"]),
            to_coordinates=parse_coordinates(e["to_coordinates"]),
            conductor_type=e.get("conductor_type","aluminum"),
            voltage_kv=float(e.get("voltage_kv", 13.8)),
            metadata={}
        )
        for e in edges_raw
    ]
    return edges, demand

# ------------------------------------------------------------------------------------
# quick self-test
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    g = SharedGridData.load(None)
    print(f"Loaded {len(g.edges)} edges and {len(g.demand)} demand entries")

    # toy flow: push each EP demand back on its SS edge just to exercise math
    flows = []
    for e in g.edges:
        if e.to_node.startswith("EP"):
            pkw = g.demand.get(e.to_node, 0.0)
            flows.append({"from_node": e.from_node, "to_node": e.to_node, "power_kw": pkw})

    res = aggregate_flows_to_result(g.edges, flows, per_node=g.demand)
    print("Total power (kW):", res.total_power_kw)
    print("Losses (kW):", res.total_losses_kw)
    print("Peak loading (%):", res.peak_loading_pct)
