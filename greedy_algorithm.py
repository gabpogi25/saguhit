
import time, pandas as pd
from typing import List
from shared_components import FlowResult, aggregate_flows_to_result, make_flow_header, compute_I_from_power_kw, compute_I2R_loss_kw, SharedGridData

def run(shared: SharedGridData, df_edges: pd.DataFrame):
    t0 = time.time()
    df = df_edges.copy().reset_index(drop=True)
    if "Energy_kW" not in df.columns:
        df["Energy_kW"] = df.get("Energy_MWh",0.0)*1000.0
    if "eff_dist_km" not in df.columns:
        df["eff_dist_km"] = df.get("Distance_km",0.0)

    # Greedy: push energy onto shortest-distance edges first respecting capacity
    df["priority"] = df["eff_dist_km"].fillna(0.0)
    df = df.sort_values("priority").reset_index(drop=True)
    flows: List[FlowResult] = []
    for _, row in df.iterrows():
        fr = make_flow_header()
        fr.from_node = row.get("From",""); fr.to_node = row.get("To","")
        cap = float(row.get("line_capacity_kw", row.get("Capacity_kW", 1e9)))
        p = min(float(row.get("Energy_kW",0.0)), cap)
        fr.power_kw = p; fr.voltage_kv = float(row.get("voltage_kv",13.8))
        dist = float(row.get("eff_dist_km", row.get("Distance_km",0.0) or 0.0)); Rpkm=float(row.get("resistance_ohm_per_km",0.03))
        I = compute_I_from_power_kw(fr.power_kw, fr.voltage_kv); R=Rpkm*max(dist,0.0)
        fr.current_a = float(I); fr.power_loss_kw = float(compute_I2R_loss_kw(I,R,phases=3))
        denom = fr.power_kw + fr.power_loss_kw
        fr.efficiency_percent = float(100.0 * fr.power_kw / denom) if denom>0 else 100.0
        fr.demand_met_kw = float(fr.power_kw); fr.demand_satisfaction_percent = 100.0
        flows.append(fr)
    return aggregate_flows_to_result(flows, "Greedy", t0, len(df), None)
