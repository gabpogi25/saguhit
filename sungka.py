
import time, pandas as pd
from typing import Dict, Any, List
from shared_components import aggregate_flows_to_result, make_flow_header, compute_I_from_power_kw, compute_I2R_loss_kw, FlowResult, SharedGridData

def run(shared: SharedGridData, df_edges: pd.DataFrame, step_fraction: float = 0.1, max_rounds: int = 100):
    t0 = time.time()
    df = df_edges.copy().reset_index(drop=True)
    if "Energy_kW" not in df.columns:
        df["Energy_kW"] = df.get("Energy_MWh",0.0)*1000.0
    if 'eff_dist_km' not in df.columns:
        df['eff_dist_km'] = df.get('Distance_km', 0.0)

    history: List[Dict[str,Any]] = []
    for r in range(max_rounds):
        improved = False
        g = df.reset_index().groupby("From")
        for _, sub in g:
            if len(sub) < 2: continue
            sub = sub.copy()
            sub["E_km"] = sub["Energy_kW"].fillna(0.0) * sub["eff_dist_km"].fillna(0.0)
            donor = sub.sort_values("E_km", ascending=False).iloc[0]
            receiver = sub.sort_values("E_km", ascending=True).iloc[0]
            d_idx = int(donor["index"]); r_idx = int(receiver["index"])
            donor_e = float(df.loc[d_idx, "Energy_kW"])
            shift = donor_e * step_fraction
            if shift <= 0: continue
            cap = float(df.loc[r_idx].get("line_capacity_kw", df.loc[r_idx].get("Capacity_kW", 1e9)))
            cur = float(df.loc[r_idx, "Energy_kW"])
            actual = min(shift, max(0.0, cap - cur))
            if actual <= 0: continue
            df.loc[d_idx, "Energy_kW"] = donor_e - actual
            df.loc[r_idx, "Energy_kW"] = float(df.loc[r_idx, "Energy_kW"]) + actual
            improved = True
            history.append({"round": r+1, "from_idx": d_idx, "to_idx": r_idx, "shift_kw": float(actual)})
        if not improved:
            break

    flows: List[FlowResult] = []
    for _, row in df.iterrows():
        fr = make_flow_header()
        fr.from_node = row.get("From",""); fr.to_node = row.get("To","")
        fr.power_kw = float(row.get("Energy_kW",0.0)); fr.voltage_kv = float(row.get("voltage_kv",13.8))
        dist = float(row.get("eff_dist_km", row.get("Distance_km",0.0) or 0.0)); Rpkm=float(row.get("resistance_ohm_per_km",0.03))
        I = compute_I_from_power_kw(fr.power_kw, fr.voltage_kv); R=Rpkm*max(dist,0.0)
        fr.current_a = float(I); fr.power_loss_kw = float(compute_I2R_loss_kw(I,R,phases=3))
        denom = fr.power_kw + fr.power_loss_kw
        fr.efficiency_percent = float(100.0 * fr.power_kw / denom) if denom>0 else 100.0
        fr.demand_met_kw = float(fr.power_kw); fr.demand_satisfaction_percent = 100.0
        flows.append(fr)

    res = aggregate_flows_to_result(flows, "Sungka_Heuristic", t0, len(history), shared=shared)
    res.metadata["history"] = history
    return res
