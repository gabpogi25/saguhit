
"""
Simplified DC-OPF baseline using cvxpy (linear cost proxy).
If cvxpy is unavailable at runtime, raise a clear error for the user to install it.
"""

import time, pandas as pd, numpy as np
from typing import List, Dict, Set
from shared_components import FlowResult, aggregate_flows_to_result, make_flow_header, compute_I_from_power_kw, compute_I2R_loss_kw, SharedGridData

def run(shared: SharedGridData, df_edges: pd.DataFrame):
    t0 = time.time()
    try:
        import cvxpy as cp
    except Exception as e:
        raise RuntimeError("cvxpy is required for DC-OPF baseline. Install with: pip install cvxpy osqp ecos") from e

    df = df_edges.copy().reset_index(drop=True)
    
    # Ensure required columns exist
    if "Energy_kW" not in df.columns:
        df["Energy_kW"] = df.get("Energy_MWh", 0.0) * 1000.0
    if "eff_dist_km" not in df.columns:
        df["eff_dist_km"] = df.get("Distance_km", 0.0)
    
    # Extract network parameters
    n_lines = len(df)
    line_capacity_kw = df["line_capacity_kw"].fillna(1e9).values
    distance_km = df["eff_dist_km"].fillna(0.0).values
    available_energy_kw = df["Energy_kW"].fillna(0.0).values
    voltage_kv = df["voltage_kv"].fillna(13.8).values
    resistance_ohm_per_km = df["resistance_ohm_per_km"].fillna(0.03).values
    
    # Build network topology
    nodes = set()
    for _, row in df.iterrows():
        nodes.add(row.get("From", ""))
        nodes.add(row.get("To", ""))
    nodes = sorted(list(nodes))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    
    # Extract demand information from shared data or estimate from available energy
    node_demands = {}
    if hasattr(shared, 'node_demands') and shared.node_demands:
        node_demands = shared.node_demands
    else:
        # Estimate demands based on energy availability and network structure
        for _, row in df.iterrows():
            to_node = row.get("To", "")
            if to_node and to_node not in node_demands:
                # Estimate demand as fraction of available energy
                node_demands[to_node] = max(row.get("Energy_kW", 0.0) * 0.8, 100.0)
    
    # Ensure all nodes have demand values
    for node in nodes:
        if node not in node_demands:
            node_demands[node] = 0.0
    
    # Decision variables
    power_flow_kw = cp.Variable(n_lines, nonneg=True)  # Power flow on each line
    node_generation_kw = cp.Variable(n_nodes, nonneg=True)  # Generation at each node
    node_demand_served_kw = cp.Variable(n_nodes, nonneg=True)  # Demand served at each node
    
    # Compute resistance for each line
    line_resistance_ohm = resistance_ohm_per_km * distance_km
    
    # Objective: Minimize total I²R losses
    # For DC approximation: Loss ≈ P²R/V² where P is power flow, R is resistance, V is voltage
    loss_terms = []
    for i in range(n_lines):
        if voltage_kv[i] > 0 and line_resistance_ohm[i] > 0:
            # I²R loss approximation: (P/V)² * R = P² * R / V²
            loss_terms.append(cp.square(power_flow_kw[i]) * line_resistance_ohm[i] / (voltage_kv[i] ** 2))
    
    if loss_terms:
        objective = cp.Minimize(cp.sum(loss_terms))
    else:
        # Fallback: minimize total power flow
        objective = cp.Minimize(cp.sum(power_flow_kw))
    
    # Constraints
    constraints = []
    
    # 1. Line capacity constraints
    constraints.append(power_flow_kw <= line_capacity_kw)
    
    # 2. Generation capacity constraints (based on available energy)
    for i, node in enumerate(nodes):
        # Find available generation capacity for this node
        available_gen = 0.0
        for j, row in df.iterrows():
            if row.get("From", "") == node:
                available_gen += available_energy_kw[j]
        constraints.append(node_generation_kw[i] <= max(available_gen, 0.0))
    
    # 3. Demand constraints - must serve minimum demand
    for i, node in enumerate(nodes):
        min_demand = node_demands.get(node, 0.0) * 0.9  # Serve at least 90% of demand
        constraints.append(node_demand_served_kw[i] >= min_demand)
        # Don't serve more demand than exists
        constraints.append(node_demand_served_kw[i] <= node_demands.get(node, 0.0))
    
    # 4. Power balance constraints for each node
    # Generation + Inflow = Demand + Outflow
    for i, node in enumerate(nodes):
        inflow = 0
        outflow = 0
        
        for j, row in df.iterrows():
            if row.get("To", "") == node:  # Power flowing into this node
                inflow += power_flow_kw[j]
            if row.get("From", "") == node:  # Power flowing out of this node
                outflow += power_flow_kw[j]
        
        # Power balance: Generation + Inflow = Demand Served + Outflow
        constraints.append(node_generation_kw[i] + inflow == node_demand_served_kw[i] + outflow)
    
    # 5. Encourage demand satisfaction (soft constraint via objective modification)
    # Add penalty for unmet demand
    total_demand = sum(node_demands.values())
    if total_demand > 0:
        demand_penalty = cp.sum(cp.maximum(0, 
            cp.hstack([node_demands.get(node, 0.0) for node in nodes]) - node_demand_served_kw
        )) * 1000  # Large penalty weight
        objective = cp.Minimize(objective.args[0] + demand_penalty)
    
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Try alternative solver
            problem.solve(solver=cp.OSQP, verbose=False)
    except:
        pass
    
    # Extract solution or use fallback
    if power_flow_kw.value is not None and problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        optimal_flows = np.maximum(power_flow_kw.value, 0.0)
        node_gen_solution = np.maximum(node_generation_kw.value, 0.0) if node_generation_kw.value is not None else np.zeros(n_nodes)
        node_demand_solution = np.maximum(node_demand_served_kw.value, 0.0) if node_demand_served_kw.value is not None else np.zeros(n_nodes)
    else:
        # Fallback: Proportional allocation based on demand
        optimal_flows = np.zeros(n_lines)
        for i in range(n_lines):
            # Allocate based on demand at destination node
            to_node = df.iloc[i].get("To", "")
            demand_factor = min(node_demands.get(to_node, 0.0) / max(sum(node_demands.values()), 1.0), 1.0)
            optimal_flows[i] = min(available_energy_kw[i] * demand_factor, line_capacity_kw[i])
        
        node_gen_solution = np.zeros(n_nodes)
        node_demand_solution = np.array([node_demands.get(node, 0.0) for node in nodes])
    
    # Create flow results with actual loss calculations
    flows: List[FlowResult] = []
    total_demand_served = 0.0
    total_demand_required = sum(node_demands.values())
    
    for i, row in df.iterrows():
        fr = make_flow_header()
        fr.from_node = row.get("From", "")
        fr.to_node = row.get("To", "")
        fr.power_kw = float(optimal_flows[i])
        fr.voltage_kv = float(voltage_kv[i])
        
        # Calculate actual current and I²R losses
        current_a = compute_I_from_power_kw(fr.power_kw, fr.voltage_kv)
        resistance_total = resistance_ohm_per_km[i] * max(distance_km[i], 0.0)
        loss_kw = compute_I2R_loss_kw(current_a, resistance_total, phases=3)
        
        fr.current_a = float(current_a)
        fr.power_loss_kw = float(loss_kw)
        
        # Calculate efficiency
        total_power = fr.power_kw + fr.power_loss_kw
        fr.efficiency_percent = float(100.0 * fr.power_kw / total_power) if total_power > 0 else 100.0
        
        # Demand satisfaction metrics
        to_node = row.get("To", "")
        if to_node in node_demands:
            demand_at_node = node_demands[to_node]
            fr.demand_met_kw = float(min(fr.power_kw, demand_at_node))
            fr.demand_satisfaction_percent = float(100.0 * fr.demand_met_kw / demand_at_node) if demand_at_node > 0 else 100.0
            total_demand_served += fr.demand_met_kw
        else:
            fr.demand_met_kw = float(fr.power_kw)
            fr.demand_satisfaction_percent = 100.0
        
        flows.append(fr)
    
    # Calculate overall demand satisfaction
    overall_demand_satisfaction = 100.0 * total_demand_served / total_demand_required if total_demand_required > 0 else 100.0
    
    
    optimization_metrics = {
        "total_demand_required_kw": total_demand_required,
        "total_demand_served_kw": total_demand_served,
        "overall_demand_satisfaction_percent": overall_demand_satisfaction,
        "optimization_status": problem.status if 'problem' in locals() else "fallback",
        "total_generation_kw": float(np.sum(node_gen_solution)),
        "total_transmission_losses_kw": sum(fr.power_loss_kw for fr in flows),
        "average_line_utilization_percent": np.mean([min(fr.power_kw / max(line_capacity_kw[i], 1), 1.0) * 100 for i, fr in enumerate(flows)])
    }
    
    
    result = aggregate_flows_to_result(flows, "DC_OPF_Demand_Driven", t0, 1, optimization_metrics)
    return result