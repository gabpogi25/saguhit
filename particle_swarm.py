
"""
Demand-Driven Particle Swarm Optimization for Power Flow
Optimizes power allocation to minimize transmission losses while meeting demand requirements.
Uses PSO to find optimal power flows that satisfy network constraints and demand.
"""
import time, numpy as np, pandas as pd
from typing import List, Dict, Tuple
from shared_components import FlowResult, aggregate_flows_to_result, make_flow_header, compute_I_from_power_kw, compute_I2R_loss_kw, SharedGridData

def run(shared: SharedGridData, df_edges: pd.DataFrame, swarm=25, iters=40, w=0.6, c1=1.4, c2=1.4):
    t0 = time.time()
    df = df_edges.copy().reset_index(drop=True)
    
    # Ensure required columns exist
    if "Energy_kW" not in df.columns:
        df["Energy_kW"] = df.get("Energy_MWh", 0.0) * 1000.0
    if "eff_dist_km" not in df.columns:
        df["eff_dist_km"] = df.get("Distance_km", 0.0)
    
    # Extract network parameters
    n_lines = len(df)
    line_capacity_kw = df["line_capacity_kw"].fillna(1e9).values
    available_energy_kw = df["Energy_kW"].fillna(0.0).values
    distance_km = df["eff_dist_km"].fillna(0.0).values
    voltage_kv = df["voltage_kv"].fillna(13.8).values
    resistance_ohm_per_km = df["resistance_ohm_per_km"].fillna(0.03).values
    
    # Build network topology and extract demands
    nodes = set()
    for _, row in df.iterrows():
        nodes.add(row.get("From", ""))
        nodes.add(row.get("To", ""))
    nodes = sorted(list(nodes))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    
    # Extract or estimate node demands
    node_demands = {}
    if hasattr(shared, 'node_demands') and shared.node_demands:
        node_demands = shared.node_demands
    else:
        # Estimate demands based on energy flow patterns
        for _, row in df.iterrows():
            to_node = row.get("To", "")
            if to_node and to_node not in node_demands:
                # Estimate demand as 70-90% of available energy
                demand_factor = 0.7 + 0.2 * np.random.rand()
                node_demands[to_node] = max(row.get("Energy_kW", 0.0) * demand_factor, 50.0)
    
    # Ensure all nodes have demand values
    for node in nodes:
        if node not in node_demands:
            node_demands[node] = 0.0
    
    total_demand = sum(node_demands.values())
    
    # Precompute line resistances
    line_resistance_ohm = resistance_ohm_per_km * distance_km
    
    def calculate_network_losses(power_flows):
        """Calculate total I²R losses for given power flows"""
        total_losses = 0.0
        for i in range(n_lines):
            if voltage_kv[i] > 0:
                current = power_flows[i] / (voltage_kv[i] * np.sqrt(3))  # 3-phase current
                loss = current**2 * line_resistance_ohm[i] * 3  # 3-phase I²R loss
                total_losses += loss
        return total_losses
    
    def calculate_demand_satisfaction(power_flows):
        """Calculate how well the power flows satisfy node demands"""
        node_power_delivered = {node: 0.0 for node in nodes}
        
        # Sum power delivered to each node
        for i, row in df.iterrows():
            to_node = row.get("To", "")
            if to_node in node_power_delivered:
                # Account for transmission losses
                current = power_flows[i] / max(voltage_kv[i] * np.sqrt(3), 1e-6)
                loss = current**2 * line_resistance_ohm[i] * 3
                delivered_power = max(power_flows[i] - loss, 0.0)
                node_power_delivered[to_node] += delivered_power
        
        # Calculate satisfaction metrics
        total_demand_met = 0.0
        demand_penalty = 0.0
        
        for node in nodes:
            required = node_demands.get(node, 0.0)
            delivered = node_power_delivered[node]
            met = min(delivered, required)
            total_demand_met += met
            
            if required > 0:
                shortage = max(required - delivered, 0.0)
                demand_penalty += (shortage / required) ** 2  # Quadratic penalty for unmet demand
        
        return total_demand_met, demand_penalty
    
    def fitness(power_flows):
        """
        Fitness function: Minimize transmission losses while maximizing demand satisfaction
        Higher fitness = better solution
        """
        # Ensure within capacity constraints
        power_flows = np.clip(power_flows, 0, line_capacity_kw)
        
        # Calculate transmission losses
        losses = calculate_network_losses(power_flows)
        
        # Calculate demand satisfaction
        demand_met, demand_penalty = calculate_demand_satisfaction(power_flows)
        
        # Multi-objective fitness: maximize demand satisfaction, minimize losses
        # Normalize components
        max_possible_losses = calculate_network_losses(line_capacity_kw)
        normalized_losses = losses / max(max_possible_losses, 1.0)
        
        demand_satisfaction_ratio = demand_met / max(total_demand, 1.0)
        
        # Weighted fitness: prioritize demand satisfaction, then minimize losses
        fitness_value = (
            demand_satisfaction_ratio * 1000.0  # High weight for demand satisfaction
            - normalized_losses * 100.0         # Moderate weight for loss minimization  
            - demand_penalty * 500.0             # Heavy penalty for unmet demand
        )
        
        return float(fitness_value)
    
    # Initialize particle swarm
    # Start with demand-proportional allocation
    X = np.zeros((swarm, n_lines))
    for p in range(swarm):
        for i in range(n_lines):
            to_node = df.iloc[i].get("To", "")
            demand_factor = node_demands.get(to_node, 0.0) / max(total_demand, 1.0)
            
            # Add randomization around demand-proportional allocation
            base_allocation = available_energy_kw[i] * demand_factor
            random_factor = 0.7 + 0.6 * np.random.rand()  # 0.7 to 1.3 multiplier
            
            X[p, i] = min(base_allocation * random_factor, line_capacity_kw[i])
    
    # Initialize velocities and personal bests
    V = np.random.randn(swarm, n_lines) * 0.1 * np.mean(line_capacity_kw)
    P = X.copy()
    P_fit = np.array([fitness(x) for x in X])
    
    # Global best
    g_idx = np.argmax(P_fit)
    G = P[g_idx].copy()
    G_fit = P_fit[g_idx]
    
    # Track optimization progress
    best_fitness_history = [G_fit]
    
    # PSO main loop
    for iteration in range(iters):
        # Update velocities and positions
        r1 = np.random.rand(swarm, n_lines)
        r2 = np.random.rand(swarm, n_lines)
        
        # PSO velocity update
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (G - X)
        
        # Update positions with constraints
        X = X + V
        X = np.clip(X, 0, line_capacity_kw)  # Respect capacity constraints
        
        # Ensure we don't exceed available energy
        for i in range(n_lines):
            X[:, i] = np.minimum(X[:, i], available_energy_kw[i])
        
        # Evaluate fitness
        fit = np.array([fitness(x) for x in X])
        
        # Update personal bests
        better = fit > P_fit
        P[better] = X[better]
        P_fit[better] = fit[better]
        
        # Update global best
        if P_fit.max() > G_fit:
            g_idx = np.argmax(P_fit)
            G = P[g_idx].copy()
            G_fit = P_fit.max()
        
        best_fitness_history.append(G_fit)
        
        # Adaptive inertia weight (decrease over time)
        w = 0.9 - 0.5 * (iteration / iters)
    
    # Use the best solution found
    optimal_flows = G
    
    # Create flow results with proper demand tracking
    flows: List[FlowResult] = []
    total_demand_served = 0.0
    total_losses = 0.0
    
    for i, row in df.iterrows():
        fr = make_flow_header()
        fr.from_node = row.get("From", "")
        fr.to_node = row.get("To", "")
        fr.power_kw = float(optimal_flows[i])
        fr.voltage_kv = float(voltage_kv[i])
        
        # Calculate actual transmission losses
        current_a = compute_I_from_power_kw(fr.power_kw, fr.voltage_kv)
        resistance_total = resistance_ohm_per_km[i] * max(distance_km[i], 0.0)
        loss_kw = compute_I2R_loss_kw(current_a, resistance_total, phases=3)
        
        fr.current_a = float(current_a)
        fr.power_loss_kw = float(loss_kw)
        total_losses += loss_kw
        
        # Calculate efficiency
        total_power = fr.power_kw + fr.power_loss_kw
        fr.efficiency_percent = float(100.0 * fr.power_kw / total_power) if total_power > 0 else 100.0
        
        # Demand satisfaction metrics
        to_node = row.get("To", "")
        delivered_power = max(fr.power_kw - fr.power_loss_kw, 0.0)  # Power after losses
        
        if to_node in node_demands:
            demand_at_node = node_demands[to_node]
            fr.demand_met_kw = float(min(delivered_power, demand_at_node))
            fr.demand_satisfaction_percent = float(100.0 * fr.demand_met_kw / demand_at_node) if demand_at_node > 0 else 100.0
            total_demand_served += fr.demand_met_kw
        else:
            fr.demand_met_kw = float(delivered_power)
            fr.demand_satisfaction_percent = 100.0
        
        flows.append(fr)
    
    # Calculate overall metrics
    overall_demand_satisfaction = 100.0 * total_demand_served / total_demand if total_demand > 0 else 100.0
    
    # PSO-specific optimization metrics
    optimization_metrics = {
        "total_demand_required_kw": total_demand,
        "total_demand_served_kw": total_demand_served,
        "overall_demand_satisfaction_percent": overall_demand_satisfaction,
        "total_transmission_losses_kw": total_losses,
        "final_fitness_value": float(G_fit),
        "fitness_improvement": float(G_fit - best_fitness_history[0]) if len(best_fitness_history) > 1 else 0.0,
        "convergence_iterations": iters,
        "swarm_size": swarm,
        "average_line_utilization_percent": np.mean([min(fr.power_kw / max(line_capacity_kw[i], 1), 1.0) * 100 for i, fr in enumerate(flows)]),
        "optimization_method": "PSO_Demand_Driven"
    }
    
    return aggregate_flows_to_result(flows, "Particle_Swarm_Optimization_Demand_Driven", t0, iters, optimization_metrics)