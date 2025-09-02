
import time, random, pandas as pd, numpy as np
from typing import List, Dict
from shared_components import FlowResult, aggregate_flows_to_result, make_flow_header, compute_I_from_power_kw, compute_I2R_loss_kw, SharedGridData

def demand_driven_fitness(individual, df, shared: SharedGridData, demand_map: Dict[str, float]):
    """
    Fitness function optimized for demand-driven allocation:
    - Maximizes demand satisfaction
    - Minimizes transmission losses
    - Penalizes over-allocation and voltage violations
    """
    total_score = 0.0
    delivered_power = {}
    total_losses = 0.0
    total_allocated = 0.0
    voltage_violations = 0.0
    
    # Calculate delivered power and losses for each edge
    for i, row in df.iterrows():
        allocated_kw = float(individual[i])
        total_allocated += allocated_kw
        
        if allocated_kw <= 0:
            continue
            
        # Calculate line losses
        voltage_kv = float(row.get("voltage_kv", 13.8))
        distance_km = float(row.get("eff_dist_km", row.get("Distance_km", 0.0) or 0.0))
        resistance_ohm_per_km = float(row.get("resistance_ohm_per_km", 0.03))
        
        if voltage_kv > 0 and distance_km > 0:
            current_a = compute_I_from_power_kw(allocated_kw, voltage_kv)
            resistance_ohm = resistance_ohm_per_km * distance_km
            loss_kw = compute_I2R_loss_kw(current_a, resistance_ohm, phases=3)
            
            # Check voltage drop violations (critical for rural areas)
            voltage_drop_kv = current_a * resistance_ohm / 1000.0
            if voltage_drop_kv > voltage_kv * 0.05:  # >5% voltage drop
                voltage_violations += voltage_drop_kv * 1000  # Heavy penalty
        else:
            loss_kw = allocated_kw * 0.02  # 2% estimated loss
        
        total_losses += loss_kw
        delivered_kw = max(0.0, allocated_kw - loss_kw)
        
        # Accumulate delivered power by destination node
        to_node = row.get("To", "")
        delivered_power[to_node] = delivered_power.get(to_node, 0.0) + delivered_kw
    
    # Evaluate demand satisfaction
    total_demand = sum(demand_map.values()) if demand_map else 1.0
    total_delivered = sum(delivered_power.values())
    demand_satisfaction_ratio = min(1.0, total_delivered / total_demand) if total_demand > 0 else 1.0
    
    # Calculate unmet demand penalty
    unmet_demand = 0.0
    for node_id, required_demand in demand_map.items():
        delivered = delivered_power.get(node_id, 0.0)
        if delivered < required_demand:
            unmet_demand += (required_demand - delivered)
    
    # Calculate over-allocation penalty
    theoretical_minimum = total_demand * 1.15  # Allow 15% headroom
    over_allocation = max(0.0, total_allocated - theoretical_minimum)
    
    # Calculate efficiency
    efficiency_ratio = total_delivered / total_allocated if total_allocated > 0 else 0.0
    
    # Composite fitness score (higher is better)
    fitness_score = (
        demand_satisfaction_ratio * 10000.0 +    # Primary: meet demand
        efficiency_ratio * 2000.0 +              # Secondary: efficiency
        -unmet_demand * 100.0 +                  # Penalty: unmet demand
        -total_losses * 2.0 +                    # Penalty: transmission losses
        -over_allocation * 50.0 +                # Penalty: over-allocation
        -voltage_violations * 10.0               # Penalty: voltage violations
    )
    
    return float(fitness_score)

def create_demand_driven_individual(df, demand_map: Dict[str, float], loss_estimates: Dict[int, float]):
    """Create individual based on downstream demand requirements"""
    individual = np.zeros(len(df))
    
    for i, row in df.iterrows():
        to_node = row.get("To", "")
        downstream_demand = demand_map.get(to_node, 0.0)
        estimated_loss = loss_estimates.get(i, downstream_demand * 0.05)  # 5% loss estimate
        
        # Allocate demand + losses + small headroom (10-20%)
        headroom_factor = 1.1 + 0.1 * np.random.rand()
        required_allocation = (downstream_demand + estimated_loss) * headroom_factor
        
        # Respect line capacity constraints
        line_capacity = float(row.get("line_capacity_kw", row.get("Capacity_kW", 1e9)))
        individual[i] = min(required_allocation, line_capacity)
        
        # Minimum allocation to keep lines active
        individual[i] = max(individual[i], 1.0)
    
    return individual

def estimate_line_losses(df, demand_map: Dict[str, float]) -> Dict[int, float]:
    """Pre-estimate losses for each line based on typical loading"""
    loss_estimates = {}
    
    for i, row in df.iterrows():
        to_node = row.get("To", "")
        typical_demand = demand_map.get(to_node, 50.0)  # Default 50kW if unknown
        
        voltage_kv = float(row.get("voltage_kv", 13.8))
        distance_km = float(row.get("eff_dist_km", row.get("Distance_km", 0.0) or 0.0))
        resistance_ohm_per_km = float(row.get("resistance_ohm_per_km", 0.03))
        
        if voltage_kv > 0 and distance_km > 0:
            current_a = compute_I_from_power_kw(typical_demand, voltage_kv)
            resistance_ohm = resistance_ohm_per_km * distance_km
            loss_kw = compute_I2R_loss_kw(current_a, resistance_ohm, phases=3)
            loss_estimates[i] = loss_kw
        else:
            loss_estimates[i] = typical_demand * 0.03  # 3% loss estimate
    
    return loss_estimates

def adaptive_crossover(parent1, parent2, demand_map: Dict[str, float], df):
    """Demand-aware crossover that preserves good demand-allocation patterns"""
    n = len(parent1)
    child = np.zeros(n)
    
    # For each edge, choose parent based on how well it serves downstream demand
    for i in range(n):
        to_node = df.iloc[i].get("To", "")
        demand = demand_map.get(to_node, 0.0)
        
        if demand > 0:
            # Choose parent whose allocation is closer to demand requirement
            p1_diff = abs(parent1[i] - demand * 1.1)  # 10% headroom
            p2_diff = abs(parent2[i] - demand * 1.1)
            
            if p1_diff < p2_diff:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        else:
            # Random choice for non-demand nodes
            child[i] = parent1[i] if np.random.rand() < 0.5 else parent2[i]
    
    return child

def demand_aware_mutation(individual, df, demand_map: Dict[str, float], loss_estimates: Dict[int, float], mutation_rate=0.1):
    """Mutation that adjusts allocations toward optimal demand satisfaction"""
    mutated = individual.copy()
    n = len(individual)
    
    for i in range(n):
        if np.random.rand() < mutation_rate:
            to_node = df.iloc[i].get("To", "")
            demand = demand_map.get(to_node, 0.0)
            estimated_loss = loss_estimates.get(i, 0.0)
            
            if demand > 0:
                # Mutate toward optimal allocation (demand + losses + headroom)
                optimal_allocation = (demand + estimated_loss) * (1.1 + 0.1 * np.random.rand())
                
                # Blend current allocation with optimal
                blend_factor = 0.3 + 0.4 * np.random.rand()  # 30-70% toward optimal
                new_allocation = mutated[i] * (1 - blend_factor) + optimal_allocation * blend_factor
                
                # Respect capacity constraints
                line_capacity = float(df.iloc[i].get("line_capacity_kw", 1e9))
                mutated[i] = max(1.0, min(new_allocation, line_capacity))
            else:
                # Random perturbation for non-demand nodes
                perturbation = 0.9 + 0.2 * np.random.rand()
                line_capacity = float(df.iloc[i].get("line_capacity_kw", 1e9))
                mutated[i] = max(1.0, min(mutated[i] * perturbation, line_capacity))
    
    return mutated

def run(shared: SharedGridData, df_edges: pd.DataFrame, pop=50, gens=100, elite_ratio=0.2, mutation_prob=0.3):
    """
    Demand-driven genetic algorithm for power flow optimization
    Focuses on meeting demand efficiently while minimizing losses and over-allocation
    """
    t0 = time.time()
    df = df_edges.copy().reset_index(drop=True)
    
    # Preprocess data
    if "Energy_kW" not in df.columns:
        df["Energy_kW"] = df.get("Energy_MWh", 0.0) * 1000.0
    if "eff_dist_km" not in df.columns:
        df["eff_dist_km"] = df.get("Distance_km", 0.0)
    
    # Build demand topology
    demand_map = {}
    for node_id, node in shared.nodes.items():
        demand_kw = float(getattr(node, 'base_demand_kw', 0.0) or 0.0)
        if demand_kw > 0:
            demand_map[node_id] = demand_kw
    
    # Pre-estimate line losses
    loss_estimates = estimate_line_losses(df, demand_map)
    
    # Initialize population with demand-driven individuals
    population = []
    for _ in range(pop):
        individual = create_demand_driven_individual(df, demand_map, loss_estimates)
        population.append(individual)
    
    # Evolution parameters
    elite_count = max(1, int(pop * elite_ratio))
    
    # Evolution loop
    best_fitness_history = []
    
    for generation in range(gens):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = demand_driven_fitness(individual, df, shared, demand_map)
            fitness_scores.append(fitness)
        
        # Track best fitness
        best_fitness = max(fitness_scores)
        best_fitness_history.append(best_fitness)
        
        # Selection: keep elite individuals
        fitness_order = np.argsort(fitness_scores)[::-1]  # Descending order
        elite_individuals = [population[i] for i in fitness_order[:elite_count]]
        
        # Generate new population
        new_population = elite_individuals.copy()
        
        # Tournament selection for breeding
        tournament_size = max(2, pop // 10)
        
        while len(new_population) < pop:
            # Tournament selection for parents
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parent1 = population[winner_idx]
            
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parent2 = population[winner_idx]
            
            # Crossover
            child = adaptive_crossover(parent1, parent2, demand_map, df)
            
            # Mutation
            if np.random.rand() < mutation_prob:
                child = demand_aware_mutation(child, df, demand_map, loss_estimates)
            
            new_population.append(child)
        
        population = new_population
    
    # Get best individual
    final_fitness = [demand_driven_fitness(ind, df, shared, demand_map) for ind in population]
    best_individual = population[np.argmax(final_fitness)]
    
    # Build flow results
    flows = []
    for i, row in df.iterrows():
        fr = make_flow_header()
        fr.from_node = row.get("From", "")
        fr.to_node = row.get("To", "")
        fr.power_kw = float(best_individual[i])
        fr.voltage_kv = float(row.get("voltage_kv", 13.8))
        
        # Calculate losses and efficiency
        distance_km = float(row.get("eff_dist_km", row.get("Distance_km", 0.0) or 0.0))
        resistance_ohm_per_km = float(row.get("resistance_ohm_per_km", 0.03))
        
        if fr.power_kw > 0 and fr.voltage_kv > 0:
            current_a = compute_I_from_power_kw(fr.power_kw, fr.voltage_kv)
            resistance_ohm = resistance_ohm_per_km * max(distance_km, 0.0)
            fr.current_a = float(current_a)
            fr.power_loss_kw = float(compute_I2R_loss_kw(current_a, resistance_ohm, phases=3))
            
            # Calculate actual delivered power
            delivered_power = max(0.0, fr.power_kw - fr.power_loss_kw)
            fr.demand_met_kw = float(delivered_power)
            fr.efficiency_percent = float(100.0 * delivered_power / fr.power_kw) if fr.power_kw > 0 else 100.0
        else:
            fr.current_a = 0.0
            fr.power_loss_kw = 0.0
            fr.demand_met_kw = 0.0
            fr.efficiency_percent = 100.0
        
        # Calculate demand satisfaction for destination node
        to_node_demand = demand_map.get(fr.to_node, 0.0)
        if to_node_demand > 0:
            fr.demand_satisfaction_percent = float(100.0 * min(1.0, fr.demand_met_kw / to_node_demand))
        else:
            fr.demand_satisfaction_percent = 100.0
        
        flows.append(fr)
    
    # Aggregate results
    result = aggregate_flows_to_result(flows, "Demand_Driven_Genetic_Algorithm", t0, gens, shared)
    
    # Add metadata
    total_demand = sum(demand_map.values())
    total_delivered = sum(flow.demand_met_kw for flow in flows)
    total_allocated = sum(flow.power_kw for flow in flows)
    total_losses = sum(flow.power_loss_kw for flow in flows)
    
    result.metadata.update({
        "population_size": pop,
        "generations": gens,
        "elite_ratio": elite_ratio,
        "mutation_probability": mutation_prob,
        "total_demand_kw": float(total_demand),
        "total_delivered_kw": float(total_delivered),
        "total_allocated_kw": float(total_allocated),
        "total_losses_kw": float(total_losses),
        "demand_satisfaction_percent": float(100.0 * total_delivered / total_demand) if total_demand > 0 else 100.0,
        "system_efficiency_percent": float(100.0 * total_delivered / total_allocated) if total_allocated > 0 else 100.0,
        "best_fitness": float(max(final_fitness)),
        "fitness_improvement": float(max(final_fitness) - best_fitness_history[0]) if best_fitness_history else 0.0
    })
    
    return result