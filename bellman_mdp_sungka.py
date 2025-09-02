import time, math, numpy as np, pandas as pd, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from shared_components import SharedGridData, FlowResult, aggregate_flows_to_result, make_flow_header, compute_I_from_power_kw, compute_I2R_loss_kw

@dataclass
class BellmanConfig:
    step_fraction: float = 0.10
    gamma: float = 0.95
    value_iteration_tol: float = 1e-4
    max_value_iterations: int = 100
    max_policy_iterations: int = 30
    unmet_penalty_per_kw: float = 1000.0
    loss_weight: float = 1.0
    overprovision_penalty: float = 100.0
    headroom_factor: float = 0.15
    min_allocation_kw: float = 1.0
    allocation_step_kw: float = 10.0
    voltage_penalty_weight: float = 500.0
    max_episode_length: int = 50

@dataclass(frozen=True)
class PowerFlowState:
    flows: Tuple[Tuple[str, str, int]]
    current_step: int
    
    def __hash__(self):
        return hash((self.flows, self.current_step))

class SungkaAction:
    def __init__(self, edge: Tuple[str, str], allocation_change: int, action_type: str = "adjust"):
        self.edge = edge
        self.allocation_change = allocation_change
        self.action_type = action_type
    
    def __eq__(self, other):
        return (isinstance(other, SungkaAction) and 
                self.edge == other.edge and
                self.allocation_change == other.allocation_change and
                self.action_type == other.action_type)
    
    def __hash__(self):
        return hash((self.edge, self.allocation_change, self.action_type))

class BellmanMDPSungka:
    def __init__(self, shared: SharedGridData, df_edges: pd.DataFrame, cfg: BellmanConfig = None):
        self.shared = shared
        self.df = df_edges.copy().reset_index(drop=True)
        if 'Energy_kW' not in self.df.columns:
            self.df['Energy_kW'] = self.df.get('Energy_MWh', 0.0) * 1000.0
        if 'eff_dist_km' not in self.df.columns:
            self.df['eff_dist_km'] = self.df.get('Distance_km', 0.0)
        self.cfg = cfg or BellmanConfig()
        
        self.demand_map = self._build_demand_topology()
        self.loss_estimates = self._estimate_line_losses()
        self.max_power_levels = self._compute_max_power_levels()
        
        self.states: Dict[PowerFlowState, int] = {}
        self.state_values: Dict[PowerFlowState, float] = {}
        self.q_values: Dict[Tuple[PowerFlowState, SungkaAction], float] = {}
        self.policy: Dict[PowerFlowState, SungkaAction] = {}
        self.actions_cache: Dict[PowerFlowState, List[SungkaAction]] = {}
    
    def _build_demand_topology(self) -> Dict[str, float]:
        demands = {}
        for node_id, node in self.shared.nodes.items():
            demand_kw = float(getattr(node, 'base_demand_kw', 0.0) or 0.0)
            if demand_kw > 0:
                demands[node_id] = demand_kw
        return demands
    
    def _compute_max_power_levels(self) -> Dict[Tuple[str, str], float]:
        max_levels = {}
        for _, row in self.df.iterrows():
            key = (row['From'], row['To'])
            capacity = row.get('line_capacity_kw', row.get('Capacity_kW', 
                              row.get('Energy_kW', 1000.0) * 2))
            max_levels[key] = float(capacity)
        return max_levels
    
    def _estimate_line_losses(self) -> Dict[Tuple[str, str], float]:
        loss_estimates = {}
        for _, row in self.df.iterrows():
            edge = (row['From'], row['To'])
            to_node = row['To']
            typical_power = self.demand_map.get(to_node, 100.0)
            
            voltage_kv = float(row.get('voltage_kv', 13.8))
            distance_km = float(row.get('eff_dist_km', row.get('Distance_km', 0.0) or 0.0))
            resistance_ohm_per_km = float(row.get('resistance_ohm_per_km', 0.03))
            
            if voltage_kv > 0 and distance_km > 0:
                current_a = compute_I_from_power_kw(typical_power, voltage_kv)
                resistance_ohm = resistance_ohm_per_km * distance_km
                loss_kw = compute_I2R_loss_kw(current_a, resistance_ohm, phases=3)
                loss_estimates[edge] = loss_kw
            else:
                loss_estimates[edge] = typical_power * 0.02
        return loss_estimates
    
    def _discretize_state(self, df_state: pd.DataFrame) -> PowerFlowState:
        flows = []
        for _, row in df_state.iterrows():
            from_node, to_node = row['From'], row['To']
            power_kw = float(row.get('Energy_kW', 0.0))
            max_power = self.max_power_levels.get((from_node, to_node), 1000.0)
            
            if max_power > 0:
                discrete_level = min(99, int((power_kw / max_power) * 100))
            else:
                discrete_level = 0
            flows.append((from_node, to_node, discrete_level))
        
        return PowerFlowState(tuple(sorted(flows)), 0)
    
    def _state_to_dataframe(self, state: PowerFlowState) -> pd.DataFrame:
        df = self.df.copy()
        allocation_dict = {(f, t): level for f, t, level in state.flows}
        
        for idx, row in df.iterrows():
            key = (row['From'], row['To'])
            if key in allocation_dict:
                discrete_level = allocation_dict[key]
                max_power = self.max_power_levels.get(key, 1000.0)
                allocated_kw = (discrete_level / 100.0) * max_power
                df.loc[idx, 'Energy_kW'] = max(self.cfg.min_allocation_kw, allocated_kw)
            else:
                df.loc[idx, 'Energy_kW'] = self.cfg.min_allocation_kw
        return df
    
    def _unmet_of_state(self, df_state: pd.DataFrame) -> float:
        recv = {}
        for _, row in df_state.iterrows():
            to_node = row.get("To", "")
            power_kw = float(row.get("Energy_kW", 0.0))
            
            voltage_kv = float(row.get("voltage_kv", 13.8))
            distance_km = float(row.get("eff_dist_km", row.get("Distance_km", 0.0) or 0.0))
            resistance_ohm_per_km = float(row.get("resistance_ohm_per_km", 0.03))
            
            if power_kw > 0 and voltage_kv > 0:
                current_a = compute_I_from_power_kw(power_kw, voltage_kv)
                resistance_ohm = resistance_ohm_per_km * max(distance_km, 0.0)
                loss_kw = compute_I2R_loss_kw(current_a, resistance_ohm, phases=3)
                delivered_kw = max(0.0, power_kw - loss_kw)
            else:
                delivered_kw = 0.0
            
            recv[to_node] = recv.get(to_node, 0.0) + delivered_kw
        
        unmet = 0.0
        for nid, node in self.shared.nodes.items():
            demand = float(getattr(node, "base_demand_kw", 0.0) or 0.0)
            received = recv.get(nid, 0.0)
            if demand > received:
                unmet += (demand - received)
        return float(unmet)
    
    def _losses_of_state(self, df_state: pd.DataFrame) -> float:
        losses = 0.0
        for _, row in df_state.iterrows():
            P = float(row.get("Energy_kW", 0.0))
            V = float(row.get("voltage_kv", 13.8))
            d = float(row.get("eff_dist_km", row.get("Distance_km", 0.0) or 0.0))
            Rpkm = float(row.get("resistance_ohm_per_km", 0.03))
            R = Rpkm * max(d, 0.0)
            I = compute_I_from_power_kw(P, V)
            losses += compute_I2R_loss_kw(I, R, phases=3)
        return float(losses)
    
    def _overprovision_penalty(self, df_state: pd.DataFrame) -> float:
        total_allocated = sum(float(row.get("Energy_kW", 0.0)) for _, row in df_state.iterrows())
        total_demand = sum(self.demand_map.values()) if self.demand_map else 1.0
        over_allocation = max(0.0, total_allocated - total_demand * (1.0 + self.cfg.headroom_factor))
        return over_allocation
    
    def _voltage_violations(self, df_state: pd.DataFrame) -> float:
        violations = 0.0
        for _, row in df_state.iterrows():
            power_kw = float(row.get("Energy_kW", 0.0))
            voltage_kv = float(row.get("voltage_kv", 13.8))
            distance_km = float(row.get("eff_dist_km", row.get("Distance_km", 0.0) or 0.0))
            resistance_ohm_per_km = float(row.get("resistance_ohm_per_km", 0.03))
            
            if power_kw > 0 and voltage_kv > 0 and distance_km > 0:
                current_a = compute_I_from_power_kw(power_kw, voltage_kv)
                voltage_drop_kv = current_a * resistance_ohm_per_km * distance_km / 1000.0
                if voltage_drop_kv > voltage_kv * 0.05:
                    violations += voltage_drop_kv
        return violations
    
    def _reward(self, state: PowerFlowState) -> float:
        df_state = self._state_to_dataframe(state)
        losses = self._losses_of_state(df_state)
        unmet = self._unmet_of_state(df_state)
        overprovision = self._overprovision_penalty(df_state)
        voltage_viol = self._voltage_violations(df_state)
        
        return -(self.cfg.loss_weight * losses + 
                self.cfg.unmet_penalty_per_kw * unmet +
                self.cfg.overprovision_penalty * overprovision +
                self.cfg.voltage_penalty_weight * voltage_viol)
    
    def _enumerate_actions(self, state: PowerFlowState) -> List[SungkaAction]:
        if state in self.actions_cache:
            return self.actions_cache[state]
        
        if state.current_step >= self.cfg.max_episode_length:
            return [SungkaAction(('_term_', '_term_'), 0, "terminate")]
        
        actions = []
        df_state = self._state_to_dataframe(state)
        allocation_dict = {(f, t): level for f, t, level in state.flows}
        
        # Group by source node for Sungka transfers
        edges_by_source = defaultdict(list)
        for _, row in df_state.iterrows():
            from_node = row['From']
            to_node = row['To']
            edge = (from_node, to_node)
            current_level = allocation_dict.get(edge, 1)
            
            # Check if this destination needs more/less power
            demand = self.demand_map.get(to_node, 0.0)
            current_kw = current_level * self.cfg.allocation_step_kw
            estimated_loss = self.loss_estimates.get(edge, 0.0)
            needed_kw = demand + estimated_loss
            
            shortage = max(0.0, needed_kw - current_kw)
            excess = max(0.0, current_kw - needed_kw * 1.1)
            
            edges_by_source[from_node].append((edge, current_level, shortage, excess))
        
        # Generate Sungka actions (transfers between lines from same source)
        for source_node, edge_info in edges_by_source.items():
            if len(edge_info) < 2:
                continue
            
            # Sort by shortage (highest first) and excess (lowest first)
            shortage_sorted = sorted(edge_info, key=lambda x: x[2], reverse=True)
            excess_sorted = sorted(edge_info, key=lambda x: x[3], reverse=True)
            
            # Transfer from excess lines to shortage lines
            for donor_edge, donor_level, _, donor_excess in excess_sorted:
                if donor_level <= 1 or donor_excess <= 0:
                    continue
                for receiver_edge, receiver_level, receiver_shortage, _ in shortage_sorted:
                    if donor_edge == receiver_edge or receiver_shortage <= 0:
                        continue
                    
                    transfer_levels = min(3, donor_level // 2, 
                                        int(receiver_shortage / self.cfg.allocation_step_kw))
                    if transfer_levels > 0:
                        actions.append(SungkaAction(donor_edge, -transfer_levels, "decrease"))
                        actions.append(SungkaAction(receiver_edge, transfer_levels, "increase"))
        
        # Direct demand-driven adjustments
        for _, row in df_state.iterrows():
            edge = (row['From'], row['To'])
            to_node = row['To']
            demand = self.demand_map.get(to_node, 0.0)
            current_level = allocation_dict.get(edge, 1)
            current_kw = current_level * self.cfg.allocation_step_kw
            
            if demand > current_kw:
                shortage_levels = min(5, int((demand - current_kw) / self.cfg.allocation_step_kw) + 1)
                actions.append(SungkaAction(edge, shortage_levels, "increase"))
            elif current_kw > demand * 1.2:
                excess_levels = min(3, int((current_kw - demand * 1.1) / self.cfg.allocation_step_kw))
                if current_level > excess_levels:
                    actions.append(SungkaAction(edge, -excess_levels, "decrease"))
        
        actions.append(SungkaAction(('_term_', '_term_'), 0, "terminate"))
        self.actions_cache[state] = actions
        return actions
    
    def _apply_action(self, state: PowerFlowState, action: SungkaAction) -> PowerFlowState:
        if action.action_type == "terminate":
            return PowerFlowState(state.flows, self.cfg.max_episode_length)
        
        allocation_dict = {(f, t): level for f, t, level in state.flows}
        
        if action.edge in allocation_dict:
            current_level = allocation_dict[action.edge]
            new_level = max(1, current_level + action.allocation_change)
            allocation_dict[action.edge] = new_level
        
        new_flows = tuple(sorted([(f, t, level) for (f, t), level in allocation_dict.items()]))
        return PowerFlowState(new_flows, state.current_step + 1)
    
    def _value_iteration(self) -> bool:
        max_delta = 0.0
        new_state_values = {}
        
        for state in self.states:
            if state not in self.state_values:
                self.state_values[state] = 0.0
            
            old_value = self.state_values[state]
            actions = self._enumerate_actions(state)
            
            if not actions or state.current_step >= self.cfg.max_episode_length:
                new_state_values[state] = self._reward(state)
                continue
            
            max_q_value = float('-inf')
            for action in actions:
                next_state = self._apply_action(state, action)
                immediate_reward = self._reward(next_state)
                
                if next_state not in self.states:
                    self.states[next_state] = len(self.states)
                    self.state_values[next_state] = 0.0
                
                future_value = self.state_values.get(next_state, 0.0)
                q_value = immediate_reward + self.cfg.gamma * future_value
                max_q_value = max(max_q_value, q_value)
            
            new_state_values[state] = max_q_value
            delta = abs(new_state_values[state] - old_value)
            max_delta = max(max_delta, delta)
        
        self.state_values.update(new_state_values)
        return max_delta < self.cfg.value_iteration_tol
    
    def _policy_improvement(self):
        for state in self.states:
            actions = self._enumerate_actions(state)
            if not actions:
                continue
            
            best_action = None
            best_q_value = float('-inf')
            
            for action in actions:
                next_state = self._apply_action(state, action)
                immediate_reward = self._reward(next_state)
                future_value = self.state_values.get(next_state, 0.0)
                q_value = immediate_reward + self.cfg.gamma * future_value
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            if best_action is not None:
                self.policy[state] = best_action
    
    def _create_initial_state(self) -> PowerFlowState:
        allocations = []
        for _, row in self.df.iterrows():
            from_node = row['From']
            to_node = row['To']
            edge = (from_node, to_node)
            
            downstream_demand = self.demand_map.get(to_node, 0.0)
            estimated_loss = self.loss_estimates.get(edge, 0.0)
            headroom = (downstream_demand + estimated_loss) * self.cfg.headroom_factor
            
            required_kw = downstream_demand + estimated_loss + headroom
            discrete_level = max(1, int(required_kw / self.cfg.allocation_step_kw))
            allocations.append((from_node, to_node, discrete_level))
        
        return PowerFlowState(tuple(sorted(allocations)), 0)
    
    def optimize(self):
        t0 = time.time()
        
        initial_state = self._create_initial_state()
        self.states[initial_state] = 0
        self.state_values[initial_state] = self._reward(initial_state)
        
        converged = False
        policy_iter = 0
        total_value_iters = 0
        
        while not converged and policy_iter < self.cfg.max_policy_iterations:
            policy_iter += 1
            
            value_iter = 0
            value_converged = False
            while not value_converged and value_iter < self.cfg.max_value_iterations:
                value_iter += 1
                total_value_iters += 1
                value_converged = self._value_iteration()
            
            old_policy = self.policy.copy()
            self._policy_improvement()
            
            if old_policy == self.policy:
                converged = True
        
        current_state = initial_state
        execution_steps = 0
        
        while current_state.current_step < self.cfg.max_episode_length:
            if current_state not in self.policy:
                break
            
            action = self.policy[current_state]
            if action.action_type == "terminate":
                break
            
            next_state = self._apply_action(current_state, action)
            if next_state == current_state:
                break
            
            current_state = next_state
            execution_steps += 1
        
        final_df = self._state_to_dataframe(current_state)
        
        flows = []
        for _, row in final_df.iterrows():
            fr = make_flow_header()
            fr.from_node = row.get("From", "")
            fr.to_node = row.get("To", "")
            fr.power_kw = float(row.get("Energy_kW", 0.0))
            fr.voltage_kv = float(row.get("voltage_kv", 13.8))
            
            dist = float(row.get("eff_dist_km", row.get("Distance_km", 0.0) or 0.0))
            rpkm = float(row.get("resistance_ohm_per_km", 0.03))
            
            I = compute_I_from_power_kw(fr.power_kw, fr.voltage_kv)
            R = rpkm * max(dist, 0.0)
            fr.current_a = float(I)
            fr.power_loss_kw = float(compute_I2R_loss_kw(I, R, phases=3))
            
            delivered_power = max(0.0, fr.power_kw - fr.power_loss_kw)
            fr.demand_met_kw = float(delivered_power)
            
            denom = fr.power_kw + fr.power_loss_kw
            fr.efficiency_percent = float(100.0 * fr.power_kw / denom) if denom > 0 else 100.0
            
            to_demand = self.demand_map.get(fr.to_node, 0.0)
            fr.demand_satisfaction_percent = float(100.0 * min(1.0, delivered_power / to_demand)) if to_demand > 0 else 100.0
            
            flows.append(fr)
        
        result = aggregate_flows_to_result(flows, "Demand_Driven_Bellman_Sungka", t0, total_value_iters, shared=self.shared)
        
        result.metadata.update({
            "policy_iterations": policy_iter,
            "value_iterations": total_value_iters,
            "total_states_explored": len(self.states),
            "execution_steps": execution_steps,
            "converged": converged,
            "gamma": self.cfg.gamma
        })
        
        return result

def run(shared: SharedGridData, df_edges: pd.DataFrame):
    return BellmanMDPSungka(shared, df_edges, BellmanConfig()).optimize()
