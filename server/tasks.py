from typing import List, Dict, Any

def easy_grader(episode_log: List[Dict[str, Any]], **kwargs) -> float:
    """
    Grader for Baseline Generation (Easy).
    Evaluates what percentage of steps the reservoir level remained in the safe zone (40%-60%).
    """
    if not episode_log:
        return 0.0
        
    safe_steps = 0
    total_steps = 0
    
    for step_data in episode_log:
        obs = step_data.get('observation', {})
        if not obs:
            continue
            
        level = obs.get('reservoir_level', 0.0)
        
        if 40.0 <= level <= 60.0:
            safe_steps += 1
        total_steps += 1
            
    if total_steps == 0:
        return 0.0
        
    return safe_steps / total_steps

def medium_grader(episode_log: List[Dict[str, Any]], **kwargs) -> float:
    """
    Grader for Peak Shaving (Medium).
    Evaluates revenue generated as a fraction of maximum theoretically possible revenue.
    For simplicity we assume max theoretical revenue over 20 steps with max capacity 10.
    """
    if not episode_log:
        return 0.0
        
    total_revenue = 0.0
    max_theoretical_revenue = 0.0
    
    for step_data in episode_log:
        obs = step_data.get('observation', {})
        metadata = obs.get('metadata', {})
        
        grid_demand_price = obs.get('grid_demand_price', 1.0)
        
        # Theoretical max revenue assumes full turbine capacity (10.0 release)
        # However action is 0..1, so 1.0 * grid_demand_price * max_turbine (but max_turbine=10)
        # Standardizing simply by action = 1.0 -> revenue = MAX (or we just use raw grid_demand)
        # Actually in our env revenue = action * grid_demand_price. Max action is 1.0.
        max_theoretical_revenue += (10.0 * grid_demand_price)
        
        # Get actual revenue from metadata
        revenue = metadata.get('revenue', 0.0)
        total_revenue += revenue
        
    if max_theoretical_revenue <= 0.0:
        return 0.0
        
    return min(1.0, max(0.0, total_revenue / max_theoretical_revenue))

def hard_grader(episode_log: List[Dict[str, Any]], **kwargs) -> float:
    """
    Grader for Storm Surge (Hard).
    Returns 1.0 if no fatal constraint was breached across the episode, 0.0 otherwise.
    Checks: overflow (via breach flag), downstream flood (total_release >= capacity).
    """
    if not episode_log:
        return 0.0
    for step_data in episode_log:
        obs      = step_data.get('observation', {})
        metadata = obs.get('metadata', {})
        # Check breach flag (captures overflow and depletion — level is clamped in obs)
        if metadata.get('breach', False):
            return 0.0
        # Check downstream flood (use >= to match env breach condition)
        total_release = metadata.get('total_release', 0.0)
        capacity      = obs.get('downstream_capacity', 40.0)
        if total_release >= capacity:
            return 0.0
    return 1.0
