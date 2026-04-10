# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HydroPulse-v1 Environment Implementation.

A continuous-state hydroelectric dam management environment.
The agent controls turbine and spillway releases to maximise revenue
while preventing reservoir overflow and downstream flooding.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import HydropulseAction, HydropulseObservation
except ImportError:
    from HydroPulse.models import HydropulseAction, HydropulseObservation


class HydropulseEnvironment(Environment):
    """
    HydroPulse-v1: Continuous-state hydroelectric dam management.

    Physics:
        new_level = current_level + inflow_rate
                  - (turbine_release * MAX_TURBINE_CAPACITY)
                  - (spillway_release * MAX_SPILLWAY_CAPACITY)

    Constraints:
        - reservoir_level must stay in [0.0, MAX_CAPACITY]
        - total_release must not exceed DOWNSTREAM_CAPACITY

    Reward: Revenue normalised to [0.0, 1.0], 0.0 on any breach.
    """

    # Enable concurrent WebSocket sessions.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_CAPACITY         = 100.0
    MAX_TURBINE_CAPACITY = 10.0
    # MAX_TURBINE + MAX_SPILLWAY == DOWNSTREAM_CAPACITY so the agent can
    # always exactly break even during the worst surge without flooding.
    MAX_SPILLWAY_CAPACITY = 30.0
    DOWNSTREAM_CAPACITY   = 40.0

    def __init__(self):
        """Initialize the HydroPulse environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.reservoir_level = 50.0
        self.inflow_rate = 5.0
        self.grid_demand_price = 1.0
        self.task_type = "easy"

    def reset(self) -> HydropulseObservation:
        """
        Reset the environment and set up the scenario.

        Returns:
            HydropulseObservation with initial state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        import random
        self.task_type = random.choice(["easy", "medium", "hard"])
        
        self.reservoir_level = 50.0
        
        if self.task_type == "easy":
            # Steady inflow — turbine=0.5 gives net=0 (level stable at 50)
            self.inflow_rate = 5.0
            self.grid_demand_price = 1.0
        elif self.task_type == "medium":
            # Low base inflow; price spikes at step 10
            # turbine=0.5 gives net=0 (5 in, 5 out) — level stays stable
            self.inflow_rate = 5.0
            self.grid_demand_price = 1.0
        elif self.task_type == "hard":
            # Moderate pre-surge inflow; agent must prepare before step 5
            self.inflow_rate = 5.0
            self.grid_demand_price = 1.0

        return HydropulseObservation(
            reservoir_level=self.reservoir_level,
            inflow_rate=self.inflow_rate,
            grid_demand_price=self.grid_demand_price,
            downstream_capacity=self.DOWNSTREAM_CAPACITY,
            value=0.0,
            done=False,
            reward=0.0,
            metadata={"task": self.task_type, "step": 0}
        )

    def step(self, action: HydropulseAction) -> HydropulseObservation:  # type: ignore[override]
        """
        Execute a step in the environment, advancing the physics simulation.

        Args:
            action: HydropulseAction containing turbine and spillway release factors

        Returns:
            HydropulseObservation with updated state and reward
        """
        self._state.step_count += 1
        step = self._state.step_count

        # Scenario dynamics
        if self.task_type == "medium" and step == 10:
            self.grid_demand_price = 5.0  # Spike
        elif self.task_type == "medium" and step > 10:
            self.grid_demand_price = max(1.0, self.grid_demand_price - 0.5)

        if self.task_type == "hard" and step == 5:
            # Storm surge: inflow = 30.0. Max release also = 40.0 > 30.0
            # so the agent CAN survive if spillway is open.
            self.inflow_rate = 30.0
        elif self.task_type == "hard" and step > 15:
            self.inflow_rate = 5.0

        turbine_release = max(0.0, min(1.0, action.turbine_release))
        spillway_release = max(0.0, min(1.0, action.spillway_release))

        turbine_flow = turbine_release * self.MAX_TURBINE_CAPACITY
        spillway_flow = spillway_release * self.MAX_SPILLWAY_CAPACITY
        total_release = turbine_flow + spillway_flow

        # Physics: new_level = current_level + inflow - release
        raw_level = self.reservoir_level + self.inflow_rate - total_release

        # Reward shaping — all values clamped strictly to [0.0, 1.0]
        breach = False

        # Breach conditions
        if raw_level > self.MAX_CAPACITY:
            breach = True   # Overflow — dam breaches
        if raw_level < 0.0:
            breach = True   # Reservoir emptied — physically impossible
        if total_release > self.DOWNSTREAM_CAPACITY:
            breach = True   # Downstream flood

        # Clamp level to physical bounds (even on breach, for next step safety)
        self.reservoir_level = max(0.0, min(self.MAX_CAPACITY, raw_level))

        if breach:
            # Any breach zeroes out the reward for this step
            reward = 0.0
        else:
            # Revenue normalised by max possible: turbine_release (max=1.0) * max_price (5.0)
            MAX_PRICE = 5.0
            revenue_normalised = (turbine_release * self.grid_demand_price) / MAX_PRICE

            # Buffer zone bonus: additional 0.1 added when level is safely in 40-60%
            buffer_bonus = 0.1 if 40.0 <= self.reservoir_level <= 60.0 else 0.0

            # Combine and clamp to [0.0, 1.0]
            reward = min(1.0, max(0.0, revenue_normalised + buffer_bonus))

        revenue = turbine_release * self.grid_demand_price
        done = step >= 20  # Fixed episode length

        return HydropulseObservation(
            reservoir_level=self.reservoir_level,
            inflow_rate=self.inflow_rate,
            grid_demand_price=self.grid_demand_price,
            downstream_capacity=self.DOWNSTREAM_CAPACITY,
            value=reward,
            done=done,
            reward=reward,
            metadata={
                "task": self.task_type,
                "step": step,
                "revenue": revenue,
                "total_release": total_release,
                "breach": breach,
            },
        )

    @property
    def state(self) -> State:
        return self._state
