# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hydropulse Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import HydropulseAction, HydropulseObservation
except ImportError:
    from models import HydropulseAction, HydropulseObservation


class HydropulseEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = HydropulseEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Hydropulse environment ready!"
        >>>
        >>> obs = env.step(HydropulseAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_CAPACITY = 100.0
    MAX_TURBINE_CAPACITY = 10.0
    MAX_SPILLWAY_CAPACITY = 50.0
    DOWNSTREAM_CAPACITY = 40.0

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
            self.inflow_rate = 5.0
            self.grid_demand_price = 1.0
        elif self.task_type == "medium":
            self.inflow_rate = 3.0
            self.grid_demand_price = 1.0
        elif self.task_type == "hard":
            self.inflow_rate = 15.0 # Massive inflow
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
            self.inflow_rate = 40.0  # Storm surge
        elif self.task_type == "hard" and step > 15:
            self.inflow_rate = 5.0

        turbine_release = max(0.0, min(1.0, action.turbine_release))
        spillway_release = max(0.0, min(1.0, action.spillway_release))

        turbine_flow = turbine_release * self.MAX_TURBINE_CAPACITY
        spillway_flow = spillway_release * self.MAX_SPILLWAY_CAPACITY
        total_release = turbine_flow + spillway_flow

        # Physics: new_level = current_level + inflow - release
        self.reservoir_level = self.reservoir_level + self.inflow_rate - total_release

        # Reward shaping — all values clamped strictly to [0.0, 1.0]
        breach = False

        # Hard penalties: flag a breach but don't let reward go negative
        if self.reservoir_level > self.MAX_CAPACITY:
            breach = True
        if total_release > self.DOWNSTREAM_CAPACITY:
            breach = True

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
