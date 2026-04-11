# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HydroPulse-v1 Environment Implementation.

A continuous-state hydroelectric dam management environment using non-linear
fluid dynamics (Torricelli's Law / Hydraulic Head) and evaporation modelling.

Physics (Torricelli's Law):
    head_pressure        = sqrt(current_level / 100.0)
    actual_turbine_flow  = turbine_release * 10.0 * head_pressure
    actual_spillway_flow = spillway_release * 30.0 * head_pressure
    evap_loss            = 0.05 * (current_level ** 0.66)
    new_level            = current_level + inflow - turbine_flow
                           - spillway_flow - evap_loss

Reward: Power-revenue normalised to [0.0, 1.0]. Breach → 0.0 and done=True.
"""

import math
import random
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

    Upgraded physics engine uses:
      - Hydraulic Head / Torricelli's Law for pressure-dependent flow
      - Evaporation loss modelled as a power-law of current level
      - Diurnal grid-demand price via a sine wave (peaks ~step 6 and 18)

    Constraints (all cause reward = 0.0 and episode termination):
      - reservoir_level > MAX_CAPACITY  → dam overflow
      - reservoir_level < 0.0           → reservoir depleted (physically impossible)
      - total_release   > DOWNSTREAM_CAPACITY → downstream flood
    """

    # Enable concurrent WebSocket sessions.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_CAPACITY          = 100.0
    MAX_TURBINE_CAPACITY  = 10.0   # max turbine flow at full head (units/step)
    MAX_SPILLWAY_CAPACITY = 30.0   # max spillway flow at full head (units/step)
    DOWNSTREAM_CAPACITY   = 40.0   # max safe combined outflow (units/step)

    # Reward normalisation denominator:
    # theoretical max revenue = MAX_TURBINE * MAX_GRID_PRICE
    #                         = 10.0 * 80.0 = 800.0
    MAX_TURBINE_FLOW  = 10.0
    MAX_GRID_PRICE    = 80.0       # sine wave peaks at 50 + 30 = 80

    def __init__(self):
        """Initialise the HydroPulse environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.reservoir_level   = 50.0
        self.inflow_rate       = 5.0
        self.grid_demand_price = 1.0
        self.task_type         = "easy"
        self._rng              = random.Random()  # per-episode RNG (seeded on reset)

    def reset(self) -> HydropulseObservation:
        """
        Reset the environment and randomise the task scenario.

        Returns:
            HydropulseObservation with initial state (step 0).
        """
        episode_id = str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)
        self._reset_count += 1

        # Seed per-episode RNG from episode_id for reproducibility
        self._rng.seed(episode_id)
        self.task_type = self._rng.choice(["easy", "medium", "hard"])

        self.reservoir_level = 50.0

        if self.task_type == "easy":
            # Steady inflow — agent must hold level 40-60 while generating power
            self.inflow_rate = 5.0
        elif self.task_type == "medium":
            # Steady inflow; price cycles drive strategic dispatch timing
            self.inflow_rate = 5.0
        elif self.task_type == "hard":
            # Pre-surge inflow; agent must prepare headroom before step 5
            self.inflow_rate = 5.0

        # Compute initial diurnal price at step 0
        self.grid_demand_price = self._diurnal_price(0)

        return HydropulseObservation(
            reservoir_level=self.reservoir_level,
            inflow_rate=self.inflow_rate,
            grid_demand_price=self.grid_demand_price,
            downstream_capacity=self.DOWNSTREAM_CAPACITY,
            value=0.0,
            done=False,
            reward=0.0,
            step_number=0,
            metadata={"task": self.task_type, "step": 0},
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _diurnal_price(self, step: int) -> float:
        """
        Dynamic grid demand price using a sine wave.

        Peaks at ~80.0 (evening demand), troughs at ~20.0 (night).
        Period = 24 steps (simulate 24-hour day).
        """
        return 50.0 + 30.0 * math.sin(2.0 * math.pi * step / 24.0)

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(self, action: HydropulseAction) -> HydropulseObservation:  # type: ignore[override]
        """
        Execute one step using Torricelli hydraulic-head dynamics.

        Non-linear physics:
            head_pressure        = sqrt(current_level / 100.0)
            actual_turbine_flow  = turbine_release * 10.0 * head_pressure
            actual_spillway_flow = spillway_release * 30.0 * head_pressure
            evap_loss            = 0.05 * (current_level ** 0.66)
            new_level            = current_level + inflow - turbine_flow
                                   - spillway_flow - evap_loss

        Returns:
            HydropulseObservation with updated state and normalised reward.
        """
        self._state.step_count += 1
        step = self._state.step_count

        # ── 1. Scenario dynamics ───────────────────────────────────────────────
        if self.task_type == "hard" and step == 5:
            # Storm surge: inflow=45 exceeds absolute max release (40.0) at any head pressure,
            # guaranteeing overflow unless agent pre-drains below ~20% before step 5.
            self.inflow_rate = 45.0
        elif self.task_type == "hard" and step > 14:
            self.inflow_rate = 5.0

        # Diurnal grid price + Gaussian noise (σ=5) — forces real uncertainty
        # Noise makes the price signal unpredictable; LLM must reason, not memorise.
        base_price    = self._diurnal_price(step)
        price_noise   = self._rng.gauss(0.0, 5.0)
        current_price = max(10.0, base_price + price_noise)  # floor at 10 (market minimum)
        self.grid_demand_price = current_price

        # ── 2. Clamp actions ──────────────────────────────────────────────────
        # Stochastic inflow perturbation: ±1.5 units (uniform) — simulates
        # measurement uncertainty and sub-step rainfall variability.
        inflow_noise      = self._rng.uniform(-1.5, 1.5)
        current_inflow    = max(0.5, self.inflow_rate + inflow_noise)  # min 0.5
        self.inflow_rate  = current_inflow  # expose noisy value in observation

        turbine_release  = max(0.0, min(1.0, action.turbine_release))
        spillway_release = max(0.0, min(1.0, action.spillway_release))

        # ── 3. Hydraulic Head (Torricelli's Law) ──────────────────────────────
        # head_pressure ∈ [0.0, 1.0]: flow is proportional to sqrt(fill fraction)
        # A full reservoir at 100 units produces max flow; an empty one produces 0.
        head_pressure = math.sqrt(max(0.0, self.reservoir_level) / self.MAX_CAPACITY)

        actual_turbine_flow  = turbine_release  * self.MAX_TURBINE_CAPACITY  * head_pressure
        actual_spillway_flow = spillway_release * self.MAX_SPILLWAY_CAPACITY * head_pressure
        total_release        = actual_turbine_flow + actual_spillway_flow

        # ── 4. Evaporation (power-law of current level) ───────────────────────
        evap_loss = 0.05 * (max(0.0, self.reservoir_level) ** 0.66)

        # ── 5. Mass-balance ───────────────────────────────────────────────────
        raw_level = (
            self.reservoir_level
            + current_inflow
            - actual_turbine_flow
            - actual_spillway_flow
            - evap_loss
        )

        # ── 6. Breach detection ───────────────────────────────────────────────
        breach = False
        if raw_level > self.MAX_CAPACITY:
            breach = True   # Overflow — dam fails
        if raw_level < 0.0:
            breach = True   # Reservoir fully depleted
        if total_release >= self.DOWNSTREAM_CAPACITY:
            breach = True   # Downstream flood

        # Clamp level to physical bounds regardless of breach
        self.reservoir_level = max(0.0, min(self.MAX_CAPACITY, raw_level))

        # ── 7. Reward — strictly [0.0, 1.0] ──────────────────────────────────
        if breach:
            reward = 0.0
            done   = True   # Episode ends immediately on breach
        else:
            # Revenue = power generated × current price
            # Normalised against theoretical max: MAX_TURBINE_FLOW * MAX_GRID_PRICE
            revenue = actual_turbine_flow * current_price
            reward  = min(1.0, max(0.0, revenue / (self.MAX_TURBINE_FLOW * self.MAX_GRID_PRICE)))
            done    = step >= 20  # Normal episode end

        return HydropulseObservation(
            reservoir_level=self.reservoir_level,
            inflow_rate=self.inflow_rate,
            grid_demand_price=current_price,
            downstream_capacity=self.DOWNSTREAM_CAPACITY,
            value=reward,
            done=done,
            reward=reward,
            step_number=step,
            metadata={
                "task":                self.task_type,
                "step":                step,
                "head_pressure":       round(head_pressure, 4),
                "actual_turbine_flow": round(actual_turbine_flow, 4),
                "actual_spillway_flow": round(actual_spillway_flow, 4),
                "evap_loss":           round(evap_loss, 4),
                "total_release":       round(total_release, 4),
                "breach":              breach,
            },
        )

    @property
    def state(self) -> State:
        return self._state
