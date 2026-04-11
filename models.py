# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Hydropulse Environment.

The HydroPulse environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class HydropulseAction(Action):
    """Action for the Hydropulse environment - turbine and spillway release control."""
    turbine_release: float = Field(..., ge=0.0, le=1.0, description="Turbine release factor (0.0 to 1.0)")
    spillway_release: float = Field(..., ge=0.0, le=1.0, description="Spillway release factor (0.0 to 1.0)")


class HydropulseObservation(Observation):
    """Observation from the Hydropulse environment - water levels and grid demand."""
    reservoir_level: float = Field(default=0.0, description="Current water level in reservoir")
    inflow_rate: float = Field(default=0.0, description="Rate of water entering the reservoir")
    grid_demand_price: float = Field(default=0.0, description="Current price multiplier for generated power")
    downstream_capacity: float = Field(default=40.0, description="Maximum safe downstream flow")
    value: float = Field(default=0.0, description="Current step's calculated reward value")
    step_number: int = Field(default=0, description="Current step within the episode (0–20)")
