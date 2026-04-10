# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hydropulse Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import HydropulseAction, HydropulseObservation


class HydropulseEnv(
    EnvClient[HydropulseAction, HydropulseObservation, State]
):
    """
    Client for the Hydropulse Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with HydropulseEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(HydropulseAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = HydropulseEnv.from_docker_image("HydroPulse-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(HydropulseAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: HydropulseAction) -> Dict:
        """
        Convert HydropulseAction to JSON payload for step message.

        Args:
            action: HydropulseAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "turbine_release": action.turbine_release,
            "spillway_release": action.spillway_release,
        }

    def _parse_result(self, payload: Dict) -> StepResult[HydropulseObservation]:
        """
        Parse server response into StepResult[HydropulseObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with HydropulseObservation
        """
        obs_data = payload.get("observation", {})
        observation = HydropulseObservation(
            reservoir_level=obs_data.get("reservoir_level", 0.0),
            inflow_rate=obs_data.get("inflow_rate", 0.0),
            grid_demand_price=obs_data.get("grid_demand_price", 0.0),
            downstream_capacity=obs_data.get("downstream_capacity", 40.0),
            value=obs_data.get("value", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
