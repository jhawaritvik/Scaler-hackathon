# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed OpenEnv client for the wildfire environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import WildfireAction, WildfireObservation


class WildfireEnv(EnvClient[WildfireAction, WildfireObservation, State]):
    """WebSocket client for interacting with a running wildfire environment."""

    def _step_payload(self, action: WildfireAction) -> dict[str, Any]:
        """Serialize an action for the OpenEnv WebSocket step message."""
        return action.model_dump(mode="json")

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[WildfireObservation]:
        """Parse reset/step responses into a typed observation result."""
        observation_data = payload.get("observation", payload)
        observation = WildfireObservation.model_validate(observation_data)
        reward = payload.get("reward", observation.reward)
        done = payload.get("done", observation.done)
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: dict[str, Any]) -> State:
        """Parse the environment state response."""
        return State.model_validate(payload)
