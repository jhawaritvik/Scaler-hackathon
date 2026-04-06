# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wildfire environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import WildfireAction, WildfireObservation


class WildfireEnv(EnvClient[WildfireAction, WildfireObservation, State]):
    """Client for interacting with the wildfire environment."""

    def _step_payload(self, action: WildfireAction) -> Dict:
        """Convert a wildfire action into the JSON payload expected by the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[WildfireObservation]:
        """Parse a reset/step response into a typed observation."""
        obs_data = payload.get("observation", {})
        observation = WildfireObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", obs_data.get("done", False)),
                "reward": payload.get("reward", obs_data.get("reward")),
            }
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse the server-side environment state."""
        return State.model_validate(payload)
