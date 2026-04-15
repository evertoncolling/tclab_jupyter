from __future__ import annotations

from typing import Any, cast

import numpy as np

from tclab_jupyter.backends import BackendSnapshot, SimulationBackend
from tclab_jupyter.runtime import ControlRuntime, RuntimeConfig


def test_runtime_keeps_bounded_history() -> None:
    runtime = ControlRuntime(
        SimulationBackend(seed=42),
        RuntimeConfig(delta_t=4.0, history_seconds=12.0),
    )
    runtime.initialize(0.0, 0.0, 30.0, 30.0)
    for _ in range(5):
        snapshot = runtime.step_manual(0.0, 0.0)
    assert len(snapshot.time_s) == 3
    assert snapshot.time_s[-1] == 20.0


def test_runtime_uses_backend_reported_elapsed_time() -> None:
    class FakeBackend:
        def initialize(self) -> BackendSnapshot:
            return BackendSnapshot(
                sensor_temperatures_k=np.array([295.15, 295.15]),
                heater_temperatures_k=np.array([295.15, 295.15]),
                elapsed_s=0.0,
            )

        def step(self, q1: float, q2: float, delta_t: float) -> BackendSnapshot:
            return BackendSnapshot(
                sensor_temperatures_k=np.array([296.15, 297.15]),
                heater_temperatures_k=np.array([296.15, 297.15]),
                elapsed_s=1.5,
            )

        def shutdown(self) -> None:
            return

    runtime = ControlRuntime(
        cast(Any, FakeBackend()),
        RuntimeConfig(delta_t=4.0, history_seconds=10.0),
    )
    runtime.initialize(0.0, 0.0, 30.0, 30.0)
    snapshot = runtime.step_manual(0.0, 0.0)
    assert snapshot.time_s[-1] == 1.5
