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
    assert len(snapshot.time_s) == 4
    assert snapshot.time_s[0] == 8.0
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


def test_runtime_update_config_changes_delta_t_without_immediate_history_drop() -> None:
    runtime = ControlRuntime(
        SimulationBackend(seed=42),
        RuntimeConfig(delta_t=4.0, history_seconds=12.0),
    )
    runtime.initialize(0.0, 0.0, 30.0, 30.0)
    for _ in range(3):
        runtime.step_manual(0.0, 0.0)

    before = runtime.history.snapshot()
    runtime.update_config(delta_t=10.0)
    after_update = runtime.history.snapshot()
    snapshot = runtime.step_manual(0.0, 0.0)

    assert runtime.config.delta_t == 10.0
    assert np.array_equal(before.time_s, np.array([0.0, 4.0, 8.0, 12.0]))
    assert np.array_equal(after_update.time_s, before.time_s)
    assert np.array_equal(snapshot.time_s, np.array([12.0, 22.0]))
