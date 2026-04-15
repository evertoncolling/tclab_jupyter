from __future__ import annotations

from tclab_jupyter.backends import SimulationBackend
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
