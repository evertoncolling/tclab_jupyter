from __future__ import annotations

import numpy as np

from tclab_jupyter.backends import HardwareBackend, SimulationBackend


def test_simulation_backend_is_reproducible_with_seed() -> None:
    backend_a = SimulationBackend(seed=7)
    backend_b = SimulationBackend(seed=7)
    backend_a.initialize()
    backend_b.initialize()
    snap_a = backend_a.step(30.0, 10.0, 4.0)
    snap_b = backend_b.step(30.0, 10.0, 4.0)
    assert np.allclose(snap_a.sensor_temperatures_k, snap_b.sensor_temperatures_k)
    assert snap_a.elapsed_s == 4.0
    assert snap_b.elapsed_s == 4.0


def test_hardware_backend_can_be_mocked() -> None:
    class FakeLab:
        def __init__(self) -> None:
            self.T1 = 22.5
            self.T2 = 23.0
            self.q1 = None
            self.q2 = None
            self.closed = False

        def Q1(self, value: float) -> None:
            self.q1 = value

        def Q2(self, value: float) -> None:
            self.q2 = value

        def close(self) -> None:
            self.closed = True

    lab = FakeLab()
    clock_values = iter([100.0, 100.75])
    backend = HardwareBackend(
        lab_factory=lambda: lab,
        monotonic=lambda: next(clock_values),
    )
    initial = backend.initialize()
    assert initial.sensor_temperatures_k[0] == 22.5 + 273.15
    assert initial.elapsed_s == 0.0
    step = backend.step(12.0, 18.0, 4.0)
    assert lab.q1 == 12.0
    assert lab.q2 == 18.0
    assert step.elapsed_s == 0.75
    backend.shutdown()
    assert lab.closed is True
