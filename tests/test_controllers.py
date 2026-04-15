from __future__ import annotations

import pytest

from tclab_jupyter.controllers import (
    ControllerSolveError,
    MpcConfig,
    MpcController,
    PIDState,
    on_off_output,
    pid_output,
)


def test_pid_output_matches_legacy_anti_windup_behavior() -> None:
    state = PIDState()
    output = pid_output(
        setpoint=30.0,
        process_value=20.0,
        previous_process_value=19.0,
        state=state,
        dt=4.0,
        kc=10.0,
        tau_i=50.0,
        tau_d=1.0,
    )
    assert output == pytest.approx(100.0)
    assert state.integral_error == pytest.approx(0.0)


def test_on_off_output_uses_hysteresis() -> None:
    assert on_off_output(24.0, 30.0, 0.1, 0.0) == 100.0
    assert on_off_output(31.0, 30.0, 0.1, 100.0) == 0.0
    assert on_off_output(30.05, 30.0, 0.1, 100.0) == 100.0


def test_mpc_controller_raises_when_solver_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = MpcController()

    def boom(*_args, **_kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(controller.model, "solve", boom)

    with pytest.raises(ControllerSolveError):
        controller.compute(22.0, 22.0, 30.0, 30.0, 4.0, MpcConfig())
