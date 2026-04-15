"""Pure controller logic extracted from the notebook UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gekko import GEKKO


class ControllerSolveError(RuntimeError):
    """Raised when the MPC solver fails."""


@dataclass(slots=True)
class PIDState:
    """Mutable PID integrator state."""

    integral_error: float = 0.0


def pid_output(
    setpoint: float,
    process_value: float,
    previous_process_value: float,
    state: PIDState,
    dt: float,
    kc: float = 10.0,
    tau_i: float = 50.0,
    tau_d: float = 1.0,
) -> float:
    """Return the PID output and update the integral term in place."""

    kp = kc
    ki = 1e5 if tau_i == 0 else kc / tau_i
    kd = kc * tau_d
    lower = 0.0
    upper = 100.0

    error = setpoint - process_value
    candidate_integral = state.integral_error + ki * error * dt
    derivative = (process_value - previous_process_value) / dt
    output = kp * error + candidate_integral - kd * derivative

    if output < lower or output > upper:
        output = max(lower, min(upper, output))
    else:
        state.integral_error = candidate_integral

    return output


def on_off_output(
    process_value: float,
    setpoint: float,
    deadband: float,
    previous_output: float,
) -> float:
    """Return the next on/off actuator output with hysteresis."""

    if process_value < (setpoint - deadband):
        return 100.0
    if process_value > (setpoint + deadband):
        return 0.0
    return previous_output


@dataclass(slots=True)
class MpcConfig:
    """Configuration for the GEKKO MPC model."""

    solver: str = "1 - APOPT"
    cvtype: str = "1 - Deadband"
    t1_deadband: float = 0.1
    t1_tau: float = 10.0
    t2_deadband: float = 0.1
    t2_tau: float = 10.0
    q1_dmax: float = 30.0
    q1_dcost: float = 1.0
    q2_dmax: float = 30.0
    q2_dcost: float = 1.0


class MpcController:
    """Small wrapper around the legacy GEKKO MPC model."""

    def __init__(self) -> None:
        self._model = self._create_model()

    @staticmethod
    def _create_model() -> Any:
        model: Any = GEKKO(remote=False)
        model.time = [0, 4, 8, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90]

        model.U = model.FV(value=10)
        model.tau = model.FV(value=5)
        model.alpha1 = model.FV(value=0.01)
        model.alpha2 = model.FV(value=0.0075)

        model.Q1 = model.MV(value=0)
        model.Q1.STATUS = 1
        model.Q1.FSTATUS = 0
        model.Q1.LOWER = 0.0
        model.Q1.UPPER = 100.0
        model.Q1.DMAX = 20.0
        model.Q1.COST = 0.0
        model.Q1.DCOST = 2.0

        model.Q2 = model.MV(value=0)
        model.Q2.STATUS = 1
        model.Q2.FSTATUS = 0
        model.Q2.LOWER = 0.0
        model.Q2.UPPER = 100.0
        model.Q2.DMAX = 20.0
        model.Q2.COST = 0.0
        model.Q2.DCOST = 2.0

        model.TC1 = model.CV(value=22)
        model.TC1.STATUS = 1
        model.TC1.FSTATUS = 1
        model.TC1.TR_INIT = 1
        model.TC1.TAU = 10

        model.TC2 = model.CV(value=22)
        model.TC2.STATUS = 1
        model.TC2.FSTATUS = 1
        model.TC2.TR_INIT = 1
        model.TC2.TAU = 10

        model.TH1 = model.SV(value=22)
        model.TH2 = model.SV(value=22)

        model.Ta = model.Param(value=23.0 + 273.15)
        model.mass = model.Param(value=4.0 / 1000.0)
        model.Cp = model.Param(value=0.5 * 1000.0)
        model.A = model.Param(value=10.0 / 100.0**2)
        model.As = model.Param(value=2.0 / 100.0**2)
        model.eps = model.Param(value=0.9)
        model.sigma = model.Const(5.67e-8)

        model.T1i = model.Intermediate(model.TH1 + 273.15)
        model.T2i = model.Intermediate(model.TH2 + 273.15)
        model.Q_C12 = model.Intermediate(model.U * model.As * (model.T2i - model.T1i))
        model.Q_R12 = model.Intermediate(
            model.eps * model.sigma * model.As * (model.T2i**4 - model.T1i**4)
        )

        model.Equation(
            model.TH1.dt()
            == (1.0 / (model.mass * model.Cp))
            * (
                model.U * model.A * (model.Ta - model.T1i)
                + model.eps * model.sigma * model.A * (model.Ta**4 - model.T1i**4)
                + model.Q_C12
                + model.Q_R12
                + model.alpha1 * model.Q1
            )
        )
        model.Equation(
            model.TH2.dt()
            == (1.0 / (model.mass * model.Cp))
            * (
                model.U * model.A * (model.Ta - model.T2i)
                + model.eps * model.sigma * model.A * (model.Ta**4 - model.T2i**4)
                - model.Q_C12
                - model.Q_R12
                + model.alpha2 * model.Q2
            )
        )

        model.Equation(model.tau * model.TC1.dt() == -model.TC1 + model.TH1)
        model.Equation(model.tau * model.TC2.dt() == -model.TC2 + model.TH2)

        model.options.IMODE = 6
        model.options.CV_TYPE = 1
        model.options.NODES = 3
        model.options.SOLVER = 3

        return model

    @property
    def model(self) -> Any:
        return self._model

    def configure(self, delta_t: float, config: MpcConfig) -> None:
        solver_map = {"1 - APOPT": 1, "2 - BPOPT": 2, "3 - IPOPT": 3}
        self._model.options.SOLVER = solver_map.get(config.solver, 3)
        self._model.options.CV_TYPE = 1 if config.cvtype == "1 - Deadband" else 2
        self._model.TC1.TAU = config.t1_tau
        self._model.TC2.TAU = config.t2_tau
        self._model.Q1.DMAX = config.q1_dmax
        self._model.Q1.DCOST = config.q1_dcost
        self._model.Q2.DMAX = config.q2_dmax
        self._model.Q2.DCOST = config.q2_dcost
        dt = delta_t
        self._model.time = [
            0,
            dt,
            dt * 2,
            dt * 3,
            dt * 4,
            dt * 5,
            dt * 6,
            dt * 7,
            dt * 8,
            dt * 10,
            dt * 12,
            dt * 15,
            dt * 18,
            dt * 20,
            dt * 25,
        ]

    def compute(
        self,
        measured_t1_c: float,
        measured_t2_c: float,
        setpoint_t1_c: float,
        setpoint_t2_c: float,
        delta_t: float,
        config: MpcConfig,
    ) -> tuple[float, float]:
        self.configure(delta_t=delta_t, config=config)
        self._model.TC1.MEAS = measured_t1_c
        self._model.TC2.MEAS = measured_t2_c

        if self._model.options.CV_TYPE == 1:
            self._model.TC1.SPHI = setpoint_t1_c + config.t1_deadband
            self._model.TC1.SPLO = setpoint_t1_c - config.t1_deadband
            self._model.TC2.SPHI = setpoint_t2_c + config.t2_deadband
            self._model.TC2.SPLO = setpoint_t2_c - config.t2_deadband
        else:
            self._model.TC1.SP = setpoint_t1_c
            self._model.TC2.SP = setpoint_t2_c

        try:
            self._model.solve(disp=False)
        except Exception as exc:  # pragma: no cover - defensive path
            raise ControllerSolveError("GEKKO failed to solve the MPC problem") from exc

        if self._model.options.APPSTATUS != 1:
            raise ControllerSolveError("GEKKO did not report a successful solve")

        return float(self._model.Q1.NEWVAL), float(self._model.Q2.NEWVAL)
