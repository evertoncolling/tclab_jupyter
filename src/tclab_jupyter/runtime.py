"""Stateful runtime loop coordinating backends and controllers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .backends import HardwareBackend, SimulationBackend
from .controllers import (
    MpcConfig,
    MpcController,
    PIDState,
    on_off_output,
    pid_output,
)


@dataclass(slots=True)
class RuntimeConfig:
    delta_t: float = 4.0
    history_seconds: float = 500.0

    @property
    def max_points(self) -> int:
        return int(self.history_seconds / self.delta_t)


@dataclass(slots=True)
class RuntimeSnapshot:
    time_s: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    t1_c: np.ndarray
    t2_c: np.ndarray
    sp1_c: np.ndarray
    sp2_c: np.ndarray


@dataclass(slots=True)
class _History:
    history_seconds: float
    time_s: deque[float] = field(init=False)
    q1: deque[float] = field(init=False)
    q2: deque[float] = field(init=False)
    t1_c: deque[float] = field(init=False)
    t2_c: deque[float] = field(init=False)
    sp1_c: deque[float] = field(init=False)
    sp2_c: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.time_s = deque()
        self.q1 = deque()
        self.q2 = deque()
        self.t1_c = deque()
        self.t2_c = deque()
        self.sp1_c = deque()
        self.sp2_c = deque()

    def append(
        self,
        *,
        time_s: float,
        q1: float,
        q2: float,
        t1_c: float,
        t2_c: float,
        sp1_c: float,
        sp2_c: float,
    ) -> None:
        self.time_s.append(time_s)
        self.q1.append(q1)
        self.q2.append(q2)
        self.t1_c.append(t1_c)
        self.t2_c.append(t2_c)
        self.sp1_c.append(sp1_c)
        self.sp2_c.append(sp2_c)
        self.prune()

    def prune(self) -> None:
        if not self.time_s:
            return
        oldest_time = self.time_s[-1] - self.history_seconds
        while self.time_s and self.time_s[0] < oldest_time:
            self.time_s.popleft()
            self.q1.popleft()
            self.q2.popleft()
            self.t1_c.popleft()
            self.t2_c.popleft()
            self.sp1_c.popleft()
            self.sp2_c.popleft()

    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            time_s=np.asarray(self.time_s, dtype=float),
            q1=np.asarray(self.q1, dtype=float),
            q2=np.asarray(self.q2, dtype=float),
            t1_c=np.asarray(self.t1_c, dtype=float),
            t2_c=np.asarray(self.t2_c, dtype=float),
            sp1_c=np.asarray(self.sp1_c, dtype=float),
            sp2_c=np.asarray(self.sp2_c, dtype=float),
        )


class ControlRuntime:
    """Controller/backend orchestrator with bounded history."""

    def __init__(
        self,
        backend: SimulationBackend | HardwareBackend,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or RuntimeConfig()
        self.history = _History(self.config.history_seconds)
        self._elapsed_s = 0.0
        self._pid1 = PIDState()
        self._pid2 = PIDState()
        self._mpc = MpcController()

    def initialize(
        self, q1: float, q2: float, sp1_c: float, sp2_c: float
    ) -> RuntimeSnapshot:
        self._elapsed_s = 0.0
        self.history = _History(self.config.history_seconds)
        snapshot = self.backend.initialize()
        t1_c, t2_c = snapshot.sensor_temperatures_k - 273.15
        self.history.append(
            time_s=0.0,
            q1=q1,
            q2=q2,
            t1_c=float(t1_c),
            t2_c=float(t2_c),
            sp1_c=sp1_c,
            sp2_c=sp2_c,
        )
        return self.history.snapshot()

    def shutdown(self) -> None:
        self.backend.shutdown()

    def update_config(
        self,
        *,
        delta_t: float | None = None,
        history_seconds: float | None = None,
    ) -> None:
        if delta_t is not None:
            self.config.delta_t = delta_t
        if history_seconds is not None:
            self.config.history_seconds = history_seconds
        self.history.history_seconds = self.config.history_seconds
        self.history.prune()

    def _advance(
        self, q1: float, q2: float, sp1_c: float, sp2_c: float
    ) -> RuntimeSnapshot:
        backend_snapshot = self.backend.step(q1=q1, q2=q2, delta_t=self.config.delta_t)
        self._elapsed_s += backend_snapshot.elapsed_s
        t1_c, t2_c = backend_snapshot.sensor_temperatures_k - 273.15
        self.history.append(
            time_s=self._elapsed_s,
            q1=q1,
            q2=q2,
            t1_c=float(t1_c),
            t2_c=float(t2_c),
            sp1_c=sp1_c,
            sp2_c=sp2_c,
        )
        return self.history.snapshot()

    def step_manual(self, q1: float, q2: float) -> RuntimeSnapshot:
        return self._advance(
            q1=q1,
            q2=q2,
            sp1_c=self.history.sp1_c[-1],
            sp2_c=self.history.sp2_c[-1],
        )

    def step_on_off(
        self,
        q1: float,
        q2: float,
        sp1_c: float,
        sp2_c: float,
        deadband1: float,
        deadband2: float,
    ) -> tuple[RuntimeSnapshot, float, float]:
        last_t1 = self.history.t1_c[-1]
        last_t2 = self.history.t2_c[-1]
        next_q1 = on_off_output(last_t1, sp1_c, deadband1, q1)
        next_q2 = on_off_output(last_t2, sp2_c, deadband2, q2)
        return self._advance(next_q1, next_q2, sp1_c, sp2_c), next_q1, next_q2

    def step_pid(
        self,
        q1: float,
        q2: float,
        sp1_c: float,
        sp2_c: float,
        gain1: float,
        reset1: float,
        rate1: float,
        gain2: float,
        reset2: float,
        rate2: float,
    ) -> tuple[RuntimeSnapshot, float, float]:
        last_t1 = self.history.t1_c[-1]
        last_t2 = self.history.t2_c[-1]
        prev_t1 = self.history.t1_c[-2] if len(self.history.t1_c) > 1 else last_t1
        prev_t2 = self.history.t2_c[-2] if len(self.history.t2_c) > 1 else last_t2
        next_q1 = pid_output(
            sp1_c,
            last_t1,
            prev_t1,
            self._pid1,
            self.config.delta_t,
            gain1,
            reset1,
            rate1,
        )
        next_q2 = pid_output(
            sp2_c,
            last_t2,
            prev_t2,
            self._pid2,
            self.config.delta_t,
            gain2,
            reset2,
            rate2,
        )
        return self._advance(next_q1, next_q2, sp1_c, sp2_c), next_q1, next_q2

    def step_mpc(
        self,
        q1: float,
        q2: float,
        sp1_c: float,
        sp2_c: float,
        config: MpcConfig,
    ) -> tuple[RuntimeSnapshot, float, float]:
        last_t1 = self.history.t1_c[-1]
        last_t2 = self.history.t2_c[-1]
        next_q1, next_q2 = self._mpc.compute(
            measured_t1_c=last_t1,
            measured_t2_c=last_t2,
            setpoint_t1_c=sp1_c,
            setpoint_t2_c=sp2_c,
            delta_t=self.config.delta_t,
            config=config,
        )
        return self._advance(next_q1, next_q2, sp1_c, sp2_c), next_q1, next_q2
