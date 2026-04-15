"""Backend adapters for simulation and hardware."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from scipy.integrate import odeint


@dataclass(slots=True)
class SimulationParameters:
    ambient_temperature_k: float = 23 + 273.15
    mass_kg: float = 4.0 / 1000.0
    heat_capacity_j_per_kg_k: float = 0.5 * 1000.0
    area_m2: float = 10.0 / 100.0**2
    exchange_area_m2: float = 2.0 / 100.0**2
    emissivity: float = 0.9
    sigma: float = 5.67e-8
    alpha1: float = 0.00640897365
    alpha2: float = 0.00310952441
    sensor_tau: float = 17.7176964
    convection_base: float = 4.87519009


@dataclass(slots=True)
class BackendSnapshot:
    sensor_temperatures_k: np.ndarray
    heater_temperatures_k: np.ndarray | None = None


class TCLabLike(Protocol):
    T1: float
    T2: float

    def Q1(self, value: float) -> None: ...

    def Q2(self, value: float) -> None: ...

    def close(self) -> None: ...


class SimulationBackend:
    """Legacy nonlinear simulated plant extracted from the demo script."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        parameters: SimulationParameters | None = None,
    ) -> None:
        self.parameters = parameters or SimulationParameters()
        self._rng = np.random.default_rng(seed)
        self.heater_temperatures_k = np.array([293.15, 293.15], dtype=float)
        self.sensor_temperatures_k = np.array([293.15, 293.15], dtype=float)

    def initialize(self) -> BackendSnapshot:
        return BackendSnapshot(
            sensor_temperatures_k=self.sensor_temperatures_k.copy(),
            heater_temperatures_k=self.heater_temperatures_k.copy(),
        )

    def _heater(self, x: np.ndarray, t: float, q1: float, q2: float) -> list[float]:
        p = self.parameters
        u = p.convection_base + (float(self._rng.random()) - 0.5)
        th1, th2 = x
        conv12 = u * p.exchange_area_m2 * (th2 - th1)
        rad12 = p.emissivity * p.sigma * p.exchange_area_m2 * (th2**4 - th1**4)
        dth1dt = (1.0 / (p.mass_kg * p.heat_capacity_j_per_kg_k)) * (
            u * p.area_m2 * (p.ambient_temperature_k - th1)
            + p.emissivity * p.sigma * p.area_m2 * (p.ambient_temperature_k**4 - th1**4)
            + conv12
            + rad12
            + p.alpha1 * q1
        )
        dth2dt = (1.0 / (p.mass_kg * p.heat_capacity_j_per_kg_k)) * (
            u * p.area_m2 * (p.ambient_temperature_k - th2)
            + p.emissivity * p.sigma * p.area_m2 * (p.ambient_temperature_k**4 - th2**4)
            - conv12
            - rad12
            + p.alpha2 * q2
        )
        return [dth1dt, dth2dt]

    def _sensor(
        self, x: np.ndarray, t: float, th1: float, th2: float
    ) -> list[float]:
        tc1, tc2 = x
        tau = self.parameters.sensor_tau
        return [(-tc1 + th1) / tau, (-tc2 + th2) / tau]

    def step(self, q1: float, q2: float, delta_t: float) -> BackendSnapshot:
        ts = [0.0, delta_t]
        heater_state = odeint(
            self._heater,
            self.heater_temperatures_k,
            ts,
            args=(q1, q2),
        )
        self.heater_temperatures_k = np.asarray(heater_state[-1], dtype=float)
        sensor_state = odeint(
            self._sensor,
            self.sensor_temperatures_k,
            ts,
            args=(self.heater_temperatures_k[0], self.heater_temperatures_k[1]),
        )
        self.sensor_temperatures_k = np.asarray(sensor_state[-1], dtype=float)
        noise = np.array(
            [
                float(self._rng.random()) - 0.5,
                float(self._rng.random()) - 0.5,
            ],
            dtype=float,
        )
        return BackendSnapshot(
            sensor_temperatures_k=self.sensor_temperatures_k + noise,
            heater_temperatures_k=self.heater_temperatures_k.copy(),
        )

    def shutdown(self) -> None:
        return None


class HardwareBackend:
    """Thin adapter around the optional `tclab` hardware dependency."""

    def __init__(
        self,
        lab_factory: Callable[[], TCLabLike] | None = None,
    ) -> None:
        if lab_factory is None:
            try:
                module = importlib.import_module("tclab")
            except ImportError as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    "Hardware support requires the optional 'hardware' extra."
                ) from exc
            lab_factory = cast(Callable[[], TCLabLike], module.TCLab)
        self._lab_factory = lab_factory
        self._lab: TCLabLike | None = None

    def initialize(self) -> BackendSnapshot:
        self._lab = self._lab_factory()
        return BackendSnapshot(sensor_temperatures_k=self._read_temperatures())

    def _require_lab(self) -> TCLabLike:
        if self._lab is None:
            raise RuntimeError("Hardware backend has not been initialized")
        return self._lab

    def _read_temperatures(self) -> np.ndarray:
        lab = self._require_lab()
        return np.array([lab.T1 + 273.15, lab.T2 + 273.15], dtype=float)

    def step(self, q1: float, q2: float, delta_t: float) -> BackendSnapshot:
        lab = self._require_lab()
        lab.Q1(q1)
        lab.Q2(q2)
        return BackendSnapshot(sensor_temperatures_k=self._read_temperatures())

    def shutdown(self) -> None:
        lab = self._lab
        if lab is None:
            return
        lab.Q1(0)
        lab.Q2(0)
        lab.close()
        self._lab = None
