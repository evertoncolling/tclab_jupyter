"""Notebook UI wrappers around the extracted runtime."""

from __future__ import annotations

import importlib
import queue
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from ..backends import HardwareBackend, SimulationBackend
from ..controllers import ControllerSolveError, MpcConfig
from ..runtime import ControlRuntime, RuntimeConfig, RuntimeSnapshot

Mode = Literal["Manual", "On-Off", "PID", "MPC"]


def _load_widget_modules() -> tuple[Any, Any, Any]:
    display = importlib.import_module("IPython.display").display
    bq = importlib.import_module("bqplot")
    wi = importlib.import_module("ipywidgets").widgets
    return wi, bq, display


def _create_periodic_callback(callback: Callable[[], None], interval_ms: int) -> Any:
    periodic_callback = importlib.import_module("tornado.ioloop").PeriodicCallback
    return periodic_callback(callback, interval_ms)


def _coerce_float(value: object) -> float:
    return float(cast(float | int | str, value))


def _coerce_mode(value: object) -> Mode:
    if value not in {"Manual", "On-Off", "PID", "MPC"}:
        raise ValueError(f"Unsupported notebook mode: {value!r}")
    return cast(Mode, value)


class RuntimeLike(Protocol):
    def initialize(
        self, q1: float, q2: float, sp1_c: float, sp2_c: float
    ) -> RuntimeSnapshot: ...

    def shutdown(self) -> None: ...

    def update_config(
        self,
        *,
        delta_t: float | None = None,
        history_seconds: float | None = None,
    ) -> None: ...

    def step_manual(self, q1: float, q2: float) -> RuntimeSnapshot: ...

    def step_on_off(
        self,
        q1: float,
        q2: float,
        sp1_c: float,
        sp2_c: float,
        deadband1: float,
        deadband2: float,
    ) -> tuple[RuntimeSnapshot, float, float]: ...

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
    ) -> tuple[RuntimeSnapshot, float, float]: ...

    def step_mpc(
        self,
        q1: float,
        q2: float,
        sp1_c: float,
        sp2_c: float,
        config: MpcConfig,
    ) -> tuple[RuntimeSnapshot, float, float]: ...


@dataclass(slots=True)
class NotebookSessionState:
    delta_t: float = 4.0
    history_seconds: float = 500.0
    sleep_s: float = 0.5
    mode: Mode = "Manual"
    t1_sp: float = 30.0
    t2_sp: float = 30.0
    q1: float = 0.0
    q2: float = 0.0
    suspend_control_updates: bool = False
    last_rendered_snapshot: RuntimeSnapshot | None = None
    worker_error: BaseException | None = None
    q1_dt_on_off: float = 0.1
    q2_dt_on_off: float = 0.1
    pid1_gain: float = 10.0
    pid1_reset: float = 50.0
    pid1_rate: float = 1.0
    pid2_gain: float = 10.0
    pid2_reset: float = 50.0
    pid2_rate: float = 1.0
    solver: str = "1 - APOPT"
    cvtype: str = "1 - Deadband"
    t1_dt: float = 0.1
    t1_tau: float = 10.0
    t2_dt: float = 0.1
    t2_tau: float = 10.0
    q1_dmax: float = 30.0
    q1_dcost: float = 1.0
    q2_dmax: float = 30.0
    q2_dcost: float = 1.0

    @contextmanager
    def suspended_updates(self):
        self.suspend_control_updates = True
        try:
            yield
        finally:
            self.suspend_control_updates = False

    @property
    def time_window_min(self) -> float:
        return self.history_seconds / 60.0

    @property
    def is_manual_mode(self) -> bool:
        return self.mode == "Manual"

    def runtime_config(self) -> RuntimeConfig:
        return RuntimeConfig(delta_t=self.delta_t, history_seconds=self.history_seconds)

    def mpc_config(self) -> MpcConfig:
        return MpcConfig(
            solver=self.solver,
            cvtype=self.cvtype,
            t1_deadband=self.t1_dt,
            t1_tau=self.t1_tau,
            t2_deadband=self.t2_dt,
            t2_tau=self.t2_tau,
            q1_dmax=self.q1_dmax,
            q1_dcost=self.q1_dcost,
            q2_dmax=self.q2_dmax,
            q2_dcost=self.q2_dcost,
        )

    def reset_for_mode(self, mode: Mode) -> None:
        self.mode = mode
        self.t1_sp = 30.0
        self.t2_sp = 30.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.worker_error = None
        self.last_rendered_snapshot = None


@dataclass(slots=True)
class NotebookWidgets:
    app_root: Any
    config_root: Any
    mode: Any
    start_button: Any
    stop_button: Any
    time_scale: Any
    t1_meas: Any
    t2_meas: Any
    t1_set_point: Any
    t2_set_point: Any
    u1: Any
    u2: Any
    pt1: Any
    lt1: Any
    pt2: Any
    lt2: Any
    q1_slider: Any
    q1_text: Any
    q1_button: Any
    q2_slider: Any
    q2_text: Any
    q2_button: Any
    t1_slider: Any
    t1_text: Any
    t1_button: Any
    t2_slider: Any
    t2_text: Any
    t2_button: Any
    general_delta: Any
    general_sleep: Any
    onoff_q1_deadband: Any
    onoff_q2_deadband: Any
    pid1_gain: Any
    pid1_reset: Any
    pid1_rate: Any
    pid2_gain: Any
    pid2_reset: Any
    pid2_rate: Any
    mpc_solver: Any
    mpc_cvtype: Any
    mpc_t1_dt: Any
    mpc_t1_tau: Any
    mpc_t2_dt: Any
    mpc_t2_tau: Any
    mpc_q1_dmax: Any
    mpc_q1_dcost: Any
    mpc_q2_dmax: Any
    mpc_q2_dcost: Any


@dataclass(slots=True)
class NotebookView:
    wi: Any
    bq: Any
    widgets: NotebookWidgets | None = None
    _colors: list[str] = field(
        default_factory=lambda: [
            "#1f77b4",
            "#5ebdff",
            "#d62728",
            "#ff7f0e",
            "#1f77b4",
            "#d62728",
        ]
    )

    def build(self, state: NotebookSessionState) -> NotebookWidgets:
        style = {"description_width": "initial"}
        time_scale = self.bq.LinearScale(min=0.0, max=state.time_window_min)
        temperature_scale = self.bq.LinearScale(min=20.0, max=60.0)
        heater_scale = self.bq.LinearScale(min=0.0, max=100.0)
        temperature_scale_set = {"x": time_scale, "y": temperature_scale}
        heater_scale_set = {"x": time_scale, "y": heater_scale}

        ax_x1 = self.bq.Axis(label="", scale=time_scale, visible=False)
        ax_x2 = self.bq.Axis(label="time [min]", scale=time_scale)
        ax_t1 = self.bq.Axis(
            label="Temperature 1 [C]",
            scale=temperature_scale,
            orientation="vertical",
            label_color=self._colors[0],
        )
        ax_t2 = self.bq.Axis(
            label="Temperature 2 [C]",
            scale=temperature_scale,
            orientation="vertical",
            label_color=self._colors[2],
        )
        ax_q = self.bq.Axis(
            label="Heater Output [%]",
            scale=heater_scale,
            orientation="vertical",
            label_color="black",
        )

        t1_meas = self.bq.Scatter(
            x=[],
            y=[],
            scales=temperature_scale_set,
            marker="circle",
            colors=[self._colors[0]],
            default_size=20,
        )
        t2_meas = self.bq.Scatter(
            x=[],
            y=[],
            scales=temperature_scale_set,
            marker="circle",
            colors=[self._colors[2]],
            default_size=20,
        )
        t1_set_point = self.bq.Lines(
            x=[],
            y=[],
            scales=temperature_scale_set,
            stroke_width=4,
            colors=[self._colors[1]],
            interpolation="step-before",
        )
        t2_set_point = self.bq.Lines(
            x=[],
            y=[],
            scales=temperature_scale_set,
            stroke_width=4,
            colors=[self._colors[3]],
            interpolation="step-before",
        )
        u1 = self.bq.Lines(
            x=[],
            y=[],
            scales=heater_scale_set,
            stroke_width=2,
            colors=[self._colors[4]],
            interpolation="step-before",
            display_legend=True,
            labels=["Heater 1"],
        )
        u2 = self.bq.Lines(
            x=[],
            y=[],
            scales=heater_scale_set,
            stroke_width=2,
            colors=[self._colors[5]],
            interpolation="step-before",
            display_legend=True,
            labels=["Heater 2"],
        )

        fig_up = self.wi.Layout(width="400px", height="240px")
        fig_down = self.wi.Layout(width="400px", height="290px")
        fig1 = self.bq.Figure(
            layout=fig_up,
            axes=[ax_x1, ax_t1],
            marks=[t1_meas, t1_set_point],
            fig_margin=dict(top=10, bottom=0, left=60, right=5),
        )
        fig2 = self.bq.Figure(
            layout=fig_up,
            axes=[ax_x1, ax_t2],
            marks=[t2_meas, t2_set_point],
            fig_margin=dict(top=10, bottom=0, left=60, right=20),
        )
        fig3 = self.bq.Figure(
            layout=fig_down,
            axes=[ax_x2, ax_q],
            marks=[u1, u2],
            fig_margin=dict(top=10, bottom=50, left=60, right=10),
            legend_location="left",
            legend_style={"fill": "white", "fill-opacity": "0.7", "width": "130px"},
        )

        v_space = self.wi.Label(value="", layout=self.wi.Layout(height="2px"))
        h_space = self.wi.Label(value="", layout=self.wi.Layout(width="2px"))

        pt1 = self.wi.FloatProgress(
            value=20.0,
            min=0,
            max=100.0,
            description="PV (°C):",
            bar_style="warning",
            orientation="horizontal",
            style=style,
        )
        lt1 = self.wi.Label(value=self.format_temperature_label(pt1.value))
        pt2 = self.wi.FloatProgress(
            value=20.0,
            min=0,
            max=100.0,
            description="PV (°C):",
            bar_style="warning",
            orientation="horizontal",
            style=style,
        )
        lt2 = self.wi.Label(value=self.format_temperature_label(pt2.value))

        q1_slider = self.wi.FloatSlider(
            value=state.q1,
            min=0,
            max=100.0,
            step=0.5,
            description="Q1 (%):",
            continuous_update=False,
            orientation="horizontal",
            readout=False,
            style=style,
            layout=self.wi.Layout(width="230px"),
        )
        q1_text = self.wi.BoundedFloatText(
            value=state.q1,
            min=0,
            max=100.0,
            step=0.5,
            layout=self.wi.Layout(width="60px"),
        )
        self.wi.jslink((q1_slider, "value"), (q1_text, "value"))
        q1_button = self.wi.Button(description="Set", layout=self.wi.Layout(width="50px"))

        q2_slider = self.wi.FloatSlider(
            value=state.q2,
            min=0,
            max=100.0,
            step=0.5,
            description="Q2 (%):",
            continuous_update=False,
            orientation="horizontal",
            readout=False,
            style=style,
            layout=self.wi.Layout(width="230px"),
        )
        q2_text = self.wi.BoundedFloatText(
            value=state.q2,
            min=0,
            max=100.0,
            step=0.5,
            layout=self.wi.Layout(width="60px"),
        )
        self.wi.jslink((q2_slider, "value"), (q2_text, "value"))
        q2_button = self.wi.Button(description="Set", layout=self.wi.Layout(width="50px"))

        t1_slider = self.wi.FloatSlider(
            value=state.t1_sp,
            min=20,
            max=60.0,
            step=0.5,
            description="T1 SP:",
            continuous_update=False,
            orientation="horizontal",
            readout=False,
            style=style,
            disabled=True,
            layout=self.wi.Layout(width="230px"),
        )
        t1_text = self.wi.BoundedFloatText(
            value=state.t1_sp,
            min=20,
            max=60.0,
            step=0.5,
            disabled=True,
            layout=self.wi.Layout(width="60px"),
        )
        self.wi.jslink((t1_slider, "value"), (t1_text, "value"))
        t1_button = self.wi.Button(
            description="Set",
            layout=self.wi.Layout(width="50px"),
            disabled=True,
        )

        t2_slider = self.wi.FloatSlider(
            value=state.t2_sp,
            min=20,
            max=60.0,
            step=0.5,
            description="T2 SP:",
            continuous_update=False,
            orientation="horizontal",
            readout=False,
            style=style,
            disabled=True,
            layout=self.wi.Layout(width="230px"),
        )
        t2_text = self.wi.BoundedFloatText(
            value=state.t2_sp,
            min=20,
            max=60.0,
            step=0.5,
            disabled=True,
            layout=self.wi.Layout(width="60px"),
        )
        self.wi.jslink((t2_slider, "value"), (t2_text, "value"))
        t2_button = self.wi.Button(
            description="Set",
            layout=self.wi.Layout(width="50px"),
            disabled=True,
        )

        mode = self.wi.ToggleButtons(
            options=["Manual", "On-Off", "PID", "MPC"],
            style={"button_width": "100px"},
        )
        start_button = self.wi.Button(
            description="Start",
            button_style="success",
            icon="play",
            layout=self.wi.Layout(width="100px", height="32px"),
        )
        stop_button = self.wi.Button(
            description="Stop",
            button_style="warning",
            icon="stop",
            layout=self.wi.Layout(width="100px", height="32px"),
        )

        buttons = self.wi.HBox(
            (
                start_button,
                h_space,
                stop_button,
                self.wi.Label(value="", layout=self.wi.Layout(width="165px")),
                mode,
            )
        )
        q1_set = self.wi.HBox(
            (h_space, q1_slider, q1_text, q1_button),
            layout=self.wi.Layout(border="solid 2px gray", width="360px"),
        )
        t1_set = self.wi.HBox(
            (h_space, t1_slider, t1_text, t1_button),
            layout=self.wi.Layout(border="solid 2px gray", width="360px"),
        )
        t1_view = self.wi.HBox(
            (h_space, pt1, lt1),
            layout=self.wi.Layout(border="solid 2px gray", width="360px"),
        )
        q2_set = self.wi.HBox(
            (h_space, q2_slider, q2_text, q2_button),
            layout=self.wi.Layout(border="solid 2px gray", width="360px"),
        )
        t2_set = self.wi.HBox(
            (h_space, t2_slider, t2_text, t2_button),
            layout=self.wi.Layout(border="solid 2px gray", width="360px"),
        )
        t2_view = self.wi.HBox(
            (h_space, pt2, lt2),
            layout=self.wi.Layout(border="solid 2px gray", width="360px"),
        )
        co1 = self.wi.VBox((t1_set, v_space, t1_view, v_space, q1_set))
        co2 = self.wi.VBox((t2_set, v_space, t2_view, v_space, q2_set))
        controls = self.wi.VBox((v_space, v_space, co1, v_space, v_space, v_space, co2))
        figures = self.wi.VBox(
            (self.wi.HBox((fig1, fig2)), v_space, self.wi.HBox((fig3, h_space, h_space, controls))),
            layout=self.wi.Layout(border="solid 2px gray", width="800px"),
        )
        app_root = self.wi.VBox((buttons, v_space, figures))

        general_delta = self.wi.FloatSlider(
            value=state.delta_t,
            min=1.0,
            max=10.0,
            step=0.5,
            description="",
            style=style,
        )
        general_sleep = self.wi.FloatSlider(
            value=state.sleep_s,
            min=0.25,
            max=1.5,
            step=0.05,
            description="",
            style=style,
        )
        onoff_q1_deadband = self.wi.FloatSlider(
            value=state.q1_dt_on_off,
            min=0.0,
            max=2.0,
            step=0.1,
            description="",
            style=style,
            layout=self.wi.Layout(width="200px"),
        )
        onoff_q2_deadband = self.wi.FloatSlider(
            value=state.q2_dt_on_off,
            min=0.0,
            max=2.0,
            step=0.1,
            description="",
            style=style,
            layout=self.wi.Layout(width="200px"),
        )
        pid1_gain = self.wi.FloatSlider(
            value=state.pid1_gain,
            min=0.0,
            max=20.0,
            step=0.5,
            description="",
            style=style,
            layout=self.wi.Layout(width="200px"),
        )
        pid1_reset = self.wi.FloatSlider(
            value=state.pid1_reset,
            min=0.0,
            max=200.0,
            step=1.0,
            description="",
            style=style,
            layout=self.wi.Layout(width="200px"),
        )
        pid1_rate = self.wi.FloatSlider(
            value=state.pid1_rate,
            min=0.0,
            max=10.0,
            step=0.5,
            description="",
            style=style,
            layout=self.wi.Layout(width="200px"),
        )
        pid2_gain = self.wi.FloatSlider(
            value=state.pid2_gain,
            min=0.0,
            max=20.0,
            step=0.5,
            description="",
            style=style,
            layout=self.wi.Layout(width="200px"),
        )
        pid2_reset = self.wi.FloatSlider(
            value=state.pid2_reset,
            min=0.0,
            max=200.0,
            step=1.0,
            description="",
            style=style,
            layout=self.wi.Layout(width="200px"),
        )
        pid2_rate = self.wi.FloatSlider(
            value=state.pid2_rate,
            min=0.0,
            max=10.0,
            step=0.5,
            description="",
            style=style,
            layout=self.wi.Layout(width="200px"),
        )
        mpc_solver = self.wi.Dropdown(
            options=["1 - APOPT", "2 - BPOPT", "3 - IPOPT"],
            value=state.solver,
            layout=self.wi.Layout(width="100px"),
        )
        mpc_cvtype = self.wi.Dropdown(
            options=["1 - Deadband", "2 - Trajectory"],
            value=state.cvtype,
            layout=self.wi.Layout(width="130px"),
        )
        mpc_t1_dt = self.wi.FloatSlider(
            value=state.t1_dt,
            min=0.1,
            max=1.0,
            step=0.1,
            description="",
            style=style,
            layout=self.wi.Layout(width="250px"),
        )
        mpc_t1_tau = self.wi.FloatSlider(
            value=state.t1_tau,
            min=1.0,
            max=100.0,
            step=1.0,
            description="",
            style=style,
            layout=self.wi.Layout(width="250px"),
        )
        mpc_t2_dt = self.wi.FloatSlider(
            value=state.t2_dt,
            min=0.1,
            max=1.0,
            step=0.1,
            description="",
            style=style,
            layout=self.wi.Layout(width="250px"),
        )
        mpc_t2_tau = self.wi.FloatSlider(
            value=state.t2_tau,
            min=1.0,
            max=100.0,
            step=1.0,
            description="",
            style=style,
            layout=self.wi.Layout(width="250px"),
        )
        mpc_q1_dmax = self.wi.FloatSlider(
            value=state.q1_dmax,
            min=1.0,
            max=100.0,
            step=0.5,
            description="",
            style=style,
            layout=self.wi.Layout(width="250px"),
        )
        mpc_q1_dcost = self.wi.FloatSlider(
            value=state.q1_dcost,
            min=0.0,
            max=10.0,
            step=0.1,
            description="",
            style=style,
            layout=self.wi.Layout(width="250px"),
        )
        mpc_q2_dmax = self.wi.FloatSlider(
            value=state.q2_dmax,
            min=1.0,
            max=100.0,
            step=0.5,
            description="",
            style=style,
            layout=self.wi.Layout(width="250px"),
        )
        mpc_q2_dcost = self.wi.FloatSlider(
            value=state.q2_dcost,
            min=0.0,
            max=10.0,
            step=0.1,
            description="",
            style=style,
            layout=self.wi.Layout(width="250px"),
        )

        config_root = self.wi.Tab(
            [
                self.wi.VBox(
                    (
                        self._config_row(
                            "&Delta;t (s):",
                            general_delta,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                        self._config_row(
                            "Sleep time (s):",
                            general_sleep,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                    ),
                    layout=self.wi.Layout(align_items="flex-start", padding="20px 30px"),
                ),
                self.wi.VBox(
                    (
                        self._config_section_title("T1 Controller"),
                        self._config_row(
                            "Deadband (K):",
                            onoff_q1_deadband,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                        self._config_spacer(),
                        self._config_section_title("T2 Controller"),
                        self._config_row(
                            "Deadband (K):",
                            onoff_q2_deadband,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                    ),
                    layout=self.wi.Layout(align_items="flex-start", padding="20px 30px"),
                ),
                self.wi.VBox(
                    (
                        self._config_section_title("T1 Controller"),
                        self._config_row(
                            "Kc (K/%Heater):",
                            pid1_gain,
                            width="220px",
                            margin="0 20px 0 0",
                        ),
                        self._config_row(
                            "tauI (s):",
                            pid1_reset,
                            width="220px",
                            margin="0 20px 0 0",
                        ),
                        self._config_row(
                            "tauD (s):",
                            pid1_rate,
                            width="220px",
                            margin="0 20px 0 0",
                        ),
                        self._config_spacer(),
                        self._config_section_title("T2 Controller"),
                        self._config_row(
                            "Kc (K/%Heater):",
                            pid2_gain,
                            width="220px",
                            margin="0 20px 0 0",
                        ),
                        self._config_row(
                            "tauI (s):",
                            pid2_reset,
                            width="220px",
                            margin="0 20px 0 0",
                        ),
                        self._config_row(
                            "tauD (s):",
                            pid2_rate,
                            width="220px",
                            margin="0 20px 0 0",
                        ),
                    ),
                    layout=self.wi.Layout(align_items="flex-start", padding="20px 30px"),
                ),
                self.wi.VBox(
                    (
                        self.wi.HBox(
                            (
                                self._config_label(
                                    "SOLVER:",
                                    width="120px",
                                    margin="0 15px 0 0",
                                ),
                                mpc_solver,
                                self._config_label(
                                    "CVTYPE:",
                                    width="120px",
                                    margin="0 15px 0 25px",
                                ),
                                mpc_cvtype,
                            ),
                            layout=self.wi.Layout(
                                align_items="center",
                                width="100%",
                                justify_content="flex-start",
                            ),
                        ),
                        self._config_spacer(),
                        self._config_section_title("T1 Controller"),
                        self._config_row(
                            "Deadband (K):",
                            mpc_t1_dt,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                        self._config_row(
                            "TAU:",
                            mpc_t1_tau,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                        self._config_spacer(),
                        self._config_section_title("T2 Controller"),
                        self._config_row(
                            "Deadband (K):",
                            mpc_t2_dt,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                        self._config_row(
                            "TAU:",
                            mpc_t2_tau,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                        self._config_spacer(),
                        self._config_section_title("Q1 Move Limits"),
                        self._config_row(
                            "DMAX:",
                            mpc_q1_dmax,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                        self._config_row(
                            "DCOST:",
                            mpc_q1_dcost,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                        self._config_spacer(),
                        self._config_section_title("Q2 Move Limits"),
                        self._config_row(
                            "DMAX:",
                            mpc_q2_dmax,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                        self._config_row(
                            "DCOST:",
                            mpc_q2_dcost,
                            width="180px",
                            margin="0 20px 0 0",
                        ),
                    ),
                    layout=self.wi.Layout(align_items="flex-start", padding="20px 30px"),
                ),
            ],
            layout=self.wi.Layout(width="800px", height="380px"),
        )
        config_root.set_title(0, "General Options")
        config_root.set_title(1, "On-Off Options")
        config_root.set_title(2, "PID Options")
        config_root.set_title(3, "MPC Options")

        self.widgets = NotebookWidgets(
            app_root=app_root,
            config_root=config_root,
            mode=mode,
            start_button=start_button,
            stop_button=stop_button,
            time_scale=time_scale,
            t1_meas=t1_meas,
            t2_meas=t2_meas,
            t1_set_point=t1_set_point,
            t2_set_point=t2_set_point,
            u1=u1,
            u2=u2,
            pt1=pt1,
            lt1=lt1,
            pt2=pt2,
            lt2=lt2,
            q1_slider=q1_slider,
            q1_text=q1_text,
            q1_button=q1_button,
            q2_slider=q2_slider,
            q2_text=q2_text,
            q2_button=q2_button,
            t1_slider=t1_slider,
            t1_text=t1_text,
            t1_button=t1_button,
            t2_slider=t2_slider,
            t2_text=t2_text,
            t2_button=t2_button,
            general_delta=general_delta,
            general_sleep=general_sleep,
            onoff_q1_deadband=onoff_q1_deadband,
            onoff_q2_deadband=onoff_q2_deadband,
            pid1_gain=pid1_gain,
            pid1_reset=pid1_reset,
            pid1_rate=pid1_rate,
            pid2_gain=pid2_gain,
            pid2_reset=pid2_reset,
            pid2_rate=pid2_rate,
            mpc_solver=mpc_solver,
            mpc_cvtype=mpc_cvtype,
            mpc_t1_dt=mpc_t1_dt,
            mpc_t1_tau=mpc_t1_tau,
            mpc_t2_dt=mpc_t2_dt,
            mpc_t2_tau=mpc_t2_tau,
            mpc_q1_dmax=mpc_q1_dmax,
            mpc_q1_dcost=mpc_q1_dcost,
            mpc_q2_dmax=mpc_q2_dmax,
            mpc_q2_dcost=mpc_q2_dcost,
        )
        self.apply_mode(state, running=False)
        return self.widgets

    def render(self, snapshot: RuntimeSnapshot, state: NotebookSessionState) -> None:
        widgets = self.require_widgets()
        state.last_rendered_snapshot = snapshot
        time_min = snapshot.time_s / 60.0
        latest_time = float(time_min[-1]) if len(time_min) else 0.0
        widgets.time_scale.min = max(0.0, latest_time - state.time_window_min)
        widgets.time_scale.max = max(state.time_window_min, latest_time)
        widgets.t1_meas.x = time_min
        widgets.t1_meas.y = snapshot.t1_c
        widgets.t2_meas.x = time_min
        widgets.t2_meas.y = snapshot.t2_c
        widgets.t1_set_point.x = time_min
        widgets.t1_set_point.y = snapshot.sp1_c
        widgets.t2_set_point.x = time_min
        widgets.t2_set_point.y = snapshot.sp2_c
        widgets.u1.x = time_min
        widgets.u1.y = snapshot.q1
        widgets.u2.x = time_min
        widgets.u2.y = snapshot.q2
        widgets.pt1.value = round(float(snapshot.t1_c[-1]), 1)
        widgets.pt2.value = round(float(snapshot.t2_c[-1]), 1)
        widgets.lt1.value = self.format_temperature_label(widgets.pt1.value)
        widgets.lt2.value = self.format_temperature_label(widgets.pt2.value)
        if widgets.q1_slider.disabled:
            self._assign_value(state, widgets.q1_slider, round(float(snapshot.q1[-1]), 1))
        if widgets.q2_slider.disabled:
            self._assign_value(state, widgets.q2_slider, round(float(snapshot.q2[-1]), 1))

    def reset_for_mode(self, state: NotebookSessionState) -> None:
        widgets = self.require_widgets()
        for mark in (
            widgets.t1_meas,
            widgets.t2_meas,
            widgets.t1_set_point,
            widgets.t2_set_point,
            widgets.u1,
            widgets.u2,
        ):
            mark.x = []
            mark.y = []
        widgets.time_scale.min = 0.0
        widgets.time_scale.max = state.time_window_min
        widgets.pt1.value = 20.0
        widgets.pt2.value = 20.0
        widgets.lt1.value = self.format_temperature_label(widgets.pt1.value)
        widgets.lt2.value = self.format_temperature_label(widgets.pt2.value)
        with state.suspended_updates():
            widgets.t1_slider.value = state.t1_sp
            widgets.t2_slider.value = state.t2_sp
            widgets.q1_slider.value = state.q1
            widgets.q2_slider.value = state.q2
        self.apply_mode(state, running=False)

    def apply_mode(self, state: NotebookSessionState, *, running: bool) -> None:
        widgets = self.require_widgets()
        manual = state.is_manual_mode
        widgets.mode.value = state.mode
        widgets.mode.disabled = running
        for widget in (
            widgets.q1_slider,
            widgets.q1_text,
            widgets.q1_button,
            widgets.q2_slider,
            widgets.q2_text,
            widgets.q2_button,
        ):
            widget.disabled = not manual
        for widget in (
            widgets.t1_slider,
            widgets.t1_text,
            widgets.t1_button,
            widgets.t2_slider,
            widgets.t2_text,
            widgets.t2_button,
        ):
            widget.disabled = manual

    def require_widgets(self) -> NotebookWidgets:
        if self.widgets is None:
            raise RuntimeError("Notebook widgets have not been built")
        return self.widgets

    @staticmethod
    def format_temperature_label(value: float) -> str:
        return f"{value:.1f}"

    def _assign_value(self, state: NotebookSessionState, widget: Any, value: float) -> None:
        with state.suspended_updates():
            widget.value = value

    def _config_label(self, text: str, *, width: str, margin: str) -> Any:
        return self.wi.HTML(
            value=(
                f'<p style="margin: 0; text-align: right; line-height: 32px; white-space: nowrap;">'
                f"<b>{text}</b></p>"
            ),
            layout=self.wi.Layout(
                width=width,
                min_width=width,
                margin=margin,
                display="flex",
                align_items="center",
                justify_content="flex-end",
            ),
        )

    def _config_section_title(self, text: str) -> Any:
        return self.wi.HTML(
            value=(
                f'<p style="margin: 0 0 6px 0; text-align: left; '
                f'font-size: 16px; font-weight: 700;">{text}</p>'
            ),
            layout=self.wi.Layout(width="100%"),
        )

    def _config_spacer(self) -> Any:
        return self.wi.HTML(
            value="",
            layout=self.wi.Layout(height="12px"),
        )

    def _config_row(self, label: str, widget: Any, *, width: str, margin: str) -> Any:
        return self.wi.HBox(
            (self._config_label(label, width=width, margin=margin), widget),
            layout=self.wi.Layout(
                align_items="center",
                width="100%",
                justify_content="flex-start",
            ),
        )


class NotebookRunner:
    def __init__(
        self,
        *,
        state: NotebookSessionState,
        backend_factory: Callable[[], SimulationBackend | HardwareBackend],
        on_snapshot: Callable[[RuntimeSnapshot], None],
        on_running_change: Callable[[bool], None],
        runtime_factory: Callable[[RuntimeConfig], RuntimeLike] | None = None,
        periodic_callback_factory: Callable[[Callable[[], None], int], Any]
        | None = None,
    ) -> None:
        self._state = state
        self._backend_factory = backend_factory
        self._on_snapshot = on_snapshot
        self._on_running_change = on_running_change
        self._runtime_factory = runtime_factory or self._default_runtime_factory
        self._periodic_callback_factory = (
            periodic_callback_factory or _create_periodic_callback
        )
        self._runtime: RuntimeLike | None = None
        self._worker: threading.Thread | None = None
        self._poller: Any = None
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._stop_event = threading.Event()

    @property
    def worker_error(self) -> BaseException | None:
        return self._state.worker_error

    @property
    def is_running(self) -> bool:
        worker = self._worker
        return worker is not None and worker.is_alive()

    def start(self, mode: Mode) -> None:
        if self.is_running:
            return
        self._clear_queue()
        self._initialize_runtime()
        self._state.worker_error = None
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run_worker, args=(mode,))
        self._worker.start()
        self._on_running_change(True)
        self._start_poller()

    def stop(self) -> None:
        if self._worker is None and self._runtime is None:
            return
        self._stop_event.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=2.0)
        if worker is None or not worker.is_alive():
            self.poll()
        else:
            self._start_poller()

    def poll(self) -> None:
        while True:
            try:
                event_type, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if event_type == "snapshot":
                self._on_snapshot(cast(RuntimeSnapshot, payload))
            elif event_type == "error":
                self._state.worker_error = cast(BaseException, payload)
                self._stop_event.set()
        if not self.is_running:
            self._finalize()

    def shutdown(self) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=2.0)
        self._stop_poller()
        self._shutdown_runtime()
        self._worker = None
        self._clear_queue()
        self._on_running_change(False)

    def invalidate_runtime(self) -> None:
        if self.is_running:
            return
        self._shutdown_runtime()

    def update_runtime_config(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        runtime.update_config(
            delta_t=self._state.delta_t,
            history_seconds=self._state.history_seconds,
        )

    def _initialize_runtime(self) -> None:
        if self._runtime is not None:
            return
        self._runtime = self._runtime_factory(self._state.runtime_config())
        initial = self._runtime.initialize(
            self._state.q1,
            self._state.q2,
            self._state.t1_sp,
            self._state.t2_sp,
        )
        self._on_snapshot(initial)

    def _run_worker(self, mode: Mode) -> None:
        try:
            self._run_mode(mode)
        except ControllerSolveError as exc:
            self._stop_event.set()
            self._queue.put(("error", exc))
        except Exception as exc:  # pragma: no cover - defensive path
            self._stop_event.set()
            self._queue.put(("error", exc))

    def _run_mode(self, mode: Mode) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        if mode == "Manual":
            self._loop_manual(runtime)
            return
        if mode == "On-Off":
            self._loop_on_off(runtime)
            return
        if mode == "PID":
            self._loop_pid(runtime)
            return
        self._loop_mpc(runtime)

    def _loop_manual(self, runtime: RuntimeLike) -> None:
        while not self._stop_event.is_set():
            snapshot = runtime.step_manual(self._state.q1, self._state.q2)
            self._queue.put(("snapshot", snapshot))
            if self._stop_event.wait(self._state.sleep_s):
                break

    def _loop_on_off(self, runtime: RuntimeLike) -> None:
        while not self._stop_event.is_set():
            snapshot, self._state.q1, self._state.q2 = runtime.step_on_off(
                self._state.q1,
                self._state.q2,
                self._state.t1_sp,
                self._state.t2_sp,
                self._state.q1_dt_on_off,
                self._state.q2_dt_on_off,
            )
            self._queue.put(("snapshot", snapshot))
            if self._stop_event.wait(self._state.sleep_s):
                break

    def _loop_pid(self, runtime: RuntimeLike) -> None:
        while not self._stop_event.is_set():
            snapshot, self._state.q1, self._state.q2 = runtime.step_pid(
                self._state.q1,
                self._state.q2,
                self._state.t1_sp,
                self._state.t2_sp,
                self._state.pid1_gain,
                self._state.pid1_reset,
                self._state.pid1_rate,
                self._state.pid2_gain,
                self._state.pid2_reset,
                self._state.pid2_rate,
            )
            self._queue.put(("snapshot", snapshot))
            if self._stop_event.wait(self._state.sleep_s):
                break

    def _loop_mpc(self, runtime: RuntimeLike) -> None:
        while not self._stop_event.is_set():
            config = self._state.mpc_config()
            snapshot, self._state.q1, self._state.q2 = runtime.step_mpc(
                self._state.q1,
                self._state.q2,
                self._state.t1_sp,
                self._state.t2_sp,
                config,
            )
            self._queue.put(("snapshot", snapshot))
            if self._stop_event.wait(self._state.sleep_s):
                break

    def _start_poller(self) -> None:
        if self._poller is not None:
            return
        self._poller = self._periodic_callback_factory(self.poll, 50)
        self._poller.start()

    def _stop_poller(self) -> None:
        if self._poller is None:
            return
        self._poller.stop()
        self._poller = None

    def _finalize(self) -> None:
        self._stop_poller()
        worker = self._worker
        if worker is not None and worker.is_alive():
            return
        self._worker = None
        self._shutdown_runtime()
        self._on_running_change(False)

    def _shutdown_runtime(self) -> None:
        if self._runtime is None:
            return
        self._runtime.shutdown()
        self._runtime = None

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _default_runtime_factory(self, config: RuntimeConfig) -> RuntimeLike:
        return ControlRuntime(self._backend_factory(), config)


class _NotebookApp:
    def __init__(
        self,
        backend_factory: Callable[[], SimulationBackend | HardwareBackend],
    ) -> None:
        wi, bq, display = _load_widget_modules()
        self._display = display
        self.state = NotebookSessionState()
        self.view = NotebookView(wi=wi, bq=bq)
        self.widgets = self.view.build(self.state)
        self.runner = NotebookRunner(
            state=self.state,
            backend_factory=backend_factory,
            on_snapshot=self._render_snapshot,
            on_running_change=self._set_running,
        )
        self._bind_events()
        self.view.reset_for_mode(self.state)

    def app(self) -> None:
        self._display(self.widgets.app_root)

    def config(self) -> None:
        self._display(self.widgets.config_root)

    def _bind_events(self) -> None:
        self.widgets.mode.observe(self._handle_mode_change, names="value")
        self.widgets.start_button.on_click(self._handle_start)
        self.widgets.stop_button.on_click(self._handle_stop)
        self.widgets.q1_button.on_click(self._handle_q1_button)
        self.widgets.q2_button.on_click(self._handle_q2_button)
        self.widgets.t1_button.on_click(self._handle_t1_button)
        self.widgets.t2_button.on_click(self._handle_t2_button)
        self.widgets.q1_slider.observe(self._sync_q1, names="value")
        self.widgets.q2_slider.observe(self._sync_q2, names="value")
        self.widgets.t1_slider.observe(self._sync_t1, names="value")
        self.widgets.t2_slider.observe(self._sync_t2, names="value")
        self.widgets.general_delta.observe(self._sync_general_delta, names="value")
        self.widgets.general_sleep.observe(self._sync_general_sleep, names="value")
        self.widgets.onoff_q1_deadband.observe(
            lambda change: setattr(self.state, "q1_dt_on_off", change["new"]),
            names="value",
        )
        self.widgets.onoff_q2_deadband.observe(
            lambda change: setattr(self.state, "q2_dt_on_off", change["new"]),
            names="value",
        )
        self.widgets.pid1_gain.observe(
            lambda change: setattr(self.state, "pid1_gain", change["new"]),
            names="value",
        )
        self.widgets.pid1_reset.observe(
            lambda change: setattr(self.state, "pid1_reset", change["new"]),
            names="value",
        )
        self.widgets.pid1_rate.observe(
            lambda change: setattr(self.state, "pid1_rate", change["new"]),
            names="value",
        )
        self.widgets.pid2_gain.observe(
            lambda change: setattr(self.state, "pid2_gain", change["new"]),
            names="value",
        )
        self.widgets.pid2_reset.observe(
            lambda change: setattr(self.state, "pid2_reset", change["new"]),
            names="value",
        )
        self.widgets.pid2_rate.observe(
            lambda change: setattr(self.state, "pid2_rate", change["new"]),
            names="value",
        )
        self.widgets.mpc_solver.observe(
            lambda change: setattr(self.state, "solver", change["new"]),
            names="value",
        )
        self.widgets.mpc_cvtype.observe(
            lambda change: setattr(self.state, "cvtype", change["new"]),
            names="value",
        )
        self.widgets.mpc_t1_dt.observe(
            lambda change: setattr(self.state, "t1_dt", change["new"]),
            names="value",
        )
        self.widgets.mpc_t1_tau.observe(
            lambda change: setattr(self.state, "t1_tau", change["new"]),
            names="value",
        )
        self.widgets.mpc_t2_dt.observe(
            lambda change: setattr(self.state, "t2_dt", change["new"]),
            names="value",
        )
        self.widgets.mpc_t2_tau.observe(
            lambda change: setattr(self.state, "t2_tau", change["new"]),
            names="value",
        )
        self.widgets.mpc_q1_dmax.observe(
            lambda change: setattr(self.state, "q1_dmax", change["new"]),
            names="value",
        )
        self.widgets.mpc_q1_dcost.observe(
            lambda change: setattr(self.state, "q1_dcost", change["new"]),
            names="value",
        )
        self.widgets.mpc_q2_dmax.observe(
            lambda change: setattr(self.state, "q2_dmax", change["new"]),
            names="value",
        )
        self.widgets.mpc_q2_dcost.observe(
            lambda change: setattr(self.state, "q2_dcost", change["new"]),
            names="value",
        )

    def _render_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self.view.render(snapshot, self.state)

    def _set_running(self, running: bool) -> None:
        self.view.apply_mode(self.state, running=running)

    def _handle_mode_change(self, change: dict[str, object]) -> None:
        if self.runner.is_running:
            return
        new_mode = change["new"]
        if not isinstance(new_mode, str):
            return
        self.state.reset_for_mode(_coerce_mode(new_mode))
        self.runner.invalidate_runtime()
        self.view.reset_for_mode(self.state)

    def _handle_start(self, _button: Any) -> None:
        self.runner.start(self.state.mode)

    def _handle_stop(self, _button: Any) -> None:
        self.runner.stop()

    def _handle_q1_button(self, _button: Any) -> None:
        self.state.q1 = self.widgets.q1_slider.value

    def _handle_q2_button(self, _button: Any) -> None:
        self.state.q2 = self.widgets.q2_slider.value

    def _handle_t1_button(self, _button: Any) -> None:
        self.state.t1_sp = self.widgets.t1_slider.value

    def _handle_t2_button(self, _button: Any) -> None:
        self.state.t2_sp = self.widgets.t2_slider.value

    def _sync_q1(self, change: dict[str, object]) -> None:
        if not self.state.suspend_control_updates and not self.widgets.q1_slider.disabled:
            self.state.q1 = _coerce_float(change["new"])

    def _sync_q2(self, change: dict[str, object]) -> None:
        if not self.state.suspend_control_updates and not self.widgets.q2_slider.disabled:
            self.state.q2 = _coerce_float(change["new"])

    def _sync_t1(self, change: dict[str, object]) -> None:
        if not self.state.suspend_control_updates and not self.widgets.t1_slider.disabled:
            self.state.t1_sp = _coerce_float(change["new"])

    def _sync_t2(self, change: dict[str, object]) -> None:
        if not self.state.suspend_control_updates and not self.widgets.t2_slider.disabled:
            self.state.t2_sp = _coerce_float(change["new"])

    def _sync_general_delta(self, change: dict[str, object]) -> None:
        self.state.delta_t = _coerce_float(change["new"])
        self.runner.update_runtime_config()

    def _sync_general_sleep(self, change: dict[str, object]) -> None:
        self.state.sleep_s = _coerce_float(change["new"])


class SimulationNotebookApp(_NotebookApp):
    def __init__(self) -> None:
        super().__init__(backend_factory=lambda: SimulationBackend())


class HardwareNotebookApp(_NotebookApp):
    def __init__(self) -> None:
        super().__init__(backend_factory=lambda: HardwareBackend())
