"""Notebook UI wrappers around the extracted runtime."""

from __future__ import annotations

import importlib
import queue
import threading
from typing import Any

import numpy as np

from ..backends import HardwareBackend, SimulationBackend
from ..controllers import ControllerSolveError, MpcConfig
from ..runtime import ControlRuntime, RuntimeConfig, RuntimeSnapshot


def _load_widget_modules() -> tuple[Any, Any, Any]:
    display = importlib.import_module("IPython.display").display
    bq = importlib.import_module("bqplot")
    wi = importlib.import_module("ipywidgets").widgets

    return wi, bq, display


class _BaseNotebookGUI:
    def __init__(self, backend_factory) -> None:
        self._backend_factory = backend_factory
        self._runtime: ControlRuntime | None = None
        self._worker: threading.Thread | None = None
        self._ui_poller = None
        self._running = False
        self._suspend_control_updates = False
        self._stop_event = threading.Event()
        self._ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._last_rendered_snapshot: RuntimeSnapshot | None = None
        self._worker_error: BaseException | None = None

        wi, bq, display = _load_widget_modules()
        self._display = display
        self._wi = wi
        self._bq = bq

        self._delta_t = 4.0
        self._history_seconds = 500.0
        self._sleep = 0.5
        self._T1_SP = 30.0
        self._T2_SP = 30.0
        self._Q10 = 0.0
        self._Q20 = 0.0
        self._Tc0 = np.array([293.15, 293.15], dtype=float)

        colors = ["#1f77b4", "#5ebdff", "#d62728", "#ff7f0e", "#1f77b4", "#d62728"]
        self._time_window_min = self._history_seconds / 60.0
        self._time_scale = bq.LinearScale(min=0.0, max=self._time_window_min)
        temperature_scale = bq.LinearScale(min=20.0, max=60.0)
        heater_scale = bq.LinearScale(min=0.0, max=100.0)
        temperature_scale_set = {"x": self._time_scale, "y": temperature_scale}
        heater_scale_set = {"x": self._time_scale, "y": heater_scale}
        ax_x1 = bq.Axis(label="", scale=self._time_scale, visible=False)
        ax_x2 = bq.Axis(label="time [min]", scale=self._time_scale)
        ax_T1 = bq.Axis(
            label="Temperature 1 [C]",
            scale=temperature_scale,
            orientation="vertical",
            label_color=colors[0],
        )
        ax_T2 = bq.Axis(
            label="Temperature 2 [C]",
            scale=temperature_scale,
            orientation="vertical",
            label_color=colors[2],
        )
        ax_Q = bq.Axis(
            label="Heater Output [%]",
            scale=heater_scale,
            orientation="vertical",
            label_color="black",
        )
        self._T1_meas = bq.Scatter(
            x=[],
            y=[],
            scales=temperature_scale_set,
            marker="circle",
            colors=[colors[0]],
            default_size=20,
        )
        self._T2_meas = bq.Scatter(
            x=[],
            y=[],
            scales=temperature_scale_set,
            marker="circle",
            colors=[colors[2]],
            default_size=20,
        )
        self._T1_set_point = bq.Lines(
            x=[],
            y=[],
            scales=temperature_scale_set,
            stroke_width=4,
            colors=[colors[1]],
            interpolation="step-before",
        )
        self._T2_set_point = bq.Lines(
            x=[],
            y=[],
            scales=temperature_scale_set,
            stroke_width=4,
            colors=[colors[3]],
            interpolation="step-before",
        )
        self._u1 = bq.Lines(
            x=[],
            y=[],
            scales=heater_scale_set,
            stroke_width=2,
            colors=[colors[4]],
            interpolation="step-before",
            display_legend=True,
            labels=["Heater 1"],
        )
        self._u2 = bq.Lines(
            x=[],
            y=[],
            scales=heater_scale_set,
            stroke_width=2,
            colors=[colors[5]],
            interpolation="step-before",
            display_legend=True,
            labels=["Heater 2"],
        )

        fig_lay_up = wi.Layout(width="400px", height="240px")
        fig_lay_down = wi.Layout(width="400px", height="290px")
        fig1 = bq.Figure(layout=fig_lay_up, axes=[ax_x1, ax_T1], marks=[self._T1_meas, self._T1_set_point], fig_margin=dict(top=10, bottom=0, left=60, right=5))
        fig2 = bq.Figure(layout=fig_lay_up, axes=[ax_x1, ax_T2], marks=[self._T2_meas, self._T2_set_point], fig_margin=dict(top=10, bottom=0, left=60, right=20))
        fig3 = bq.Figure(
            layout=fig_lay_down,
            axes=[ax_x2, ax_Q],
            marks=[self._u1, self._u2],
            fig_margin=dict(top=10, bottom=50, left=60, right=10),
            legend_location="left",
            legend_style={"fill": "white", "fill-opacity": "0.7", "width": "130px"},
        )

        v_space = wi.Label(value="", layout=wi.Layout(height="2px"))
        h_space = wi.Label(value="", layout=wi.Layout(width="2px"))

        self._q1_dt_on_off = 0.1
        self._q2_dt_on_off = 0.1
        self._pid1_gain = 10.0
        self._pid1_reset = 50.0
        self._pid1_rate = 1.0
        self._pid2_gain = 10.0
        self._pid2_reset = 50.0
        self._pid2_rate = 1.0
        self._SOLVER = "1 - APOPT"
        self._CVTYPE = "1 - Deadband"
        self._T1_dt = 0.1
        self._T1_tau = 10.0
        self._T2_dt = 0.1
        self._T2_tau = 10.0
        self._Q1_DMAX = 30.0
        self._Q1_DCOST = 1.0
        self._Q2_DMAX = 30.0
        self._Q2_DCOST = 1.0

        style = {"description_width": "initial"}
        self._PT1 = wi.FloatProgress(
            value=20.0,
            min=0,
            max=100.0,
            description="PV (°C):",
            bar_style="warning",
            orientation="horizontal",
            style=style,
        )
        self._LT1 = wi.Label(value=self._format_temperature_label(self._PT1.value))
        self._PT2 = wi.FloatProgress(
            value=20.0,
            min=0,
            max=100.0,
            description="PV (°C):",
            bar_style="warning",
            orientation="horizontal",
            style=style,
        )
        self._LT2 = wi.Label(value=self._format_temperature_label(self._PT2.value))

        self._wQ1 = wi.FloatSlider(
            value=self._Q10,
            min=0,
            max=100.0,
            step=0.5,
            description="Q1 (%):",
            continuous_update=False,
            orientation="horizontal",
            readout=False,
            style=style,
            layout=wi.Layout(width="230px"),
        )
        self._tQ1 = wi.BoundedFloatText(value=self._Q10, min=0, max=100.0, step=0.5, layout=wi.Layout(width="60px"))
        wi.jslink((self._wQ1, "value"), (self._tQ1, "value"))
        self._bQ1 = wi.Button(description="Set", layout=wi.Layout(width="50px"))
        self._bQ1.on_click(self._Q1_click)
        self._wQ1.observe(self._sync_q1, names="value")

        self._wQ2 = wi.FloatSlider(
            value=self._Q20,
            min=0,
            max=100.0,
            step=0.5,
            description="Q2 (%):",
            continuous_update=False,
            orientation="horizontal",
            readout=False,
            style=style,
            layout=wi.Layout(width="230px"),
        )
        self._tQ2 = wi.BoundedFloatText(value=self._Q20, min=0, max=100.0, step=0.5, layout=wi.Layout(width="60px"))
        wi.jslink((self._wQ2, "value"), (self._tQ2, "value"))
        self._bQ2 = wi.Button(description="Set", layout=wi.Layout(width="50px"))
        self._bQ2.on_click(self._Q2_click)
        self._wQ2.observe(self._sync_q2, names="value")

        self._wT1 = wi.FloatSlider(
            value=self._T1_SP,
            min=20,
            max=60.0,
            step=0.5,
            description="T1 SP:",
            continuous_update=False,
            orientation="horizontal",
            readout=False,
            style=style,
            disabled=True,
            layout=wi.Layout(width="230px"),
        )
        self._tT1 = wi.BoundedFloatText(value=self._T1_SP, min=20, max=60.0, step=0.5, disabled=True, layout=wi.Layout(width="60px"))
        wi.jslink((self._wT1, "value"), (self._tT1, "value"))
        self._bT1 = wi.Button(description="Set", layout=wi.Layout(width="50px"), disabled=True)
        self._bT1.on_click(self._T1_click)
        self._wT1.observe(self._sync_t1_sp, names="value")

        self._wT2 = wi.FloatSlider(
            value=self._T2_SP,
            min=20,
            max=60.0,
            step=0.5,
            description="T2 SP:",
            continuous_update=False,
            orientation="horizontal",
            disabled=True,
            readout=False,
            style=style,
            layout=wi.Layout(width="230px"),
        )
        self._tT2 = wi.BoundedFloatText(value=self._T2_SP, min=20, max=60.0, step=0.5, disabled=True, layout=wi.Layout(width="60px"))
        wi.jslink((self._wT2, "value"), (self._tT2, "value"))
        self._bT2 = wi.Button(description="Set", layout=wi.Layout(width="50px"), disabled=True)
        self._bT2.on_click(self._T2_click)
        self._wT2.observe(self._sync_t2_sp, names="value")

        self._mode = wi.ToggleButtons(options=["Manual", "On-Off", "PID", "MPC"], style={"button_width": "100px"})
        self._mode.observe(self._mode_switch, names="value")
        self._b_stop = wi.Button(description="Stop", button_style="warning", icon="stop", layout=wi.Layout(width="100px", height="32px"))
        self._b_stop.on_click(self._stop_click)
        self._b_play = wi.Button(description="Start", button_style="success", icon="play", layout=wi.Layout(width="100px", height="32px"))
        self._b_play.on_click(self._play_click)

        buttons = wi.HBox((self._b_play, h_space, self._b_stop, wi.Label(value="", layout=wi.Layout(width="165px")), self._mode))
        Q1_Set = wi.HBox((h_space, self._wQ1, self._tQ1, self._bQ1), layout=wi.Layout(border="solid 2px gray", width="360px"))
        T1_Set = wi.HBox((h_space, self._wT1, self._tT1, self._bT1), layout=wi.Layout(border="solid 2px gray", width="360px"))
        T1_View = wi.HBox((h_space, self._PT1, self._LT1), layout=wi.Layout(border="solid 2px gray", width="360px"))
        Q2_Set = wi.HBox((h_space, self._wQ2, self._tQ2, self._bQ2), layout=wi.Layout(border="solid 2px gray", width="360px"))
        T2_Set = wi.HBox((h_space, self._wT2, self._tT2, self._bT2), layout=wi.Layout(border="solid 2px gray", width="360px"))
        T2_View = wi.HBox((h_space, self._PT2, self._LT2), layout=wi.Layout(border="solid 2px gray", width="360px"))
        co1 = wi.VBox((T1_Set, v_space, T1_View, v_space, Q1_Set))
        co2 = wi.VBox((T2_Set, v_space, T2_View, v_space, Q2_Set))
        co = wi.VBox((v_space, v_space, co1, v_space, v_space, v_space, co2))
        figy = wi.VBox((wi.HBox((fig1, fig2)), v_space, wi.HBox((fig3, h_space, h_space, co))), layout=wi.Layout(border="solid 2px gray", width="800px"))
        self._gui = wi.VBox((buttons, v_space, figy))

        self._conf11 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>&Delta;t (s):</b></p>', layout=wi.Layout(width="120px", margin="0 20px 0 0")), wi.FloatSlider(value=4.0, min=1.0, max=10.0, step=0.5, description="", style=style)))
        self._conf12 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>Sleep time (s):</b></p>', layout=wi.Layout(width="120px", margin="0 20px 0 0")), wi.FloatSlider(value=0.5, min=0.25, max=1.5, step=0.05, description="", style=style)))
        self._conf21 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>Deadband (K):</b></p>', layout=wi.Layout(width="90px", margin="0 10px 0 15px")), wi.FloatSlider(value=0.1, min=0.0, max=2.0, step=0.1, description="", style=style, layout=wi.Layout(width="200px"))))
        self._conf23 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>Deadband (K):</b></p>', layout=wi.Layout(width="90px", margin="0 10px 0 15px")), wi.FloatSlider(value=0.1, min=0.0, max=2.0, step=0.1, description="", style=style, layout=wi.Layout(width="200px"))))
        self._conf31 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>Kc (K/%Heater):</b></p>', layout=wi.Layout(width="100px", margin="0 10px 0 15px")), wi.FloatSlider(value=10.0, min=0.0, max=20.0, step=0.5, description="", style=style, layout=wi.Layout(width="200px"))))
        self._conf32 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>tauI (s):</b></p>', layout=wi.Layout(width="100px", margin="0 10px 0 15px")), wi.FloatSlider(value=50.0, min=0.0, max=200.0, step=1.0, description="", style=style, layout=wi.Layout(width="200px"))))
        self._conf33 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>tauD (s):</b></p>', layout=wi.Layout(width="100px", margin="0 10px 0 15px")), wi.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.5, description="", style=style, layout=wi.Layout(width="200px"))))
        self._conf35 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>Kc (K/%Heater):</b></p>', layout=wi.Layout(width="100px", margin="0 10px 0 15px")), wi.FloatSlider(value=10.0, min=0.0, max=20.0, step=0.5, description="", style=style, layout=wi.Layout(width="200px"))))
        self._conf36 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>tauI (s):</b></p>', layout=wi.Layout(width="100px", margin="0 10px 0 15px")), wi.FloatSlider(value=50.0, min=0.0, max=200.0, step=1.0, description="", style=style, layout=wi.Layout(width="200px"))))
        self._conf37 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>tauD (s):</b></p>', layout=wi.Layout(width="100px", margin="0 10px 0 15px")), wi.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.5, description="", style=style, layout=wi.Layout(width="200px"))))
        self._conf40 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>SOLVER:</b></p>', layout=wi.Layout(width="60px", margin="0 10px 0 25px")), wi.Dropdown(options=["1 - APOPT", "2 - BPOPT", "3 - IPOPT"], value="1 - APOPT", layout=wi.Layout(width="100px")), wi.HTML(value='<p style="text-align: right;"><b>CVTYPE:</b></p>', layout=wi.Layout(width="60px", margin="0 10px 0 25px")), wi.Dropdown(options=["1 - Deadband", "2 - Trajectory"], value="1 - Deadband", layout=wi.Layout(width="130px"))))
        self._conf42 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>Deadband (K):</b></p>', layout=wi.Layout(width="95px", margin="0 20px 0 10px")), wi.FloatSlider(value=0.1, min=0.1, max=1.0, step=0.1, description="", style=style, layout=wi.Layout(width="250px"))))
        self._conf43 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>TAU:</b></p>', layout=wi.Layout(width="95px", margin="0 20px 0 10px")), wi.FloatSlider(value=10.0, min=1.0, max=100.0, step=1.0, description="", style=style, layout=wi.Layout(width="250px"))))
        self._conf45 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>Deadband (K):</b></p>', layout=wi.Layout(width="95px", margin="0 20px 0 10px")), wi.FloatSlider(value=0.1, min=0.1, max=1.0, step=0.1, description="", style=style, layout=wi.Layout(width="250px"))))
        self._conf46 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>TAU:</b></p>', layout=wi.Layout(width="95px", margin="0 20px 0 10px")), wi.FloatSlider(value=10.0, min=1.0, max=100.0, step=1.0, description="", style=style, layout=wi.Layout(width="250px"))))
        self._conf48 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>DMAX:</b></p>', layout=wi.Layout(width="55px", margin="0 20px 0 10px")), wi.FloatSlider(value=30.0, min=1.0, max=100.0, step=0.5, description="", style=style, layout=wi.Layout(width="250px"))))
        self._conf49 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>DCOST:</b></p>', layout=wi.Layout(width="55px", margin="0 20px 0 10px")), wi.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.1, description="", style=style, layout=wi.Layout(width="250px"))))
        self._conf411 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>DMAX:</b></p>', layout=wi.Layout(width="55px", margin="0 20px 0 10px")), wi.FloatSlider(value=30.0, min=1.0, max=100.0, step=0.5, description="", style=style, layout=wi.Layout(width="250px"))))
        self._conf412 = wi.HBox((wi.HTML(value='<p style="text-align: right;"><b>DCOST:</b></p>', layout=wi.Layout(width="55px", margin="0 20px 0 10px")), wi.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.1, description="", style=style, layout=wi.Layout(width="250px"))))
        self._conf = wi.Tab([wi.VBox((self._conf11, self._conf12)), wi.VBox((self._conf21, self._conf23)), wi.VBox((self._conf31, self._conf32, self._conf33, self._conf35, self._conf36, self._conf37)), wi.VBox((self._conf40, self._conf42, self._conf43, self._conf45, self._conf46, self._conf48, self._conf49, self._conf411, self._conf412))], layout=wi.Layout(width="800px", height="380px"))
        self._conf.set_title(0, "General Options")
        self._conf.set_title(1, "On-Off Options")
        self._conf.set_title(2, "PID Options")
        self._conf.set_title(3, "MPC Options")

    def app(self) -> None:
        self._display(self._gui)

    def config(self) -> None:
        self._display(self._conf)

    def _build_runtime(self) -> ControlRuntime:
        return ControlRuntime(
            self._backend_factory(),
            RuntimeConfig(delta_t=self._delta_t, history_seconds=self._history_seconds),
        )

    def _clear_ui_queue(self) -> None:
        while True:
            try:
                self._ui_queue.get_nowait()
            except queue.Empty:
                return

    def _initialize_runtime(self) -> None:
        if self._runtime is not None:
            return
        self._runtime = self._build_runtime()
        initial = self._runtime.initialize(self._Q10, self._Q20, self._T1_SP, self._T2_SP)
        self._render(initial)

    def _enqueue_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self._ui_queue.put(("snapshot", snapshot))

    def _enqueue_error(self, exc: BaseException) -> None:
        self._ui_queue.put(("error", exc))

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                event_type, payload = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            if event_type == "snapshot":
                self._render(payload)
            elif event_type == "error":
                self._worker_error = payload
                self._stop_event.set()
                self._running = False

    def _render(self, snapshot: RuntimeSnapshot) -> None:
        self._last_rendered_snapshot = snapshot
        time_min = snapshot.time_s / 60.0
        latest_time = float(time_min[-1]) if len(time_min) else 0.0
        window_min = self._history_seconds / 60.0
        self._time_scale.min = max(0.0, latest_time - window_min)
        self._time_scale.max = max(window_min, latest_time)
        self._T1_meas.x = time_min
        self._T1_meas.y = snapshot.t1_c
        self._T2_meas.x = time_min
        self._T2_meas.y = snapshot.t2_c
        self._T1_set_point.x = time_min
        self._T1_set_point.y = snapshot.sp1_c
        self._T2_set_point.x = time_min
        self._T2_set_point.y = snapshot.sp2_c
        self._u1.x = time_min
        self._u1.y = snapshot.q1
        self._u2.x = time_min
        self._u2.y = snapshot.q2
        self._PT1.value = round(float(snapshot.t1_c[-1]), 1)
        self._PT2.value = round(float(snapshot.t2_c[-1]), 1)
        self._LT1.value = self._format_temperature_label(self._PT1.value)
        self._LT2.value = self._format_temperature_label(self._PT2.value)
        if self._wQ1.disabled:
            self._set_control_value(self._wQ1, round(float(snapshot.q1[-1]), 1))
        if self._wQ2.disabled:
            self._set_control_value(self._wQ2, round(float(snapshot.q2[-1]), 1))

    def _conf_general(self, _button) -> None:
        self._delta_t = self._conf11.children[1].value
        self._sleep = self._conf12.children[1].value
        self._runtime = None

    def _reset_general(self, _button) -> None:
        self._conf11.children[1].value = 4.0
        self._conf12.children[1].value = 0.5
        self._conf_general(_button)

    def _conf_on_off(self, _button) -> None:
        self._q1_dt_on_off = self._conf21.children[1].value
        self._q2_dt_on_off = self._conf23.children[1].value

    def _reset_on_off(self, _button) -> None:
        self._conf21.children[1].value = 0.1
        self._conf23.children[1].value = 0.1
        self._conf_on_off(_button)

    def _conf_pid(self, _button) -> None:
        self._pid1_gain = self._conf31.children[1].value
        self._pid1_reset = self._conf32.children[1].value
        self._pid1_rate = self._conf33.children[1].value
        self._pid2_gain = self._conf35.children[1].value
        self._pid2_reset = self._conf36.children[1].value
        self._pid2_rate = self._conf37.children[1].value

    def _reset_pid(self, _button) -> None:
        self._conf31.children[1].value = 10.0
        self._conf32.children[1].value = 50.0
        self._conf33.children[1].value = 1.0
        self._conf35.children[1].value = 10.0
        self._conf36.children[1].value = 50.0
        self._conf37.children[1].value = 1.0
        self._conf_pid(_button)

    def _conf_mpc(self, _button) -> None:
        self._SOLVER = self._conf40.children[1].value
        self._CVTYPE = self._conf40.children[3].value
        self._T1_dt = self._conf42.children[1].value
        self._T1_tau = self._conf43.children[1].value
        self._T2_dt = self._conf45.children[1].value
        self._T2_tau = self._conf46.children[1].value
        self._Q1_DMAX = self._conf48.children[1].value
        self._Q1_DCOST = self._conf49.children[1].value
        self._Q2_DMAX = self._conf411.children[1].value
        self._Q2_DCOST = self._conf412.children[1].value

    def _reset_mpc(self, _button) -> None:
        self._conf40.children[1].value = "1 - APOPT"
        self._conf40.children[3].value = "1 - Deadband"
        self._conf42.children[1].value = 0.1
        self._conf43.children[1].value = 10.0
        self._conf45.children[1].value = 0.1
        self._conf46.children[1].value = 10.0
        self._conf48.children[1].value = 30.0
        self._conf49.children[1].value = 1.0
        self._conf411.children[1].value = 30.0
        self._conf412.children[1].value = 1.0
        self._conf_mpc(_button)

    def _Q1_click(self, _button) -> None:
        self._Q10 = self._wQ1.value

    def _Q2_click(self, _button) -> None:
        self._Q20 = self._wQ2.value

    def _T1_click(self, _button) -> None:
        self._T1_SP = self._wT1.value

    def _T2_click(self, _button) -> None:
        self._T2_SP = self._wT2.value

    def _set_control_value(self, widget, value: float) -> None:
        self._suspend_control_updates = True
        try:
            widget.value = value
        finally:
            self._suspend_control_updates = False

    @staticmethod
    def _format_temperature_label(value: float) -> str:
        return f"{value:.1f}"

    def _sync_q1(self, change) -> None:
        if not self._suspend_control_updates and not self._wQ1.disabled:
            self._Q10 = change["new"]

    def _sync_q2(self, change) -> None:
        if not self._suspend_control_updates and not self._wQ2.disabled:
            self._Q20 = change["new"]

    def _sync_t1_sp(self, change) -> None:
        if not self._suspend_control_updates and not self._wT1.disabled:
            self._T1_SP = change["new"]

    def _sync_t2_sp(self, change) -> None:
        if not self._suspend_control_updates and not self._wT2.disabled:
            self._T2_SP = change["new"]

    def _stop_click(self, _button) -> None:
        self._stop_worker()

    def _play_click(self, _button) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._clear_ui_queue()
        self._initialize_runtime()
        self._stop_event.clear()
        self._worker_error = None
        self._running = True
        self._mode.disabled = True
        workers = {
            "Manual": self._work_man,
            "On-Off": self._work_on_off,
            "PID": self._work_pid,
            "MPC": self._work_mpc,
        }
        self._worker = threading.Thread(target=self._run_worker, args=(workers[self._mode.value],))
        self._worker.start()
        self._start_ui_poller()

    def _finalize_worker(self) -> None:
        self._stop_ui_poller()
        worker = self._worker
        if worker is not None and worker.is_alive():
            return
        self._worker = None
        self._running = False
        self._mode.disabled = False
        if self._runtime is not None:
            self._runtime.shutdown()
            self._runtime = None

    def _stop_worker(self) -> None:
        if self._worker is None and self._runtime is None:
            return
        self._stop_event.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=2.0)
        if worker is None or not worker.is_alive():
            self._drain_ui_queue()
            self._finalize_worker()
        else:
            self._start_ui_poller()

    def _mode_switch(self, value) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._runtime = None
        self._stop_event.set()
        self._running = False
        self._worker_error = None
        self._clear_ui_queue()
        self._stop_ui_poller()
        self._T1_SP = 30.0
        self._T2_SP = 30.0
        self._Q10 = 0.0
        self._Q20 = 0.0
        for mark in (self._T1_meas, self._T2_meas, self._T1_set_point, self._T2_set_point, self._u1, self._u2):
            mark.x = []
            mark.y = []
        self._PT1.value = 20.0
        self._PT2.value = 20.0
        self._LT1.value = self._format_temperature_label(self._PT1.value)
        self._LT2.value = self._format_temperature_label(self._PT2.value)
        self._wT1.value = self._T1_SP
        self._wT2.value = self._T2_SP
        self._wQ1.value = self._Q10
        self._wQ2.value = self._Q20
        manual = value["new"] == "Manual"
        for widget in (self._wQ1, self._tQ1, self._bQ1, self._wQ2, self._tQ2, self._bQ2):
            widget.disabled = not manual
        for widget in (self._wT1, self._tT1, self._bT1, self._wT2, self._tT2, self._bT2):
            widget.disabled = manual

    def _start_ui_poller(self) -> None:
        if self._ui_poller is not None:
            return
        periodic_callback = importlib.import_module("tornado.ioloop").PeriodicCallback
        self._ui_poller = periodic_callback(self._poll_ui, 50)
        self._ui_poller.start()

    def _stop_ui_poller(self) -> None:
        if self._ui_poller is None:
            return
        self._ui_poller.stop()
        self._ui_poller = None

    def _poll_ui(self) -> None:
        self._drain_ui_queue()
        worker = self._worker
        if worker is None:
            self._finalize_worker()
            return
        if worker.is_alive():
            return
        self._finalize_worker()

    def _run_worker(self, worker) -> None:
        try:
            worker()
        except ControllerSolveError as exc:
            self._stop_event.set()
            self._enqueue_error(exc)
        except Exception as exc:
            self._stop_event.set()
            self._enqueue_error(exc)
        finally:
            self._running = False

    def _work_man(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        while not self._stop_event.is_set():
            snapshot = runtime.step_manual(self._Q10, self._Q20)
            self._enqueue_snapshot(snapshot)
            if self._stop_event.wait(self._sleep):
                break

    def _work_on_off(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        while not self._stop_event.is_set():
            snapshot, self._Q10, self._Q20 = runtime.step_on_off(
                self._Q10,
                self._Q20,
                self._T1_SP,
                self._T2_SP,
                self._q1_dt_on_off,
                self._q2_dt_on_off,
            )
            self._enqueue_snapshot(snapshot)
            if self._stop_event.wait(self._sleep):
                break

    def _work_pid(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        while not self._stop_event.is_set():
            snapshot, self._Q10, self._Q20 = runtime.step_pid(
                self._Q10,
                self._Q20,
                self._T1_SP,
                self._T2_SP,
                self._pid1_gain,
                self._pid1_reset,
                self._pid1_rate,
                self._pid2_gain,
                self._pid2_reset,
                self._pid2_rate,
            )
            self._enqueue_snapshot(snapshot)
            if self._stop_event.wait(self._sleep):
                break

    def _work_mpc(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        config = MpcConfig(
            solver=self._SOLVER,
            cvtype=self._CVTYPE,
            t1_deadband=self._T1_dt,
            t1_tau=self._T1_tau,
            t2_deadband=self._T2_dt,
            t2_tau=self._T2_tau,
            q1_dmax=self._Q1_DMAX,
            q1_dcost=self._Q1_DCOST,
            q2_dmax=self._Q2_DMAX,
            q2_dcost=self._Q2_DCOST,
        )
        while not self._stop_event.is_set():
            snapshot, self._Q10, self._Q20 = runtime.step_mpc(
                self._Q10,
                self._Q20,
                self._T1_SP,
                self._T2_SP,
                config,
            )
            self._enqueue_snapshot(snapshot)
            if self._stop_event.wait(self._sleep):
                break


class SimulationNotebookApp(_BaseNotebookGUI):
    def __init__(self) -> None:
        super().__init__(backend_factory=lambda: SimulationBackend())


class HardwareNotebookApp(_BaseNotebookGUI):
    def __init__(self) -> None:
        super().__init__(backend_factory=lambda: HardwareBackend())
