from __future__ import annotations

import threading
from typing import Any, cast

import numpy as np
import pytest

from tclab_jupyter.controllers import ControllerSolveError
from tclab_jupyter.runtime import RuntimeSnapshot


def _load_ui_components():
    pytest.importorskip("ipywidgets")
    pytest.importorskip("bqplot")
    from tclab_jupyter.ui.notebook import (
        NotebookRunner,
        NotebookSessionState,
        NotebookView,
        _load_widget_modules,
    )

    wi, bq, _display = _load_widget_modules()
    return NotebookSessionState, NotebookView, NotebookRunner, wi, bq


def _build_view():
    notebook_session_state, notebook_view, _notebook_runner, wi, bq = (
        _load_ui_components()
    )
    state = notebook_session_state()
    view = notebook_view(wi=wi, bq=bq)
    widgets = view.build(state)
    view.reset_for_mode(state)
    return state, view, widgets


def test_public_notebook_apps_instantiate_when_optional_deps_are_available() -> None:
    pytest.importorskip("ipywidgets")
    pytest.importorskip("bqplot")
    from tclab_jupyter.ui.notebook import HardwareNotebookApp, SimulationNotebookApp

    simulation = SimulationNotebookApp()
    hardware = HardwareNotebookApp()

    assert simulation.widgets.mode.value == "Manual"
    assert hardware.widgets.mode.value == "Manual"


def test_view_render_updates_time_axis_as_moving_window() -> None:
    state, view, widgets = _build_view()
    snapshot = RuntimeSnapshot(
        time_s=np.array([0.0, 300.0, 600.0]),
        q1=np.array([0.0, 10.0, 20.0]),
        q2=np.array([0.0, 0.0, 0.0]),
        t1_c=np.array([22.0, 23.0, 24.0]),
        t2_c=np.array([22.0, 23.0, 24.0]),
        sp1_c=np.array([30.0, 30.0, 30.0]),
        sp2_c=np.array([30.0, 30.0, 30.0]),
    )

    view.render(snapshot, state)

    assert widgets.time_scale.min == pytest.approx(1.666666666666666)
    assert widgets.time_scale.max == pytest.approx(10.0)


def test_view_render_keeps_progress_labels_as_strings() -> None:
    state, view, widgets = _build_view()
    snapshot = RuntimeSnapshot(
        time_s=np.array([0.0, 4.0]),
        q1=np.array([0.0, 0.0]),
        q2=np.array([0.0, 0.0]),
        t1_c=np.array([22.0, 22.1]),
        t2_c=np.array([22.0, 22.2]),
        sp1_c=np.array([30.0, 30.0]),
        sp2_c=np.array([30.0, 30.0]),
    )

    view.render(snapshot, state)

    assert isinstance(widgets.lt1.value, str)
    assert isinstance(widgets.lt2.value, str)


def test_view_does_not_overwrite_manual_controls() -> None:
    state, view, widgets = _build_view()
    widgets.q1_slider.value = 12.5
    state.q1 = 12.5
    snapshot = RuntimeSnapshot(
        time_s=np.array([0.0, 4.0]),
        q1=np.array([0.0, 0.0]),
        q2=np.array([0.0, 0.0]),
        t1_c=np.array([22.0, 22.1]),
        t2_c=np.array([22.0, 22.2]),
        sp1_c=np.array([30.0, 30.0]),
        sp2_c=np.array([30.0, 30.0]),
    )

    view.render(snapshot, state)

    assert widgets.q1_slider.value == 12.5


def test_view_apply_mode_updates_enabled_controls() -> None:
    state, view, widgets = _build_view()

    state.reset_for_mode("PID")
    view.reset_for_mode(state)

    assert widgets.q1_slider.disabled is True
    assert widgets.q2_slider.disabled is True
    assert widgets.t1_slider.disabled is False
    assert widgets.t2_slider.disabled is False


def test_runner_queues_snapshot_until_poll_renders() -> None:
    notebook_session_state, _notebook_view, notebook_runner, _wi, _bq = (
        _load_ui_components()
    )
    state = notebook_session_state()
    snapshots: list[RuntimeSnapshot] = []

    class DummyPoller:
        def start(self) -> None:
            return

        def stop(self) -> None:
            return

    runner = notebook_runner(
        state=state,
        backend_factory=lambda: cast(Any, None),
        on_snapshot=snapshots.append,
        on_running_change=lambda running: None,
        periodic_callback_factory=lambda callback, interval: DummyPoller(),
    )
    snapshot = RuntimeSnapshot(
        time_s=np.array([0.0, 4.0]),
        q1=np.array([0.0, 0.0]),
        q2=np.array([0.0, 0.0]),
        t1_c=np.array([22.0, 22.1]),
        t2_c=np.array([22.0, 22.2]),
        sp1_c=np.array([30.0, 30.0]),
        sp2_c=np.array([30.0, 30.0]),
    )

    runner._queue.put(("snapshot", snapshot))

    assert snapshots == []
    runner.poll()
    assert snapshots == [snapshot]


def test_runner_stop_waits_for_worker_before_runtime_shutdown() -> None:
    notebook_session_state, _notebook_view, notebook_runner, _wi, _bq = (
        _load_ui_components()
    )
    state = notebook_session_state()
    worker_released = threading.Event()
    worker_started = threading.Event()
    shutdown_after_worker = {"value": False}

    class DummyPoller:
        def start(self) -> None:
            return

        def stop(self) -> None:
            return

    class FakeRuntime:
        def initialize(
            self, q1: float, q2: float, sp1_c: float, sp2_c: float
        ) -> RuntimeSnapshot:
            return RuntimeSnapshot(
                time_s=np.array([0.0]),
                q1=np.array([q1]),
                q2=np.array([q2]),
                t1_c=np.array([22.0]),
                t2_c=np.array([22.0]),
                sp1_c=np.array([sp1_c]),
                sp2_c=np.array([sp2_c]),
            )

        def shutdown(self) -> None:
            shutdown_after_worker["value"] = worker_released.is_set()

        def step_manual(self, q1: float, q2: float) -> RuntimeSnapshot:
            worker_started.set()
            runner._stop_event.wait()
            worker_released.set()
            return RuntimeSnapshot(
                time_s=np.array([0.0]),
                q1=np.array([q1]),
                q2=np.array([q2]),
                t1_c=np.array([22.0]),
                t2_c=np.array([22.0]),
                sp1_c=np.array([30.0]),
                sp2_c=np.array([30.0]),
            )

    runner = notebook_runner(
        state=state,
        backend_factory=lambda: cast(Any, None),
        on_snapshot=lambda snapshot: None,
        on_running_change=lambda running: None,
        runtime_factory=lambda config: FakeRuntime(),
        periodic_callback_factory=lambda callback, interval: DummyPoller(),
    )

    runner.start("Manual")
    assert worker_started.wait(timeout=1.0)
    runner.stop()

    assert shutdown_after_worker["value"] is True
    assert runner.is_running is False


def test_runner_repeated_start_is_a_no_op() -> None:
    notebook_session_state, _notebook_view, notebook_runner, _wi, _bq = (
        _load_ui_components()
    )
    state = notebook_session_state()
    worker_started = threading.Event()
    worker_can_exit = threading.Event()
    running_updates: list[bool] = []

    class DummyPoller:
        def start(self) -> None:
            return

        def stop(self) -> None:
            return

    class FakeRuntime:
        def initialize(
            self, q1: float, q2: float, sp1_c: float, sp2_c: float
        ) -> RuntimeSnapshot:
            return RuntimeSnapshot(
                time_s=np.array([0.0]),
                q1=np.array([q1]),
                q2=np.array([q2]),
                t1_c=np.array([22.0]),
                t2_c=np.array([22.0]),
                sp1_c=np.array([sp1_c]),
                sp2_c=np.array([sp2_c]),
            )

        def shutdown(self) -> None:
            return

        def step_manual(self, q1: float, q2: float) -> RuntimeSnapshot:
            worker_started.set()
            worker_can_exit.wait(timeout=1.0)
            return RuntimeSnapshot(
                time_s=np.array([0.0]),
                q1=np.array([q1]),
                q2=np.array([q2]),
                t1_c=np.array([22.0]),
                t2_c=np.array([22.0]),
                sp1_c=np.array([30.0]),
                sp2_c=np.array([30.0]),
            )

    runner = notebook_runner(
        state=state,
        backend_factory=lambda: cast(Any, None),
        on_snapshot=lambda snapshot: None,
        on_running_change=running_updates.append,
        runtime_factory=lambda config: FakeRuntime(),
        periodic_callback_factory=lambda callback, interval: DummyPoller(),
    )

    runner.start("Manual")
    assert worker_started.wait(timeout=1.0)
    first_worker = runner._worker
    runner.start("Manual")

    assert runner._worker is first_worker

    worker_can_exit.set()
    runner.stop()

    assert running_updates == [True, False]


def test_runner_captures_mpc_failure() -> None:
    notebook_session_state, _notebook_view, notebook_runner, _wi, _bq = (
        _load_ui_components()
    )
    state = notebook_session_state(mode="MPC")

    class DummyPoller:
        def start(self) -> None:
            return

        def stop(self) -> None:
            return

    class FakeRuntime:
        def initialize(
            self, q1: float, q2: float, sp1_c: float, sp2_c: float
        ) -> RuntimeSnapshot:
            return RuntimeSnapshot(
                time_s=np.array([0.0]),
                q1=np.array([q1]),
                q2=np.array([q2]),
                t1_c=np.array([22.0]),
                t2_c=np.array([22.0]),
                sp1_c=np.array([sp1_c]),
                sp2_c=np.array([sp2_c]),
            )

        def shutdown(self) -> None:
            return

        def step_mpc(
            self,
            q1: float,
            q2: float,
            sp1_c: float,
            sp2_c: float,
            config: Any,
        ) -> tuple[RuntimeSnapshot, float, float]:
            raise ControllerSolveError("solver failed")

    runner = notebook_runner(
        state=state,
        backend_factory=lambda: cast(Any, None),
        on_snapshot=lambda snapshot: None,
        on_running_change=lambda running: None,
        runtime_factory=lambda config: FakeRuntime(),
        periodic_callback_factory=lambda callback, interval: DummyPoller(),
    )

    runner.start("MPC")
    worker = runner._worker
    assert worker is not None
    worker.join(timeout=1.0)
    runner.poll()

    assert isinstance(runner.worker_error, ControllerSolveError)
