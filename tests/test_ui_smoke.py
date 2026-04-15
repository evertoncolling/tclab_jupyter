from __future__ import annotations

import threading
from types import MethodType
from typing import Any, cast

import numpy as np
import pytest

from tclab_jupyter.controllers import ControllerSolveError
from tclab_jupyter.runtime import RuntimeSnapshot


def test_notebook_gui_instantiates_when_optional_deps_are_available() -> None:
    pytest.importorskip("ipywidgets")
    pytest.importorskip("bqplot")
    from tclab_jupyter.ui.notebook import SimulationNotebookApp

    app = SimulationNotebookApp()
    assert app._mode.value == "Manual"


def test_manual_controls_are_not_overwritten_by_render() -> None:
    pytest.importorskip("ipywidgets")
    pytest.importorskip("bqplot")
    from tclab_jupyter.ui.notebook import SimulationNotebookApp

    app = SimulationNotebookApp()
    app._wQ1.value = 12.5
    snapshot = RuntimeSnapshot(
        time_s=np.array([0.0, 4.0]),
        q1=np.array([0.0, 0.0]),
        q2=np.array([0.0, 0.0]),
        t1_c=np.array([22.0, 22.1]),
        t2_c=np.array([22.0, 22.2]),
        sp1_c=np.array([30.0, 30.0]),
        sp2_c=np.array([30.0, 30.0]),
    )
    app._render(snapshot)
    assert app._wQ1.value == 12.5


def test_render_updates_time_axis_as_moving_window() -> None:
    pytest.importorskip("ipywidgets")
    pytest.importorskip("bqplot")
    from tclab_jupyter.ui.notebook import SimulationNotebookApp

    app = SimulationNotebookApp()
    snapshot = RuntimeSnapshot(
        time_s=np.array([0.0, 300.0, 600.0]),
        q1=np.array([0.0, 10.0, 20.0]),
        q2=np.array([0.0, 0.0, 0.0]),
        t1_c=np.array([22.0, 23.0, 24.0]),
        t2_c=np.array([22.0, 23.0, 24.0]),
        sp1_c=np.array([30.0, 30.0, 30.0]),
        sp2_c=np.array([30.0, 30.0, 30.0]),
    )
    app._render(snapshot)
    assert app._time_scale.min == pytest.approx(1.666666666666666)
    assert app._time_scale.max == pytest.approx(10.0)


def test_render_keeps_progress_labels_as_strings() -> None:
    pytest.importorskip("ipywidgets")
    pytest.importorskip("bqplot")
    from tclab_jupyter.ui.notebook import SimulationNotebookApp

    app = SimulationNotebookApp()
    snapshot = RuntimeSnapshot(
        time_s=np.array([0.0, 4.0]),
        q1=np.array([0.0, 0.0]),
        q2=np.array([0.0, 0.0]),
        t1_c=np.array([22.0, 22.1]),
        t2_c=np.array([22.0, 22.2]),
        sp1_c=np.array([30.0, 30.0]),
        sp2_c=np.array([30.0, 30.0]),
    )
    app._render(snapshot)
    assert isinstance(app._LT1.value, str)
    assert isinstance(app._LT2.value, str)


def test_worker_queues_snapshot_until_ui_poll_renders() -> None:
    pytest.importorskip("ipywidgets")
    pytest.importorskip("bqplot")
    from tclab_jupyter.ui.notebook import SimulationNotebookApp

    app = SimulationNotebookApp()
    snapshot = RuntimeSnapshot(
        time_s=np.array([0.0, 4.0]),
        q1=np.array([0.0, 0.0]),
        q2=np.array([0.0, 0.0]),
        t1_c=np.array([22.0, 22.1]),
        t2_c=np.array([22.0, 22.2]),
        sp1_c=np.array([30.0, 30.0]),
        sp2_c=np.array([30.0, 30.0]),
    )
    render_calls: list[RuntimeSnapshot] = []
    original_render = app._render

    def track_render(item: RuntimeSnapshot) -> None:
        render_calls.append(item)
        original_render(item)

    app._render = cast(Any, track_render)

    def worker() -> None:
        app._enqueue_snapshot(snapshot)

    app._run_worker(worker)
    assert render_calls == []
    app._drain_ui_queue()
    assert render_calls == [snapshot]


def test_stop_waits_for_worker_before_runtime_shutdown() -> None:
    pytest.importorskip("ipywidgets")
    pytest.importorskip("bqplot")
    from tclab_jupyter.ui.notebook import SimulationNotebookApp

    app = SimulationNotebookApp()
    worker_released = threading.Event()
    worker_started = threading.Event()
    shutdown_after_worker = {"value": False}

    class FakeRuntime:
        def initialize(
            self,
            q1: float,
            q2: float,
            sp1: float,
            sp2: float,
        ) -> RuntimeSnapshot:
            return RuntimeSnapshot(
                time_s=np.array([0.0]),
                q1=np.array([q1]),
                q2=np.array([q2]),
                t1_c=np.array([22.0]),
                t2_c=np.array([22.0]),
                sp1_c=np.array([sp1]),
                sp2_c=np.array([sp2]),
            )

        def shutdown(self) -> None:
            shutdown_after_worker["value"] = worker_released.is_set()

    app._build_runtime = cast(Any, lambda: FakeRuntime())
    app._start_ui_poller = cast(Any, lambda: None)

    def blocking_worker(self) -> None:
        worker_started.set()
        self._stop_event.wait()
        worker_released.set()

    app._work_man = cast(Any, MethodType(blocking_worker, app))
    app._play_click(None)
    assert worker_started.wait(timeout=1.0)
    app._stop_worker()
    assert shutdown_after_worker["value"] is True
    assert app._worker is None
    assert app._runtime is None


def test_repeated_start_while_worker_is_running_is_a_no_op() -> None:
    pytest.importorskip("ipywidgets")
    pytest.importorskip("bqplot")
    from tclab_jupyter.ui.notebook import SimulationNotebookApp

    app = SimulationNotebookApp()
    worker_can_exit = threading.Event()
    worker_started = threading.Event()

    class FakeRuntime:
        def initialize(
            self,
            q1: float,
            q2: float,
            sp1: float,
            sp2: float,
        ) -> RuntimeSnapshot:
            return RuntimeSnapshot(
                time_s=np.array([0.0]),
                q1=np.array([q1]),
                q2=np.array([q2]),
                t1_c=np.array([22.0]),
                t2_c=np.array([22.0]),
                sp1_c=np.array([sp1]),
                sp2_c=np.array([sp2]),
            )

        def shutdown(self) -> None:
            return

    app._build_runtime = cast(Any, lambda: FakeRuntime())
    app._start_ui_poller = cast(Any, lambda: None)

    def blocking_worker(self) -> None:
        worker_started.set()
        worker_can_exit.wait(timeout=1.0)

    app._work_man = cast(Any, MethodType(blocking_worker, app))
    app._play_click(None)
    assert worker_started.wait(timeout=1.0)
    first_worker = app._worker
    app._play_click(None)
    assert app._worker is first_worker
    worker_can_exit.set()
    assert first_worker is not None
    first_worker.join(timeout=1.0)
    app._finalize_worker()


def test_mpc_worker_failure_is_captured_and_surfaced() -> None:
    pytest.importorskip("ipywidgets")
    pytest.importorskip("bqplot")
    from tclab_jupyter.ui.notebook import SimulationNotebookApp

    app = SimulationNotebookApp()

    class FakeRuntime:
        def step_mpc(self, *args, **kwargs):
            raise ControllerSolveError("solver failed")

    app._runtime = cast(Any, FakeRuntime())
    app._run_worker(app._work_mpc)
    app._drain_ui_queue()
    assert isinstance(app._worker_error, ControllerSolveError)
