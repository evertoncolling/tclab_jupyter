from __future__ import annotations

import numpy as np
import pytest

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
