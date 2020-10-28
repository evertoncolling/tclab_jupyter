[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/evertoncolling/tclab_jupyter/master)

# tclab_jupyter
A jupyter based application to explore different control techniques of a simple temperature plant

**Overview**

This code was written to showcase an Arduino based Temperature Control Lab (https://apmonitor.com/pdc/index.php/Main/ArduinoTemperatureControl) for a Lecture on Advanced Control Techniques.

The TCLab system is built with two temperature sensors and two heaters. A Matlab or Python interface is provided to read the temperature data from the board and control the heaters power output. Two classes were created, one that aims to control the Arduino System (**control_arduino.py**) and another one replacing the Arduino interface with a simulator (**control_demo.py**), so the app can be used without the real hardware, as a demonstration.

This project implements four different control techniques (Manual, On-Off, PID and MPC) so the user can test and visualize the differences between them.

There is also a configurations window that presents some parameters that can be adjusted for the whole simulation or for each control technique.

The interface was build using ipywidgets and bqplot. The dynamic plant simulation is done using scipy `odeint` function, whilst the MPC is implemented using the gekko library. For more information regarding the MPC options refer to the gekko documentation (https://gekko.readthedocs.io/en/latest/).

**Dependencies**
- numpy
- scipy
- ipywidgets (https://github.com/jupyter-widgets/ipywidgets)
- bqplot==0.11.9 (https://github.com/bloomberg/bqplot)
- gekko (https://github.com/BYU-PRISM/GEKKO)
- tclab (only for the control_arduino.py)

**Usage**

Just download the `control_demo.py` (or `control_arduino.py` if you are using it with the TCLab) to your system and create a Jupyter Notebook file (.ipynb) on the same folder.

Import the module and create an object as shown below.
```python
import control_demo as cn # change to import control_arduino to use with TCLab
demo = cn.GUI()
```

To open the main window, call the app function.
```python
demo.app()
```

<p align="center">
  <img src="https://github.com/evertoncolling/tclab_jupyter/blob/master/img/APP.PNG" alt="App Screenshot">
</p>

To open the configurations window, call the config function.
```python
demo.config()
```

<p align="center">
  <img src="https://github.com/evertoncolling/tclab_jupyter/blob/master/img/CONFIG.PNG" alt="Config Screenshot">
</p>
