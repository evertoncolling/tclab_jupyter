[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# tclab_jupyter
A jupyter based application to explore different control techniques of a simple temperature plant

**Overview**

This code was written to showcase an Arduino based Temperature Control Lab (https://apmonitor.com/pdc/index.php/Main/ArduinoTemperatureControl) for a Lecture on Advanced Control Techniques.

The TCLab system is built with two temperature sensors and two heaters. A Matlab or Python interface is provided to read the temperature data from the board and control the heaters power output. This implementation was modified, replacing the Arduino interface with a simulator, so the app can be used without the real hardware.

This demonstration app implements four different control techniques (Manual, On-Off, PID and MPC) so the user can test and visualize the differences between them.

There is also a configurations window that presents some parameters that can be adjusted for the whole simulation or for each control technique.

**Dependencies**
- ipywidgets
- numpy
- scipy
- gekko
- bqplot

Currently this code will not work on Mac OS due to limitations on the gekko library. This limitation can be overcomed by running Python from a docker container.

**Usage**

Just download the control_demo.py to your system and create a Jupyter Notebook file (.ipynb) on the same folder.

Then import the module and create an object as shown below.
```python
import control_demo as cd
demo = cd.GUI()
```

To open the main window, call the app function.
```python
demo.app()
```

<p align="center">
  <img src="https://github.com/evertoncolling/tclab_jupyter/blob/master/APP.PNG" alt="App Screenshot">
</p>

To open the configurations window, call the config function.
```python
demo.config()
```

<p align="center">
  <img src="https://github.com/evertoncolling/tclab_jupyter/blob/master/CONFIG.PNG" alt="Config Screenshot">
</p>

**MPC Options**

The MPC was implemented using the gekko library, for more information refer to the documentation (https://gekko.readthedocs.io/en/latest/).
