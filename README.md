[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# tclab_jupyter
A jupyter based application to explore different control techniques of a simple temperature plant

```python
import control_demo as cd
demo = cd.GUI()
```

```python
demo.app()
```

<p align="center">
  <img src="https://github.com/evertoncolling/tclab_jupyter/blob/master/APP.PNG" alt="App Screenshot">
</p>

```python
demo.config()
```

<p align="center">
  <img src="https://github.com/evertoncolling/tclab_jupyter/blob/master/CONFIG.PNG" alt="Config Screenshot">
</p>

**Dependencies:**
```python
from __future__ import print_function, division
from ipywidgets import widgets as wi
from IPython.display import display
import threading
import time
import numpy as np
from gekko import GEKKO
from scipy.integrate import odeint
import bqplot as bq
```
