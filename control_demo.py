#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:02:13 2019

@author: evertoncolling
@email: evertoncolling@gmail.com
@licence: MIT
"""

from __future__ import print_function, division
from ipywidgets import widgets as wi
from IPython.display import display
import threading
import time
import numpy as np
from gekko import GEKKO
from scipy.integrate import odeint
import bqplot as bq


class GUI(object):
    """
    Class that defines the _GUI applications
    """
    def __init__(self):
        """
        Initialize the _GUI elements
        """
        #######################################################################
        #                                               PLOTTING CONFIGURATION
        #######################################################################
        colors = [
            '#1f77b4',  # T1
            '#5ebdff',  # T1_SP
            '#d62728',  # T2
            '#ff7f0e',  # T2_SP
            '#1f77b4',  # Q1
            '#d62728'   # Q2
        ]

        # Create scales
        t_sc = bq.LinearScale()
        T_sc = bq.LinearScale(min=20., max=60.)
        Q_sc = bq.LinearScale(min=0., max=100.)
        T_sc_set = {'x': t_sc, 'y': T_sc}
        Q_sc_set = {'x': t_sc, 'y': Q_sc}

        # Create Axis
        ax_x1 = bq.Axis(label="", scale=t_sc, visible=False)  # upper
        ax_x2 = bq.Axis(label="time [min]", scale=t_sc)       # down
        ax_T1 = bq.Axis(label="Temperature 1 [C]", scale=T_sc,
                        orientation="vertical", label_color=colors[0])
        ax_T2 = bq.Axis(label="Temperature 2 [C]", scale=T_sc,
                        orientation="vertical", label_color=colors[2])
        ax_Q = bq.Axis(label="Heater Output [%]", scale=Q_sc,
                       orientation="vertical", label_color="black")

        # Create Lines/Markers
        self._T1_meas = bq.Scatter(x=[], y=[], scales=T_sc_set,
                                   marker='circle', colors=[colors[0]],
                                   display_legend=False, default_size=20)
        self._T2_meas = bq.Scatter(x=[], y=[], scales=T_sc_set,
                                   marker='circle', colors=[colors[2]],
                                   display_legend=False, default_size=20)

        self._T1_set_point = bq.Lines(x=[], y=[], scales=T_sc_set,
                                      stroke_width=4, colors=[colors[1]],
                                      interpolation='step-before',
                                      display_legend=False)
        self._T2_set_point = bq.Lines(x=[], y=[], scales=T_sc_set,
                                      stroke_width=4, colors=[colors[3]],
                                      interpolation='step-before',
                                      display_legend=False)

        self._u1 = bq.Lines(x=[], y=[], scales=Q_sc_set,
                            stroke_width=2, colors=[colors[4]],
                            interpolation='step-before',
                            display_legend=True, labels=['Heater 1'])
        self._u2 = bq.Lines(x=[], y=[], scales=Q_sc_set,
                            stroke_width=2, colors=[colors[5]],
                            interpolation='step-before',
                            display_legend=True, labels=['Heater 2'])

        # Mix everything and create figures
        fig_lay_up = wi.Layout(width="400px", height="240px")
        fig_lay_down = wi.Layout(width="400px", height="290px")

        box1 = dict(top=10, bottom=0, left=60, right=5)
        box2 = dict(top=10, bottom=0, left=60, right=20)
        box3 = dict(top=10, bottom=50, left=60, right=10)

        fig1 = bq.Figure(layout=fig_lay_up, axes=[ax_x1, ax_T1],
                         marks=[self._T1_meas, self._T1_set_point],
                         fig_margin=box1)

        fig2 = bq.Figure(layout=fig_lay_up, axes=[ax_x1, ax_T2],
                         marks=[self._T2_meas, self._T2_set_point],
                         fig_margin=box2)

        fig3 = bq.Figure(layout=fig_lay_down, axes=[ax_x2, ax_Q],
                         marks=[self._u1, self._u2], fig_margin=box3,
                         legend_location="left",
                         legend_style={"fill": "white",
                                       "fill-opacity": "0.7",
                                       "width": "130px"})

        #######################################################################
        #                                             SPACING WIDGETS CRIATION
        #######################################################################
        v_space = wi.Label(value="", layout=wi.Layout(height='2px'))
        h_space = wi.Label(value="", layout=wi.Layout(width='2px'))

        #######################################################################
        #                                         VARYING ARRAYS TO STORE DATA
        #######################################################################
        self._t = np.array([])
        self._Q1 = np.array([])
        self._Q2 = np.array([])
        self._T = np.array([[]]).reshape((0, 2))
        self._SP_T1 = np.array([])
        self._SP_T2 = np.array([])

        #######################################################################
        #                                                           PARAMETERS
        #######################################################################
        self._delta_t = 4.0
        self._maxtime = int(500/self._delta_t)
        self._Th0 = np.array([293.15, 293.15])
        self._Tc0 = np.array([293.15, 293.15])
        self._T1_SP = 30
        self._T2_SP = 30
        self._Q10 = 0
        self._Q20 = 0
        self._flag = False
        self._sleep = 0.5

        self._q1_dt_on_off = 0.1
        self._q2_dt_on_off = 0.1

        self._pid1_gain = 10.
        self._pid1_reset = 50.
        self._pid1_rate = 1.

        self._pid2_gain = 10.
        self._pid2_reset = 50.
        self._pid2_rate = 1.

        self._SOLVER = '1 - APOPT'
        self._CVTYPE = '1 - Deadband'

        self._T1_dt = 0.1
        self._T1_tau = 10.
        self._T2_dt = 0.1
        self._T2_tau = 10.

        self._Q1_DMAX = 30.
        self._Q1_DCOST = 1.
        self._Q2_DMAX = 30.
        self._Q2_DCOST = 1.

        #######################################################################
        #                                              OUTPUT WIDGETS CRIATION
        #######################################################################
        style = {'description_width': 'initial'}

        self._PT1 = wi.FloatProgress(value=self._Tc0[0]-273.15, min=0,
                                     max=100.0,
                                     description='<b>PV (°C):</b>',
                                     bar_style='warning',
                                     orientation='horizontal',
                                     style=style)
        self._LT1 = wi.Label(value=str(self._PT1.value))
        wi.jslink((self._PT1, 'value'), (self._LT1, 'value'))

        self._PT2 = wi.FloatProgress(value=self._Tc0[1]-273.15, min=0,
                                     max=100.0,
                                     description='<b>PV (°C):</b>',
                                     bar_style='warning',
                                     orientation='horizontal',
                                     style=style)
        self._LT2 = wi.Label(value=str(self._PT2.value))
        wi.jslink((self._PT2, 'value'), (self._LT2, 'value'))

        #######################################################################
        #                                         INTERACTION WIDGETS CREATION
        #######################################################################
        self._wQ1 = wi.FloatSlider(value=self._Q10, min=0, max=100.0, step=0.5,
                                   description='<b>Q1 (%):</b>',
                                   continuous_update=False,
                                   orientation='horizontal',
                                   readout=False, style=style,
                                   layout=wi.Layout(width='230px'))
        self._tQ1 = wi.BoundedFloatText(value=self._Q10, min=0, max=100.0,
                                        step=0.5,
                                        layout=wi.Layout(width='60px'))
        wi.jslink((self._wQ1, 'value'), (self._tQ1, 'value'))

        self._bQ1 = wi.Button(description='Set',
                              layout=wi.Layout(width='50px'))
        self._bQ1.on_click(self._Q1_click)

        self._wQ2 = wi.FloatSlider(value=self._Q20, min=0, max=100.0, step=0.5,
                                   description='<b>Q2 (%):</b>',
                                   continuous_update=False,
                                   orientation='horizontal',
                                   readout=False, style=style,
                                   layout=wi.Layout(width='230px'))
        self._tQ2 = wi.BoundedFloatText(value=self._Q20, min=0, max=100.0,
                                        step=0.5,
                                        layout=wi.Layout(width='60px'))
        wi.jslink((self._wQ2, 'value'), (self._tQ2, 'value'))

        self._bQ2 = wi.Button(description='Set',
                              layout=wi.Layout(width='50px'))
        self._bQ2.on_click(self._Q2_click)

        self._wT1 = wi.FloatSlider(value=self._T1_SP, min=20, max=60.0,
                                   step=0.5, description='<b>T1 SP:</b>',
                                   continuous_update=False,
                                   orientation='horizontal',
                                   readout=False, style=style,
                                   disabled=True,
                                   layout=wi.Layout(width='230px'))
        self._tT1 = wi.BoundedFloatText(value=self._T1_SP, min=20, max=60.0,
                                        step=0.5, disabled=True,
                                        layout=wi.Layout(width='60px'))
        wi.jslink((self._wT1, 'value'), (self._tT1, 'value'))

        self._bT1 = wi.Button(description='Set',
                              layout=wi.Layout(width='50px'),
                              disabled=True)
        self._bT1.on_click(self._T1_click)

        self._wT2 = wi.FloatSlider(value=self._T2_SP, min=20, max=60.0,
                                   step=0.5, description='<b>T2 SP:</b>',
                                   continuous_update=False,
                                   orientation='horizontal', disabled=True,
                                   readout=False, style=style,
                                   layout=wi.Layout(width='230px'))
        self._tT2 = wi.BoundedFloatText(value=self._T2_SP, min=20, max=60.0,
                                        step=0.5, disabled=True,
                                        layout=wi.Layout(width='60px'))
        wi.jslink((self._wT2, 'value'), (self._tT2, 'value'))

        self._bT2 = wi.Button(description='Set',
                              layout=wi.Layout(width='50px'), disabled=True)
        self._bT2.on_click(self._T2_click)

        #######################################################################
        #                                                       MODE SELECTION
        #######################################################################
        self._mode = wi.ToggleButtons(options=['Manual',
                                               'On-Off',
                                               'PID',
                                               'MPC'],
                                      style={'button_width': '100px'})
        self._mode.observe(self._mode_switch, names='value')

        #######################################################################
        #                                                          STOP THREAD
        #######################################################################
        self._b_stop = wi.Button(description='Stop', button_style='warning',
                                 icon='stop', layout=wi.Layout(width='100px',
                                                               height='32px'))
        self._b_stop.on_click(self._stop_click)

        #######################################################################
        #                                             START THREAD - OPEN LOOP
        #######################################################################
        self._b_play = wi.Button(description='Start', button_style='success',
                                 icon='play', layout=wi.Layout(width='100px',
                                                               height='32px'))
        self._b_play.on_click(self._play_click)

        # Join Buttons
        buttons = wi.HBox((self._b_play, h_space, self._b_stop,
                           wi.Label(value="", layout=wi.Layout(width='165px')),
                           self._mode))

        #######################################################################
        #                                                               LAYOUT
        #######################################################################
        Q1_Set = wi.HBox((h_space, self._wQ1, self._tQ1, self._bQ1),
                         layout=wi.Layout(border='solid 2px gray',
                                          width='360px'))
        T1_Set = wi.HBox((h_space, self._wT1, self._tT1, self._bT1),
                         layout=wi.Layout(border='solid 2px gray',
                                          width='360px'))
        T1_View = wi.HBox((h_space, self._PT1, self._LT1),
                          layout=wi.Layout(border='solid 2px gray',
                                           width='360px'))
        Q2_Set = wi.HBox((h_space, self._wQ2, self._tQ2, self._bQ2),
                         layout=wi.Layout(border='solid 2px gray',
                                          width='360px'))
        T2_Set = wi.HBox((h_space, self._wT2, self._tT2, self._bT2),
                         layout=wi.Layout(border='solid 2px gray',
                                          width='360px'))
        T2_View = wi.HBox((h_space, self._PT2, self._LT2),
                          layout=wi.Layout(border='solid 2px gray',
                                           width='360px'))

        co1 = wi.VBox((T1_Set, v_space, T1_View, v_space, Q1_Set))
        co2 = wi.VBox((T2_Set, v_space, T2_View, v_space, Q2_Set))
        co = wi.VBox((v_space, v_space, co1, v_space, v_space, v_space, co2))
        fig1x = wi.HBox((fig1, fig2))
        fig2x = wi.HBox((fig3, h_space, h_space, co))
        figy = wi.VBox((fig1x, v_space, fig2x),
                       layout=wi.Layout(border='solid 2px gray',
                                        width='800px'))

        #######################################################################
        #                                               APPLICATION TO DISPLAY
        #######################################################################
        self._gui = wi.VBox((buttons, v_space, figy))

        #######################################################################
        #                                                         CONFIGURATOR
        #######################################################################
        style = {'description_width': 'initial'}
        lay = wi.Layout(width='120px', margin='0 20px 0 0')

        #######################################################################
        #                                                      GENERAL OPTIONS
        #######################################################################
        self._conf11 = wi.HBox((
            wi.HTML(
                value='<p style="text-align: right;"><b>&Delta;t (s):</b></p>',
                layout=lay),
            wi.FloatSlider(value=4.0, min=1.0, max=10.0, step=0.5,
                           description='', style=style)))

        self._conf12 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;">'
                          '<b>Sleep time (s):</b></p>',
                    layout=lay),
            wi.FloatSlider(value=0.5, min=0.25, max=1.5, step=0.05,
                           description='', style=style)))

        but11 = wi.Button(description='Apply', icon='check',
                          layout=wi.Layout(width='100px', height='32px'))
        but11.on_click(self._conf_general)

        but12 = wi.Button(description='Reset', icon='refresh',
                          layout=wi.Layout(width='100px', height='32px'))
        but12.on_click(self._reset_general)

        conf13 = wi.HBox((but11, but12), layout=wi.Layout(margin='10px 0 0 0'))

        #######################################################################
        #                                                       ON-OFF OPTIONS
        #######################################################################
        lay5 = wi.Layout(width='90px', margin='0 10px 0 15px')

        conf20 = wi.Button(description='TEMPERATURE 01 / HEATER 01',
                           disabled=True,
                           layout=wi.Layout(width='320px', margin='0 0 0 0'))

        self._conf21 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;">'
                          '<b>Deadband (K):</b></p>',
                    layout=lay5),
            wi.FloatSlider(value=0.1, min=0.0, max=2.0, step=0.1,
                           description='', style=style,
                           layout=wi.Layout(width='200px'))),
            layout=wi.Layout(margin='5px 0 0 0')
        )

        box21 = wi.VBox((conf20, self._conf21),
                        layout=wi.Layout(border='solid 2px gray',
                                         width='325px',
                                         margin='10px 0 0 0px'))

        conf22 = wi.Button(description='TEMPERATURE 02 / HEATER 02',
                           disabled=True,
                           layout=wi.Layout(width='320px', margin='0 0 0 0'))

        self._conf23 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;">'
                          '<b>Deadband (K):</b></p>',
                    layout=lay5),
            wi.FloatSlider(value=0.1, min=0.0, max=2.0, step=0.1,
                           description='', style=style,
                           layout=wi.Layout(width='200px'))),
            layout=wi.Layout(margin='5px 0 0 0')
        )

        box22 = wi.VBox((conf22, self._conf23),
                        layout=wi.Layout(border='solid 2px gray',
                                         width='325px',
                                         margin='10px 0 0 10px'))

        but21 = wi.Button(description='Apply', icon='check',
                          layout=wi.Layout(width='100px', height='32px'))
        but21.on_click(self._conf_on_off)

        but22 = wi.Button(description='Reset', icon='refresh',
                          layout=wi.Layout(width='100px', height='32px'))
        but22.on_click(self._reset_on_off)

        conf24 = wi.HBox((but21, but22), layout=wi.Layout(margin='10px 0 0 0'))

        #######################################################################
        #                                                          PID OPTIONS
        #######################################################################
        lay4 = wi.Layout(width='100px', margin='0 10px 0 15px')

        conf30 = wi.Button(description='TEMPERATURE 01 / HEATER 01',
                           disabled=True,
                           layout=wi.Layout(width='330px', margin='0 0 0 0'))

        self._conf31 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;">'
                          '<b>Kc (K/%Heater):</b></p>',
                    layout=lay4),
            wi.FloatSlider(value=10.0, min=0.0, max=20.0, step=0.5,
                           description='', style=style,
                           layout=wi.Layout(width='200px'))),
            layout=wi.Layout(margin='5px 0 0 0')
        )

        self._conf32 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;"><b>tauI (s):</b></p>',
                    layout=lay4),
            wi.FloatSlider(value=50.0, min=0.0, max=200.0, step=1.0,
                           description='', style=style,
                           layout=wi.Layout(width='200px'))))

        self._conf33 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;"><b>tauD (s):</b></p>',
                    layout=lay4),
            wi.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.5,
                           description='', style=style,
                           layout=wi.Layout(width='200px'))))

        box31 = wi.VBox((conf30, self._conf31, self._conf32, self._conf33),
                        layout=wi.Layout(border='solid 2px gray',
                                         width='335px',
                                         margin='10px 0 0 0px'))

        conf34 = wi.Button(description='TEMPERATURE 02 / HEATER 02',
                           disabled=True,
                           layout=wi.Layout(width='330px', margin='0 0 0 0'))

        self._conf35 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;">'
                          '<b>Kc (K/%Heater):</b></p>',
                    layout=lay4),
            wi.FloatSlider(value=10.0, min=0.0, max=20.0, step=0.5,
                           description='', style=style,
                           layout=wi.Layout(width='200px'))),
            layout=wi.Layout(margin='5px 0 0 0')
        )

        self._conf36 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;"><b>tauI (s):</b></p>',
                    layout=lay4),
            wi.FloatSlider(value=50.0, min=0.0, max=200.0, step=1.0,
                           description='', style=style,
                           layout=wi.Layout(width='200px'))))

        self._conf37 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;"><b>tauD (s):</b></p>',
                    layout=lay4),
            wi.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.5,
                           description='', style=style,
                           layout=wi.Layout(width='200px'))))

        box32 = wi.VBox((conf34, self._conf35, self._conf36, self._conf37),
                        layout=wi.Layout(border='solid 2px gray',
                                         width='335px',
                                         margin='10px 0 0 10px'))

        but31 = wi.Button(description='Apply', icon='check',
                          layout=wi.Layout(width='100px', height='32px'))
        but31.on_click(self._conf_pid)

        but32 = wi.Button(description='Reset', icon='refresh',
                          layout=wi.Layout(width='100px', height='32px'))
        but32.on_click(self._reset_pid)

        conf38 = wi.HBox((but31, but32), layout=wi.Layout(margin='10px 0 0 0'))

        #######################################################################
        #                                                          MPC OPTIONS
        #######################################################################
        lay1 = wi.Layout(width='60px', margin='0 10px 0 25px')
        lay2 = wi.Layout(width='95px', margin='0 20px 0 10px')
        lay3 = wi.Layout(width='55px', margin='0 20px 0 10px')

        self._conf40 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;"><b>SOLVER:</b></p>',
                    layout=lay1),
            wi.Dropdown(options=['1 - APOPT', '2 - BPOPT', '3 - IPOPT'],
                        value='1 - APOPT',
                        layout=wi.Layout(width='100px')),
            wi.HTML(value='<p style="text-align: right;"><b>CVTYPE:</b></p>',
                    layout=lay1),
            wi.Dropdown(options=['1 - Deadband', '2 - Trajectory'],
                        value='1 - Deadband',
                        layout=wi.Layout(width='130px'))),
            layout=wi.Layout(margin='5px 0 0 0')
        )

        conf41 = wi.Button(description='TEMPERATURE 01',
                           disabled=True,
                           layout=wi.Layout(width='360px', margin='0 0 0 0'))

        self._conf42 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;">'
                          '<b>Deadband (K):</b></p>',
                    layout=lay2),
            wi.FloatSlider(value=0.1, min=0.1, max=1.0, step=0.1,
                           description='', style=style,
                           layout=wi.Layout(width='250px'))),
            layout=wi.Layout(margin='5px 0 0 0')
        )

        self._conf43 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;"><b>TAU:</b></p>',
                    layout=lay2),
            wi.FloatSlider(value=10.0, min=1.0, max=100.0, step=1.0,
                           description='', style=style,
                           layout=wi.Layout(width='250px'))))

        box41 = wi.VBox((conf41, self._conf42, self._conf43),
                        layout=wi.Layout(border='solid 2px gray',
                                         width='365px',
                                         margin='10px 0 0 0'))

        conf44 = wi.Button(description='TEMPERATURE 02',
                           disabled=True,
                           layout=wi.Layout(width='360px', margin='0 0 0 0'))

        self._conf45 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;">'
                          '<b>Deadband (K):</b></p>',
                    layout=lay2),
            wi.FloatSlider(value=0.1, min=0.1, max=1.0, step=0.1,
                           description='', style=style,
                           layout=wi.Layout(width='250px'))),
            layout=wi.Layout(margin='5px 0 0 0')
        )

        self._conf46 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;"><b>TAU:</b></p>',
                    layout=lay2),
            wi.FloatSlider(value=10.0, min=1.0, max=100.0, step=1.0,
                           description='', style=style,
                           layout=wi.Layout(width='250px'))))

        box42 = wi.VBox((conf44, self._conf45, self._conf46),
                        layout=wi.Layout(border='solid 2px gray',
                                         width='365px',
                                         margin='10px 0 0 0'))

        conf47 = wi.Button(description='HEATER 01',
                           disabled=True,
                           layout=wi.Layout(width='320px', margin='0 0 0 0'))

        self._conf48 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;"><b>DMAX:</b></p>',
                    layout=lay3),
            wi.FloatSlider(value=30.0, min=1.0, max=100.0, step=0.5,
                           description='', style=style,
                           layout=wi.Layout(width='250px'))),
            layout=wi.Layout(margin='5px 0 0 0')
        )

        self._conf49 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;"><b>DCOST:</b></p>',
                    layout=lay3),
            wi.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.1,
                           description='', style=style,
                           layout=wi.Layout(width='250px'))))

        box43 = wi.VBox((conf47, self._conf48, self._conf49),
                        layout=wi.Layout(border='solid 2px gray',
                                         width='325px',
                                         margin='10px 0 0 10px'))

        conf410 = wi.Button(description='HEATER 02',
                            disabled=True,
                            layout=wi.Layout(width='320px', margin='0 0 0 0'))

        self._conf411 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;"><b>DMAX:</b></p>',
                    layout=lay3),
            wi.FloatSlider(value=30.0, min=1.0, max=100.0, step=0.5,
                           description='', style=style,
                           layout=wi.Layout(width='250px'))),
            layout=wi.Layout(margin='5px 0 0 0')
        )

        self._conf412 = wi.HBox((
            wi.HTML(value='<p style="text-align: right;"><b>DCOST:</b></p>',
                    layout=lay3),
            wi.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.1,
                           description='', style=style,
                           layout=wi.Layout(width='250px'))))

        box44 = wi.VBox((conf410, self._conf411, self._conf412),
                        layout=wi.Layout(border='solid 2px gray',
                                         width='325px',
                                         margin='10px 0 0 10px'))

        but41 = wi.Button(description='Apply', icon='check',
                          layout=wi.Layout(width='100px', height='32px'))
        but41.on_click(self._conf_mpc)
        but42 = wi.Button(description='Reset', icon='refresh',
                          layout=wi.Layout(width='100px', height='32px'))
        but42.on_click(self._reset_mpc)
        conf413 = wi.HBox((but41, but42),
                          layout=wi.Layout(margin='10px 0 0 0'))

        #######################################################################
        #                                                  CONFIGURATOR LAYOUT
        #######################################################################
        tab = wi.Tab([wi.VBox((self._conf11, self._conf12,
                               wi.Label(layout=wi.Layout(height='206px')),
                               conf13)),
                      wi.VBox((wi.HBox((box21, box22)),
                               wi.Label(layout=wi.Layout(height='191px')),
                               conf24)),
                      wi.VBox((wi.HBox((box31, box32)),
                               wi.Label(layout=wi.Layout(height='127px')),
                               conf38)),
                      wi.VBox((self._conf40,
                               wi.HBox((box41, box43)),
                               wi.HBox((box42, box44)),
                               wi.Label(layout=wi.Layout(height='11px')),
                               conf413))],
                     layout=wi.Layout(width='800px', height='380px'))
        tab.set_title(0, 'General Options')
        tab.set_title(1, 'On-Off Options')
        tab.set_title(2, 'PID Options')
        tab.set_title(3, 'MPC Options')

        #######################################################################
        #                                                 DISPLAY CONFIGURATOR
        #######################################################################
        self._conf = tab

    def app(self):
        display(self._gui)

    def config(self):
        display(self._conf)

    def _conf_general(self, b):
        self._delta_t = self._conf11.children[1].value
        self._maxtime = int(500/self._delta_t)

        self._sleep = self._conf12.children[1].value

    def _reset_general(self, b):
        self._conf11.children[1].value = 4.0
        self._delta_t = self._conf11.children[1].value
        self._maxtime = int(500/self._delta_t)

        self._conf12.children[1].value = 0.5
        self._sleep = self._conf12.children[1].value

    def _conf_on_off(self, b):
        self._q1_dt_on_off = self._conf21.children[1].value

        self._q2_dt_on_off = self._conf23.children[1].value

    def _reset_on_off(self, b):
        self._conf21.children[1].value = 0.1
        self._q1_dt_on_off = self._conf21.children[1].value

        self._conf23.children[1].value = 0.1
        self._q2_dt_on_off = self._conf23.children[1].value

    def _conf_pid(self, b):
        self._pid1_gain = self._conf31.children[1].value
        self._pid1_reset = self._conf32.children[1].value
        self._pid1_rate = self._conf33.children[1].value

        self._pid2_gain = self._conf35.children[1].value
        self._pid2_reset = self._conf36.children[1].value
        self._pid2_rate = self._conf37.children[1].value

    def _reset_pid(self, b):
        self._conf31.children[1].value = 10.0  # gain
        self._conf32.children[1].value = 50.0  # reset
        self._conf33.children[1].value = 1.0   # rate
        self._pid1_gain = self._conf31.children[1].value
        self._pid1_reset = self._conf32.children[1].value
        self._pid1_rate = self._conf33.children[1].value

        self._conf35.children[1].value = 10.0  # gain
        self._conf36.children[1].value = 50.0  # reset
        self._conf37.children[1].value = 1.0   # rate
        self._pid2_gain = self._conf35.children[1].value
        self._pid2_reset = self._conf36.children[1].value
        self._pid2_rate = self._conf37.children[1].valu

    def _conf_mpc(self, b):
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

    def _reset_mpc(self, b):
        self._conf40.children[1].value = '1 - APOPT'
        self._conf40.children[3].value = '1 - Deadband'

        self._conf42.children[1].value = 0.1
        self._conf43.children[1].value = 30.
        self._conf45.children[1].value = 0.1
        self._conf46.children[1].value = 30.

        self._conf48.children[1].value = 30.
        self._conf49.children[1].value = 1.
        self._conf411.children[1].value = 30.
        self._conf412.children[1].value = 1.

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

    def _Q1_click(self, b):
        self._Q10 = self._wQ1.value

    def _Q2_click(self, b):
        self._Q20 = self._wQ2.value

    def _T1_click(self, b):
        self._T1_SP = self._wT1.value

    def _T2_click(self, b):
        self._T2_SP = self._wT2.value

    def _stop_click(self, b):
        self._flag = False
        self._mode.disabled = False

    def _play_click(self, b):
        if not self._flag:
            if self._mode.value == "Manual":
                self._flag = True
                self._mode.disabled = True
                thread = threading.Thread(target=self._work_man)
                thread.start()
            elif self._mode.value == "On-Off":
                self._flag = True
                self._mode.disabled = True
                thread = threading.Thread(target=self._work_on_off)
                thread.start()
            elif self._mode.value == "PID":
                self._flag = True
                self._mode.disabled = True
                thread = threading.Thread(target=self._work_pid)
                thread.start()
            elif self._mode.value == "MPC":
                self._flag = True
                self._mode.disabled = True
                thread = threading.Thread(target=self._work_mpc)
                thread.start()

    def _mode_switch(self, value):
        # Reinitialize parameters
        self._Th0 = np.array([293.15, 293.15])
        self._Tc0 = np.array([293.15, 293.15])
        self._T1_SP = 30
        self._T2_SP = 30
        self._Q10 = 0
        self._Q20 = 0

        # Reset figures
        self._T1_meas.x = []
        self._T1_meas.y = []
        self._T2_meas.x = []
        self._T2_meas.y = []

        self._T1_set_point.x = []
        self._T1_set_point.y = []
        self._T2_set_point.x = []
        self._T2_set_point.y = []

        self._u1.x = []
        self._u1.y = []
        self._u2.x = []
        self._u2.y = []

        # Reset controls
        self._PT1.value = self._Tc0[0]-273.15
        self._PT2.value = self._Tc0[1]-273.15
        self._wT1.value = self._T1_SP
        self._wT2.value = self._T2_SP
        self._wQ1.value = self._Q10
        self._wQ2.value = self._Q20

        if value['new'] == "Manual":
            self._wQ1.disabled = False
            self._tQ1.disabled = False
            self._bQ1.disabled = False
            self._wT1.disabled = True
            self._tT1.disabled = True
            self._bT1.disabled = True

            self._wQ2.disabled = False
            self._tQ2.disabled = False
            self._bQ2.disabled = False
            self._wT2.disabled = True
            self._tT2.disabled = True
            self._bT2.disabled = True

        else:
            self._wQ1.disabled = True
            self._tQ1.disabled = True
            self._bQ1.disabled = True
            self._wT1.disabled = False
            self._tT1.disabled = False
            self._bT1.disabled = False

            self._wQ2.disabled = True
            self._tQ2.disabled = True
            self._bQ2.disabled = True
            self._wT2.disabled = False
            self._tT2.disabled = False
            self._bT2.disabled = False

    ###########################################################################
    #                                                       _MODEL TO SIMULATE
    ###########################################################################
    def _heater(self, x, t, Q1, Q2):
        # Parameters
        U = 4.87519009 + (np.random.rand()-0.5)  # variable convection
        alpha1 = 0.00640897365
        alpha2 = 0.00310952441

        Ta = 23 + 273.15     # K
        m = 4.0/1000.0       # kg
        Cp = 0.5 * 1000.0    # J/kg-K
        A = 10.0 / 100.0**2  # Area in m^2
        As = 2.0 / 100.0**2  # Area in m^2
        eps = 0.9            # Emissivity
        sigma = 5.67e-8      # Stefan-Boltzman

        # Temperature States
        Th1 = x[0]
        Th2 = x[1]

        # Heat Transfer Exchange Between 1 and 2
        conv12 = U*As*(Th2-Th1)
        rad12 = eps*sigma*As * (Th2**4 - Th1**4)

        # Nonlinear Energy Balances
        dTh1dt = (1.0/(m*Cp)) * \
                 (U*A*(Ta-Th1) +
                  eps * sigma * A * (Ta**4 - Th1**4) +
                  conv12 + rad12 + alpha1*Q1)
        dTh2dt = (1.0/(m*Cp)) * \
                 (U*A*(Ta-Th2) +
                  eps * sigma * A * (Ta**4 - Th2**4) -
                  conv12 - rad12 + alpha2*Q2)

        return [dTh1dt, dTh2dt]

    def _sensor(self, x, t, Th1, Th2):
        # Parameter
        tau = 17.7176964

        # Temperature States
        Tc1 = x[0]
        Tc2 = x[1]

        # lag equations to emulate conduction
        dTc1dt = (-Tc1 + Th1)/tau
        dTc2dt = (-Tc2 + Th2)/tau

        return [dTc1dt, dTc2dt]

    ###########################################################################
    #                                                           PID CONTROLLER
    ###########################################################################

    # inputs -----------------------------------
    # sp = setpoint
    # pv = current temperature
    # pv_last = prior temperature
    # ierr = integral error
    # dt = time increment between measurements

    # outputs ----------------------------------
    # op = output of the PID controller
    # I = integral contribution

    def _PID(self, sp, pv, pv_last, ierr, dt, Kc=10.0, tauI=50.0, tauD=1.0):
        # Default Parameters
        # Kc   = 10.0 # K/%Heater
        # tauI = 50.0 # sec
        # tauD = 1.0  # sec

        # Parameters in terms of PID coefficients
        KP = Kc
        if tauI == 0:
            KI = 1e5
        else:
            KI = Kc/tauI
        KD = Kc*tauD

        # ubias for controller (initial heater)
        op0 = 0

        # upper and lower bounds on heater level
        ophi = 100
        oplo = 0

        # calculate the error
        error = sp-pv

        # calculate the integral error
        ierr = ierr + KI * error * dt

        # calculate the measurement derivative
        dpv = (pv - pv_last) / dt

        # calculate the PID output
        P = KP * error
        I = ierr
        D = -KD * dpv
        op = op0 + P + I + D

        # implement anti-reset windup
        if op < oplo or op > ophi:
            I = I - KI * error * dt
            # clip output
            op = max(oplo, min(ophi, op))

        # return the controller output and PID terms
        return [op, I]

    ###########################################################################
    #                                                                      MPC
    ###########################################################################
    def _MPC(self):

        m = GEKKO(remote=False)

        # 60 second time horizon, 4 sec cycle time, non-uniform
        m.time = [0, 4, 8, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90]

        # Parameters
        m.U = m.FV(value=10)
        m.tau = m.FV(value=5)
        m.alpha1 = m.FV(value=0.01)    # W / % heater
        m.alpha2 = m.FV(value=0.0075)  # W / % heater

        # Manipulated variables
        m.Q1 = m.MV(value=0)
        m.Q1.STATUS = 1   # use to control temperature
        m.Q1.FSTATUS = 0  # no feedback measurement
        m.Q1.LOWER = 0.0
        m.Q1.UPPER = 100.0
        m.Q1.DMAX = 20.0
        m.Q1.COST = 0.0
        m.Q1.DCOST = 2.0

        m.Q2 = m.MV(value=0)
        m.Q2.STATUS = 1   # use to control temperature
        m.Q2.FSTATUS = 0  # no feedback measurement
        m.Q2.LOWER = 0.0
        m.Q2.UPPER = 100.0
        m.Q2.DMAX = 20.0
        m.Q2.COST = 0.0
        m.Q2.DCOST = 2.0

        # Controlled variable
        m.TC1 = m.CV(value=22)
        m.TC1.STATUS = 1     # minimize error with setpoint range
        m.TC1.FSTATUS = 1    # receive measurement
        m.TC1.TR_INIT = 1    # reference trajectory
        m.TC1.TAU = 10       # time constant for response

        # Controlled variable
        m.TC2 = m.CV(value=22)
        m.TC2.STATUS = 1     # minimize error with setpoint range
        m.TC2.FSTATUS = 1    # receive measurement
        m.TC2.TR_INIT = 1    # reference trajectory
        m.TC2.TAU = 10       # time constant for response

        # State variables
        m.TH1 = m.SV(value=22)
        m.TH2 = m.SV(value=22)

        m.Ta = m.Param(value=23.0+273.15)     # K
        m.mass = m.Param(value=4.0/1000.0)    # kg
        m.Cp = m.Param(value=0.5*1000.0)      # J/kg-K
        m.A = m.Param(value=10.0/100.0**2)    # Area not between heaters in m^2
        m.As = m.Param(value=2.0/100.0**2)    # Area between heaters in m^2
        m.eps = m.Param(value=0.9)            # Emissivity
        m.sigma = m.Const(5.67e-8)            # Stefan-Boltzmann

        # Heater temperatures
        m.T1i = m.Intermediate(m.TH1+273.15)
        m.T2i = m.Intermediate(m.TH2+273.15)

        # Heat transfer between two heaters
        m.Q_C12 = m.Intermediate(m.U*m.As*(m.T2i-m.T1i))  # Conv
        m.Q_R12 = m.Intermediate(m.eps*m.sigma*m.As*(m.T2i**4-m.T1i**4))  # Rad

        # Semi-fundamental correlations (energy balances)
        m.Equation(m.TH1.dt() == (1.0/(m.mass*m.Cp)) *
                                 (m.U*m.A*(m.Ta-m.T1i) +
                                  m.eps * m.sigma * m.A *
                                  (m.Ta**4 - m.T1i**4) + m.Q_C12 +
                                  m.Q_R12 + m.alpha1*m.Q1))

        m.Equation(m.TH2.dt() == (1.0/(m.mass*m.Cp)) *
                                 (m.U*m.A*(m.Ta-m.T2i) +
                                  m.eps * m.sigma * m.A *
                                  (m.Ta**4 - m.T2i**4) - m.Q_C12 -
                                  m.Q_R12 + m.alpha2*m.Q2))

        # Empirical correlations (lag equations to emulate conduction)
        m.Equation(m.tau * m.TC1.dt() == -m.TC1 + m.TH1)
        m.Equation(m.tau * m.TC2.dt() == -m.TC2 + m.TH2)

        # Global Options
        m.options.IMODE = 6    # MPC
        m.options.CV_TYPE = 1  # Objective type
        m.options.NODES = 3    # Collocation nodes
        m.options.SOLVER = 3   # 1=APOPT, 3=IPOPT

        return m

    ###########################################################################
    #                                           THREADING FUNCTION - OPEN LOOP
    ###########################################################################
    def _work_man(self):
        # Paraters to start each cycle
        Th0 = self._Th0
        Tc0 = self._Tc0

        # arrays to store data
        t = np.array([])
        Q1 = np.array([])
        Q2 = np.array([])
        T = np.array([[]]).reshape((0, 2))

        t = np.append(t, np.array([0]), axis=0)
        T = np.append(T, Tc0.reshape((1, 2)), axis=0)
        Q1 = np.append(Q1, np.array([self._Q10]), axis=0)
        Q2 = np.append(Q2, np.array([self._Q20]), axis=0)

        while self._flag:

            ts = [t[-1], t[-1]+self._delta_t]
            y = odeint(self._heater, Th0, ts, args=(self._Q10, self._Q20))
            Th0 = y[-1]
            z = odeint(self._sensor, Tc0, ts, args=(Th0[0], Th0[1]))
            Tc0 = z[-1]

            # Measurement noise
            Tc_noise = np.array([
                Tc0[0] + (np.random.rand()-0.5),
                Tc0[1] + (np.random.rand()-0.5)
            ])

            if len(t) >= self._maxtime:
                t = np.delete(t, 0, 0)
                T = np.delete(T, 0, 0)
                Q1 = np.delete(Q1, 0, 0)
                Q2 = np.delete(Q2, 0, 0)

            t = np.append(t, np.array([ts[-1]]), axis=0)
            T = np.append(T, Tc_noise.reshape((1, 2)), axis=0)
            Q1 = np.append(Q1, np.array([self._Q10]), axis=0)
            Q2 = np.append(Q2, np.array([self._Q20]), axis=0)

            self._T1_meas.x = t/60
            self._T1_meas.y = T[:, 0] - 273.15
            self._PT1.value = np.round(T[-1, 0]-273.15, 1)
            self._wT1.value = np.round(T[-1, 0]-273.15, 1)

            self._T2_meas.x = t/60
            self._T2_meas.y = T[:, 1] - 273.15
            self._PT2.value = np.round(T[-1, 1]-273.15, 1)
            self._wT2.value = np.round(T[-1, 1]-273.15, 1)

            self._u1.x = t/60
            self._u1.y = Q1

            self._u2.x = t/60
            self._u2.y = Q2

            time.sleep(self._sleep)

    ###########################################################################
    #                                              THREADING FUNCTION - ON-OFF
    ###########################################################################
    def _work_on_off(self):
        # Paraters to start each cycle
        Th0 = self._Th0
        Tc0 = self._Tc0
        Q10 = self._Q10
        Q20 = self._Q20

        # arrays to store data
        t = np.array([])
        Q1 = np.array([])
        Q2 = np.array([])
        T = np.array([[]]).reshape((0, 2))
        SP_T1 = np.array([])
        SP_T2 = np.array([])

        t = np.append(t, np.array([0]), axis=0)
        T = np.append(T, Tc0.reshape((1, 2)), axis=0)
        Q1 = np.append(Q1, np.array([Q10]), axis=0)
        Q2 = np.append(Q2, np.array([Q20]), axis=0)
        SP_T1 = np.append(SP_T1, np.array([self._T1_SP]), axis=0)
        SP_T2 = np.append(SP_T2, np.array([self._T2_SP]), axis=0)

        while self._flag:

            # apply ON/OFF controller
            # heater 1
            if (Tc0[0]-273.15) < (self._T1_SP - self._q1_dt_on_off):
                Q10 = 100.0
            elif (Tc0[0]-273.15) > (self._T1_SP + self._q1_dt_on_off):
                Q10 = 0.0
            # heater 2
            if (Tc0[1]-273.15) < (self._T2_SP - self._q2_dt_on_off):
                Q20 = 100.0
            elif (Tc0[1]-273.15) > (self._T2_SP + self._q2_dt_on_off):
                Q20 = 0.0

            ts = [t[-1], t[-1]+self._delta_t]
            y = odeint(self._heater, Th0, ts, args=(Q10, Q20))
            Th0 = y[-1]
            z = odeint(self._sensor, Tc0, ts, args=(Th0[0], Th0[1]))
            Tc0 = z[-1]

            # Measurement noise
            Tc_noise = np.array([
                Tc0[0] + (np.random.rand()-0.5),
                Tc0[1] + (np.random.rand()-0.5)
            ])

            if len(t) >= self._maxtime:
                t = np.delete(t, 0, 0)
                T = np.delete(T, 0, 0)
                Q1 = np.delete(Q1, 0, 0)
                Q2 = np.delete(Q2, 0, 0)
                SP_T1 = np.delete(SP_T1, 0, 0)
                SP_T2 = np.delete(SP_T2, 0, 0)

            t = np.append(t, np.array([ts[-1]]), axis=0)
            T = np.append(T, Tc_noise.reshape((1, 2)), axis=0)
            Q1 = np.append(Q1, np.array([Q10]), axis=0)
            Q2 = np.append(Q2, np.array([Q20]), axis=0)
            SP_T1 = np.append(SP_T1, np.array([self._T1_SP]), axis=0)
            SP_T2 = np.append(SP_T2, np.array([self._T2_SP]), axis=0)

            self._T1_meas.x = t/60
            self._T1_meas.y = T[:, 0] - 273.15
            self._PT1.value = np.round(T[-1, 0]-273.15, 1)

            self._T1_set_point.x = t/60
            self._T1_set_point.y = SP_T1

            self._T2_meas.x = t/60
            self._T2_meas.y = T[:, 1] - 273.15
            self._PT2.value = np.round(T[-1, 1]-273.15, 1)

            self._T2_set_point.x = t/60
            self._T2_set_point.y = SP_T2

            self._u1.x = t/60
            self._u1.y = Q1
            self._wQ1.value = np.round(Q1[-1], 1)

            self._u2.x = t/60
            self._u2.y = Q2
            self._wQ2.value = np.round(Q2[-1], 1)

            time.sleep(self._sleep)

    ###########################################################################
    #                                                 THREADING FUNCTION - PID
    ###########################################################################
    def _work_pid(self):
        # Paraters to start each cycle
        Th0 = self._Th0
        Tc0 = self._Tc0
        Q10 = self._Q10
        Q20 = self._Q20

        # arrays to store data
        t = np.array([])
        Q1 = np.array([])
        Q2 = np.array([])
        T = np.array([[]]).reshape((0, 2))
        SP_T1 = np.array([])
        SP_T2 = np.array([])

        t = np.append(t, np.array([0]), axis=0)
        T = np.append(T, Tc0.reshape((1, 2)), axis=0)
        Q1 = np.append(Q1, np.array([Q10]), axis=0)
        Q2 = np.append(Q2, np.array([Q20]), axis=0)
        SP_T1 = np.append(SP_T1, np.array([self._T1_SP]), axis=0)
        SP_T2 = np.append(SP_T2, np.array([self._T2_SP]), axis=0)

        # Integral error
        ierr1 = 0.0
        ierr2 = 0.0

        while self._flag:

            ts = [t[-1], t[-1]+self._delta_t]
            y = odeint(self._heater, Th0, ts, args=(Q10, Q20))
            Th0 = y[-1]
            z = odeint(self._sensor, Tc0, ts, args=(Th0[0], Th0[1]))
            Tc0 = z[-1]

            # Measurement noise
            Tc_noise = np.array([
                Tc0[0] + (np.random.rand()-0.5),
                Tc0[1] + (np.random.rand()-0.5)
            ])

            if len(t) >= self._maxtime:
                t = np.delete(t, 0, 0)
                T = np.delete(T, 0, 0)
                Q1 = np.delete(Q1, 0, 0)
                Q2 = np.delete(Q2, 0, 0)
                SP_T1 = np.delete(SP_T1, 0, 0)
                SP_T2 = np.delete(SP_T2, 0, 0)

            t = np.append(t, np.array([ts[-1]]), axis=0)
            T = np.append(T, Tc_noise.reshape((1, 2)), axis=0)
            Q1 = np.append(Q1, np.array([Q10]), axis=0)
            Q2 = np.append(Q2, np.array([Q20]), axis=0)
            SP_T1 = np.append(SP_T1, np.array([self._T1_SP]), axis=0)
            SP_T2 = np.append(SP_T2, np.array([self._T2_SP]), axis=0)

            # Calculate PID output
            [Q10, ierr1] = self._PID(self._T1_SP, T[-1, 0]-273.15,
                                     T[-2, 0]-273.15, ierr1, self._delta_t,
                                     self._pid1_gain, self._pid1_reset,
                                     self._pid1_rate)
            [Q20, ierr2] = self._PID(self._T2_SP, T[-1, 1]-273.15,
                                     T[-2, 1]-273.15, ierr2, self._delta_t,
                                     self._pid2_gain, self._pid2_reset,
                                     self._pid2_rate)

            self._T1_meas.x = t/60
            self._T1_meas.y = T[:, 0] - 273.15
            self._PT1.value = np.round(T[-1, 0]-273.15, 1)

            self._T1_set_point.x = t/60
            self._T1_set_point.y = SP_T1

            self._T2_meas.x = t/60
            self._T2_meas.y = T[:, 1] - 273.15
            self._PT2.value = np.round(T[-1, 1]-273.15, 1)

            self._T2_set_point.x = t/60
            self._T2_set_point.y = SP_T2

            self._u1.x = t/60
            self._u1.y = Q1
            self._wQ1.value = np.round(Q1[-1], 1)

            self._u2.x = t/60
            self._u2.y = Q2
            self._wQ2.value = np.round(Q2[-1], 1)

            time.sleep(self._sleep)

    ###########################################################################
    #                                                 THREADING FUNCTION - MPC
    ###########################################################################
    def _work_mpc(self):
        Th0 = self._Th0
        Tc0 = self._Tc0
        Q10 = self._Q10
        Q20 = self._Q20

        # arrays to store data
        t = np.array([])
        Q1 = np.array([])
        Q2 = np.array([])
        T = np.array([[]]).reshape((0, 2))
        SP_T1 = np.array([])
        SP_T2 = np.array([])

        t = np.append(t, np.array([0]), axis=0)
        T = np.append(T, Tc0.reshape((1, 2)), axis=0)
        Q1 = np.append(Q1, np.array([Q10]), axis=0)
        Q2 = np.append(Q2, np.array([Q20]), axis=0)
        SP_T1 = np.append(SP_T1, np.array([self._T1_SP]), axis=0)
        SP_T2 = np.append(SP_T2, np.array([self._T2_SP]), axis=0)

        # Create MPC object
        m = self._MPC()

        while self._flag:
            # Change SOLVER
            if self._SOLVER == '1 - APOPT':
                m.options.SOLVER = 1
            elif self._SOLVER == '2 - BPOPT':
                m.options.SOLVER = 2
            else:
                m.options.SOLVER = 3

            # Change CVTYPE
            if self._CVTYPE == '1 - Deadband':
                m.options.CV_TYPE = 1
            else:
                m.options.CV_TYPE = 2

            # Add measurements to the MPC
            m.TC1.MEAS = T[-1, 0] - 273.15
            m.TC2.MEAS = T[-1, 1] - 273.15

            # Update Parameters
            m.TC1.TAU = self._T1_tau
            m.TC2.TAU = self._T2_tau

            m.Q1.DMAX = self._Q1_DMAX
            m.Q1.DCOST = self._Q1_DCOST
            m.Q2.DMAX = self._Q2_DMAX
            m.Q2.DCOST = self._Q2_DCOST

            # Update prediction horizon
            DT = self._delta_t
            m.time = [
                0,
                DT,
                DT*2,
                DT*3,
                DT*4,
                DT*5,
                DT*6,
                DT*7,
                DT*8,
                DT*10,
                DT*12,
                DT*15,
                DT*18,
                DT*20,
                DT*25]

            if m.options.CV_TYPE == 1:
                # Input setpoint with deadband +/- DT
                DT1 = self._T1_dt
                m.TC1.SPHI = self._T1_SP + DT1
                m.TC1.SPLO = self._T1_SP - DT1

                DT2 = self._T2_dt
                m.TC2.SPHI = self._T2_SP + DT2
                m.TC2.SPLO = self._T2_SP - DT2
            else:
                m.TC1.SP = self._T1_SP
                m.TC2.SP = self._T2_SP

            try:
                # Solve MPC
                m.solve(disp=False)
                # Check if successful solution
                if (m.options.APPSTATUS == 1):
                    # retrieve new value
                    Q10 = m.Q1.NEWVAL
                    Q20 = m.Q2.NEWVAL
            except:
                # Keep previous value
                pass

            ts = [t[-1], t[-1]+self._delta_t]
            y = odeint(self._heater, Th0, ts, args=(Q10, Q20))
            Th0 = y[-1]
            z = odeint(self._sensor, Tc0, ts, args=(Th0[0], Th0[1]))
            Tc0 = z[-1]

            # Measurement noise
            Tc_noise = np.array([
                Tc0[0] + (np.random.rand()-0.5),
                Tc0[1] + (np.random.rand()-0.5)
            ])

            if len(t) >= self._maxtime:
                t = np.delete(t, 0, 0)
                T = np.delete(T, 0, 0)
                Q1 = np.delete(Q1, 0, 0)
                Q2 = np.delete(Q2, 0, 0)
                SP_T1 = np.delete(SP_T1, 0, 0)
                SP_T2 = np.delete(SP_T2, 0, 0)

            t = np.append(t, np.array([ts[-1]]), axis=0)
            T = np.append(T, Tc_noise.reshape((1, 2)), axis=0)
            Q1 = np.append(Q1, np.array([Q10]), axis=0)
            Q2 = np.append(Q2, np.array([Q20]), axis=0)
            SP_T1 = np.append(SP_T1, np.array([self._T1_SP]), axis=0)
            SP_T2 = np.append(SP_T2, np.array([self._T2_SP]), axis=0)

            self._T1_meas.x = t/60
            self._T1_meas.y = T[:, 0] - 273.15
            self._PT1.value = np.round(T[-1, 0]-273.15, 1)

            self._T1_set_point.x = t/60
            self._T1_set_point.y = SP_T1

            self._T2_meas.x = t/60
            self._T2_meas.y = T[:, 1] - 273.15
            self._PT2.value = np.round(T[-1, 1]-273.15, 1)

            self._T2_set_point.x = t/60
            self._T2_set_point.y = SP_T2

            self._u1.x = t/60
            self._u1.y = Q1
            self._wQ1.value = np.round(Q1[-1], 1)

            self._u2.x = t/60
            self._u2.y = Q2
            self._wQ2.value = np.round(Q2[-1], 1)

            time.sleep(self._sleep)
