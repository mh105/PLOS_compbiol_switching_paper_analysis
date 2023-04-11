# djkalman
(De Jong version) Kalman Filtering and Smoothing
---
This is the [De Jong 1989] Kalman filter and fixed-interval smoother.
Multivariate observation data y is supported. 

Reference:
[De Jong, P. (1989). Smoothing and interpolation with the
state-space model. Journal of the American Statistical
Association, 84(408), 1085-1088](https://doi.org/10.2307/2290087)

Authors: Alex He, Proloy Das, Ran Liu; Last edit: 01/05/2022

---
<p align="center">
    Observation Equation: y<sub>t</sub>~ G x<sub>t</sub> + N(0,R)<br>
    State Equation: x<sub>t+1</sub> ~ F x<sub>t</sub> + N(0,Q)
</p>

**State Equation parameters**<br>
-   F double {mustBeNumeric, mustBeSquare} -- state transition matrix<br>
-   Q double {mustBeNumeric, mustBeSquare} -- state noise covariance<br>
-   mu0 double {mustBeNumeric} -- initial state means at t=0<br>
-   Q0 double {mustBeNumeric, mustBeSquare} -- initial state noise covariance at t=0<br>

**Observation Equation parameters**<br>
-   G double {mustBeNumeric} -- observation matrix<br>
-   R double {mustBeNumeric} -- observation noise (co)variance<br>
-   y double {mustBeNumeric} -- observed data time series<br>

---

## Install Pykalman 
**Pykalman** is python package containing Kalman filteirng and smoothing methods developed in this project. Both classical Kalman and De Jong versions are implemented, along with convergent steady-state version using CUDA GPU processing.

To install **Pykalman**, make sure you activate in terminal the conda environment to which you want to install, then: 

1. Fork, clone, or download this repo to your working OS
2. ``` cd ~/djkalman/python ```
3. ``` pip install . ```

**Pykalman** is now installed to the activated conda envrionment. To test the installed package, you can: 

1. ``` cd ~/djkalman/python/pykalman/test ```
2. ``` python test_pykalman.py ``` to run the test script

You should see two time calls and a message ```Pykalman tests finished without exception.```, along with two sets of histograms showing the differences between python and MATLAB implementations of kalman and djkalman filtering and smoothing on example input data.

_Note that you will also need a few other packages for running the test functions - you can install them with ```pip install matplotlib, scipy, codetiming``` if not already installed, but they are not necessary for usage of **Pykalman** functions._

## Use Pykalman 
Once the package is installed, you can use the functions similiar to any other packages in python:
```
from pykalman import kalman, djkalman, djkalman_conv_torch 
```

