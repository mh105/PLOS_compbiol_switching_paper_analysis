# Testing script for kalman filtering and smoothing functions in pykalman
# Author: Alex He; last edit: 12/28/2021

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from codetiming import Timer
from pykalman import kalman, djkalman


def test_kalman():
    input_data = scipy.io.loadmat("kalman_inputs.mat")
    for elem in ('__header__', '__version__', '__globals__'):
        input_data.pop(elem)  # pop the extra system info key-value pairs

    # run the python kalman function on the given inputs
    with Timer():
        x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp = kalman(**input_data)

    # compare with MATLAB results
    output_data = scipy.io.loadmat("kalman_outputs.mat")
    for elem in ('__header__', '__version__', '__globals__'):
        output_data.pop(elem)  # pop the extra system info key-value pairs

    fig, axs = plt.subplots(5, 2, tight_layout=True)
    fig.suptitle('kalman', y=0.995, fontweight="bold", fontsize=16)
    ax_num = 0

    for elem in output_data.keys():
        py_var = locals()[elem]
        ml_var = output_data[elem]

        axs_index = np.unravel_index(ax_num, axs.shape, 'F')

        if elem == 'fy_t_interp':
            assert np.isnan(py_var), "NaN fy_t_interp failed in Python"
            assert np.isnan(ml_var), "Nan fy_t_interp failed in MATLAB"
            axs[axs_index].axvline(x=0, color='b', linestyle='--', linewidth=4)
            axs[axs_index].set_xlim([-1, 1])
        else:
            assert py_var.shape == ml_var.shape, "Dimensions don't match between MATLAB and Python outputs"
            if (py_var == ml_var).all():
                axs[axs_index].axvline(x=0, color='b', linestyle='--', linewidth=4)
                axs[axs_index].set_xlim([-1, 1])
            else:
                var_diff = py_var.flatten() - ml_var.flatten()
                axs[axs_index].hist(var_diff)

        axs[axs_index].set_title(elem, fontweight="bold")
        ax_num += 1

    fig.show()


def test_djkalman():
    input_data = scipy.io.loadmat("kalman_inputs.mat")
    for elem in ('__header__', '__version__', '__globals__'):
        input_data.pop(elem)  # pop the extra system info key-value pairs

    # run the python djkalman function on the given inputs
    with Timer():
        x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp = djkalman(**input_data)

    # compare with MATLAB results
    output_data = scipy.io.loadmat("djkalman_outputs.mat")
    for elem in ('__header__', '__version__', '__globals__'):
        output_data.pop(elem)  # pop the extra system info key-value pairs

    fig, axs = plt.subplots(5, 2, tight_layout=True)
    fig.suptitle('djkalman', y=0.995, fontweight="bold", fontsize=16)
    ax_num = 0

    for elem in output_data.keys():
        py_var = locals()[elem]
        ml_var = output_data[elem]

        axs_index = np.unravel_index(ax_num, axs.shape, 'F')

        assert py_var.shape == ml_var.shape, "Dimensions don't match between MATLAB and Python outputs"
        if (py_var == ml_var).all():
            axs[axs_index].axvline(x=0, color='b', linestyle='--', linewidth=4)
            axs[axs_index].set_xlim([-1, 1])
        else:
            var_diff = py_var.flatten() - ml_var.flatten()
            axs[axs_index].hist(var_diff)

        axs[axs_index].set_title(elem, fontweight="bold")
        ax_num += 1

    fig.show()


if __name__ == "__main__":
    test_kalman()
    test_djkalman()
    print('Pykalman tests finished without exception.')
    plt.show(block=True)
