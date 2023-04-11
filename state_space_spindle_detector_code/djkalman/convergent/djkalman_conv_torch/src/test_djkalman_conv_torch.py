# author: Proloy Das <pdas6@mgh.harvard.edu>
import os
import torch
import scipy.io
from codetiming import Timer
from djkalman_conv_torch import djkalman_conv_torch


def test_djkalman():
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
    except NameError:  # __file__ isn't available for iPython console sessions
        dir_path = os.getcwd()
    runtime_testing = os.path.join(dir_path[:dir_path.find('djkalman')+8],
                                   'runtime_testing')  # assumes the top level repo name is djkalman
    input_data = scipy.io.loadmat(os.path.join(runtime_testing,
                                               "runtime_inputs.mat"))
    for elem in ('__header__', '__version__', '__globals__'):
        input_data.pop(elem)  # pop the extra system info key-value pairs
    for elem in input_data.keys():
        input_data[elem] = torch.as_tensor(input_data[elem], dtype=torch.float32).cuda()
    with Timer():
        djkalman_conv_torch(**input_data, conv_steps=100)


if __name__ == "__main__":
    test_djkalman()
