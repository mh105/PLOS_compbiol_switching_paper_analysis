"""
Experiment 6 - Oscillator simulation

To test the performance of switching with oscillator models on two parameters:
   - Dimensionality of the SSM
   - Numbers of switching SSMs
"""

import numpy as np
import scipy.io
from somata.basic_models import OscillatorModel as Osc
from somata.switching import VBSwitchModel as Vbs
from somata.source_loc.source_loc_utils import simulate_oscillation
from itertools import permutations

# Simulation parameters
niter = 200

# Oscillator parameters for simulating source activity
Fs = 100  # (Hz) sampling frequency
T = 10  # (s) total duration of simulated activity
a = 0.98  # (unitless) damping factor, only relevant if using Matsuda oscillator
Q = 3  # (Am^2) state noise covariance for the active oscillator only
mu0 = [0, 0]  # (Am) initial state mean for the active oscillator only
Q0 = Q  # (Am^2) initial state variance for the active oscillator only
R = 1  # (V^2) observation noise variance, assuming diagonal covariance matrix with the same noise for each channel
dwell_prob = 0.98  # HMM state dwell probability

neeg = 1
ntime = int(T * Fs) + 1

all_f_list = np.array([1, 10, 20, 30, 40])  # (Hz) center frequency of oscillation in Hertz

original_percent_correct = np.zeros((niter, len(all_f_list)-1))
improved_percent_correct = np.zeros((niter, len(all_f_list)-1))

for nf in range(1, len(all_f_list)):
    f_list = all_f_list[:nf+1]
    print(f_list)

    """ Construct different combinations of oscillations """
    # build the switching observation matrices
    G_list = []
    for ii in range(len(f_list)):
        G = np.zeros(len(f_list))
        G[:ii + 1] = 1
        for g in permutations(G):
            current_g = np.asarray(g)
            if np.any([np.all(current_g == ug) for ug in G_list]):
                continue  # keep only the unique ones
            else:
                G_list.append(current_g)

    # Number of switching models
    M = len(G_list)
    assert M == 2 ** len(f_list) - 1, 'The number of switching models does not follow 2^n-1.'

    for n in range(niter):
        print(n)

        """ Generate the switching variable and observed data """
        sim_xs = np.vstack([simulate_oscillation(f, a, Q, mu0, Q0, Fs, T) for f in f_list])
        s = np.hstack([np.random.choice(range(M)), np.zeros(ntime-1, dtype=int)])
        y = np.zeros(ntime)
        y[0] = G_list[s[0]] @ sim_xs[:, 0]
        for ii in range(ntime - 1):
            sample_dist = np.ones(M) * (1-dwell_prob) / (M - 1)
            sample_dist[s[ii]] = dwell_prob
            s[ii+1] = np.random.choice(range(M), p=sample_dist)
            y[ii+1] = G_list[s[ii+1]] @ sim_xs[:, ii+1]

        # add observation noise
        y += np.random.multivariate_normal(np.zeros(neeg), R * np.eye(neeg, neeg), ntime)[:, 0]

        """ Switching inference (Using true parameters, no M-step) """
        # Construct an array of switching models
        ssm_array = np.empty(M, dtype=Osc)  # mutable array
        for ii in range(M):
            f = f_list[G_list[ii] == 1]
            ssm_array[ii] = Osc(a=np.ones(f.shape) * a, freq=f, sigma2=np.ones(f.shape) * Q, y=y, Fs=Fs, R=R)

        vb_model = Vbs(ssm_array)
        vb_results = vb_model.learn(maxVB_iter=1, dwell_prob=dwell_prob, original=True, return_dict=True)
        Mprob = vb_results['h_t_m']
        original_percent_correct[n, nf-1] = np.mean(abs(np.argmax(Mprob, axis=0) - s) <= 0.1)

        # Construct an array of switching models
        ssm_array = np.empty(M, dtype=Osc)  # mutable array
        for ii in range(M):
            f = f_list[G_list[ii] == 1]
            ssm_array[ii] = Osc(a=np.ones(f.shape) * a, freq=f, sigma2=np.ones(f.shape) * Q, y=y, Fs=Fs, R=R)

        vb_model = Vbs(ssm_array)
        vb_results = vb_model.learn(maxVB_iter=1, dwell_prob=dwell_prob, priors_all=None, return_dict=True,
                                    shared_R=True, normalize_q_t=True)
        Mprob = vb_results['h_t_m']
        improved_percent_correct[n, nf-1] = np.mean(abs(np.argmax(Mprob, axis=0) - s) <= 0.1)

save_dict = {'original_percent_correct': original_percent_correct,
             'improved_percent_correct': improved_percent_correct}
scipy.io.savemat('Experiment6_simulation_results.mat', save_dict)
