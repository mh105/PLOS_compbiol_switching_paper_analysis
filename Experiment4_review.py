"""
Experiment 4 - VB learning comparisons
Extend simulations to bi-variate observed data and a different switching
mechanism: i.e., we have a single evolving state vector, but the SSM
parameters (specifically, an entry of the transition matrix) switch.

We will again directly compare improved VB learning algorithm with the
original one and traditional switching methods, including learning.

02/22/2023
Update made for reviewer's comments to check learnt parameters
"""

import numpy as np
import somata
import scipy.io
from somata.switching import switching
from Experiment4_VBS_class import BVAR1VBS
import matplotlib.pyplot as plt

# Simulation parameters
niter = 200
T = 200
F1 = np.array([[0.5, 0], [0, 0.5]])  # transition matrix for model 1
F2 = np.array([[0.5, 0.5], [0, 0.5]])  # transition matrix for model 2
p = 2  # number of hidden states
q = 2  # number of channels
Q = np.eye(p) * 2  # state noise covariance matrix
G = np.eye(p)  # observation matrix
R = np.eye(q) * 0.1  # observation noise covariance matrix
dwell_prob = 0.95  # HMM state dwell probability

original_percent_correct = np.zeros(niter)
improved_percent_correct = np.zeros(niter)
random_percent_correct = np.zeros(niter)
static_percent_correct = np.zeros(niter)
gpb1_percent_correct = np.zeros(niter)
gpb2_percent_correct = np.zeros(niter)
imm_percent_correct = np.zeros(niter)
learnt_F = np.zeros((2, niter, F1.shape[0], F1.shape[1]))
learnt_var = np.zeros((2, niter, Q.shape[0], Q.shape[1]))
learnt_R = np.zeros((2, niter, R.shape[0], R.shape[1]))
learnt_dwellp = np.zeros((2, niter))

for n in range(niter):
    print(n)

    # Generate a single SSM with two AR(1) and a discrete state HMM for switching
    x = np.hstack([np.random.normal(scale=np.sqrt(1), size=(p, 1)), np.zeros((p, T-1))])
    s = np.hstack([np.random.choice([1, -1]), np.zeros(T-1)])
    for ii in range(1, T):
        if np.random.uniform() >= dwell_prob:
            s[ii] = s[ii-1] * -1  # switch
        else:
            s[ii] = s[ii-1]  # stay

        if s[ii] == -1:
            x[:, ii] = F1 @ x[:, ii-1] + np.random.multivariate_normal(np.zeros(p), Q)
        else:
            x[:, ii] = F2 @ x[:, ii-1] + np.random.multivariate_normal(np.zeros(p), Q)
    s[s == -1] = 0

    # Generate observed data
    y = G @ x + np.random.multivariate_normal(np.zeros(q), R, size=T).T

    """ Introduce uncertainties in parameters to initialize VB learning """
    current_F2 = np.random.uniform(size=(2, 2))*0.2 + 0.4  # 0.4-0.6 in each element
    current_F2[1, 0] = 0

    current_F1 = current_F2.copy()
    current_F1[0, 1] = 0

    current_Q = np.random.uniform(size=(2, 2))*2 + 1  # 1-3 in diagonal entries
    current_Q[0, 1] = 0
    current_Q[1, 0] = 0

    current_R = np.random.uniform(size=(2, 2))*0.19 + 0.01  # 0.01-0.2 in diagonal entries
    current_R[0, 1] = 0
    current_R[1, 0] = 0

    current_dwell_prob = np.random.uniform()*0.09 + 0.9  # 0.9-0.99

    # Run VB learning - with M step
    s1 = somata.GeneralSSModel(F=(current_F1, current_F2), Q=current_Q,
                               mu0=np.zeros(p), Q0=current_Q, G=G, R=current_R, y=y)

    v1 = BVAR1VBS(s1)
    vb_results = v1.learn(dwell_prob=current_dwell_prob, original=True, return_dict=True)
    Mprob = vb_results['h_t_m']
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    original_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

    learnt_F[0, n, :, :] = vb_results['ssm_array'][1].F
    learnt_var[0, n, :, :] = vb_results['ssm_array'][1].Q
    learnt_R[0, n, :, :] = vb_results['ssm_array'][1].R
    learnt_dwellp[0, n] = np.trace(vb_results['A']) / 2

    v1 = BVAR1VBS(s1)
    vb_results = v1.learn(dwell_prob=current_dwell_prob, priors_all=None, return_dict=True)
    Mprob = vb_results['h_t_m']
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    improved_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

    learnt_F[1, n, :, :] = vb_results['ssm_array'][1].F
    learnt_var[1, n, :, :] = vb_results['ssm_array'][1].Q
    learnt_R[1, n, :, :] = vb_results['ssm_array'][1].R
    learnt_dwellp[1, n] = np.trace(vb_results['A']) / 2

    # Run comparison inference algorithms
    Mprob = np.random.rand(T)
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    random_percent_correct[n] = np.mean(abs(Mprob - s) <= 0.1)

    Mprob, _ = switching(s1, method='static', dwell_prob=current_dwell_prob)
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    static_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

    Mprob, _ = switching(s1, method='gpb1', dwell_prob=current_dwell_prob)
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    gpb1_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

    Mprob, _ = switching(s1, method='gpb2', dwell_prob=current_dwell_prob)
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    gpb2_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

    Mprob, _ = switching(s1, method='imm', dwell_prob=current_dwell_prob)
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    imm_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

save_dict = {'original_percent_correct': original_percent_correct,
             'improved_percent_correct': improved_percent_correct,
             'random_percent_correct': random_percent_correct,
             'static_percent_correct': static_percent_correct,
             'gpb1_percent_correct': gpb1_percent_correct,
             'gpb2_percent_correct': gpb2_percent_correct,
             'IMM_percent_correct': imm_percent_correct,
             'learnt_F': learnt_F,
             'learnt_var': learnt_var,
             'learnt_R': learnt_R,
             'learnt_dwellp': learnt_dwellp}
scipy.io.savemat('Experiment4_review_simulation_results.mat', save_dict)
