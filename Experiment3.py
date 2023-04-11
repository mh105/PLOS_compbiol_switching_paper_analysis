"""
Experiment 3 - VB inference comparisons (no M steps)
Extend simulations to bi-variate observed data and a different switching
mechanism: i.e., we have a single evolving state vector, but the SSM
parameters (specifically, an entry of the transition matrix) switch.

We will again directly compare improved VB learning algorithm with the
original one and traditional switching methods for the inference part,
i.e. E step, and do not involve any M steps.
"""

import numpy as np
import somata
from somata.switching import VBSwitchModel, switching
import scipy.io
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

    # Run VB inference - no M step
    s1 = somata.GeneralSSModel(F=(F1, F2), Q=Q, mu0=np.zeros(p), Q0=Q, G=G, R=R, y=y)

    v1 = VBSwitchModel(s1)
    vb_results = v1.learn(maxVB_iter=1, dwell_prob=dwell_prob, original=True, return_dict=True)
    Mprob = vb_results['h_t_m']
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    original_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

    v1 = VBSwitchModel(s1)
    vb_results = v1.learn(maxVB_iter=1, dwell_prob=dwell_prob, priors_all=None, return_dict=True)
    Mprob = vb_results['h_t_m']
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    improved_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

    # Run comparison inference algorithms
    Mprob = np.random.rand(T)
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    random_percent_correct[n] = np.mean(abs(Mprob - s) <= 0.1)

    Mprob, _ = switching(s1, method='static', dwell_prob=dwell_prob)
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    static_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

    Mprob, _ = switching(s1, method='gpb1', dwell_prob=dwell_prob)
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    gpb1_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

    Mprob, _ = switching(s1, method='gpb2', dwell_prob=dwell_prob)
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    gpb2_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

    Mprob, _ = switching(s1, method='imm', dwell_prob=dwell_prob)
    Mprob[Mprob >= 0.5] = 1
    Mprob[Mprob < 0.5] = 0
    imm_percent_correct[n] = np.mean(abs(Mprob[1, :] - s) <= 0.1)

save_dict = {'original_percent_correct': original_percent_correct,
             'improved_percent_correct': improved_percent_correct,
             'random_percent_correct': random_percent_correct,
             'static_percent_correct': static_percent_correct,
             'gpb1_percent_correct': gpb1_percent_correct,
             'gpb2_percent_correct': gpb2_percent_correct,
             'IMM_percent_correct': imm_percent_correct}
scipy.io.savemat('Experiment3_simulation_results.mat', save_dict)


"""
Generate an example time series and segmentation results
"""
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

plt.figure()
plt.plot(s)

# Generate observed data
y = G @ x + np.random.multivariate_normal(np.zeros(q), R, size=T).T

# Run VB inference - no M step
s1 = somata.GeneralSSModel(F=(F1, F2), Q=Q, mu0=np.zeros(p), Q0=Q, G=G, R=R, y=y)

v1 = VBSwitchModel(s1)
vb_results = v1.learn(maxVB_iter=1, dwell_prob=dwell_prob, original=True, return_dict=True)
Mprob = vb_results['h_t_m']
Mprob[Mprob >= 0.5] = 1
Mprob[Mprob < 0.5] = 0
print('Annealed: ' + str(np.mean(abs(Mprob[1, :] - s) <= 0.1)))
anneal = Mprob

v1 = VBSwitchModel(s1)
vb_results = v1.learn(maxVB_iter=1, dwell_prob=dwell_prob, priors_all=None, return_dict=True)
Mprob = vb_results['h_t_m']
Mprob[Mprob >= 0.5] = 1
Mprob[Mprob < 0.5] = 0
print('Interpolated: ' + str(np.mean(abs(Mprob[1, :] - s) <= 0.1)))
plt.plot(Mprob[1, :])
interp = Mprob

# Run comparison inference algorithms
Mprob = np.random.rand(T)
Mprob[Mprob >= 0.5] = 1
Mprob[Mprob < 0.5] = 0
print('Random: ' + str(np.mean(abs(Mprob - s) <= 0.1)))
rand = Mprob

Mprob, _ = switching(s1, method='static', dwell_prob=dwell_prob)
Mprob[Mprob >= 0.5] = 1
Mprob[Mprob < 0.5] = 0
print('Static: ' + str(np.mean(abs(Mprob[1, :] - s) <= 0.1)))
static = Mprob

Mprob, _ = switching(s1, method='gpb1', dwell_prob=dwell_prob)
Mprob[Mprob >= 0.5] = 1
Mprob[Mprob < 0.5] = 0
print('GPB1: ' + str(np.mean(abs(Mprob[1, :] - s) <= 0.1)))

Mprob, _ = switching(s1, method='gpb2', dwell_prob=dwell_prob)
Mprob[Mprob >= 0.5] = 1
Mprob[Mprob < 0.5] = 0
print('GPB2: ' + str(np.mean(abs(Mprob[1, :] - s) <= 0.1)))

Mprob, _ = switching(s1, method='imm', dwell_prob=dwell_prob)
Mprob[Mprob >= 0.5] = 1
Mprob[Mprob < 0.5] = 0
print('IMM: ' + str(np.mean(abs(Mprob[1, :] - s) <= 0.1)))
imm = Mprob

save_dict = {'s': s, 'y': y, 'anneal': anneal, 'interp': interp, 'rand': rand, 'static': static, 'imm': imm}
scipy.io.savemat('Experiment3_example_simulation.mat', save_dict)
