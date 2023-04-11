"""
Experiment 4 - VB learning comparisons
Extend simulations to bi-variate observed data and a different switching
mechanism: i.e., we have a single evolving state vector, but the SSM
parameters (specifically, an entry of the transition matrix) switch.

We compare with the modern stochastic VI algorithms from Linderman's lab
"""

import numpy as np
import numpy.random as npr
from matplotlib.colors import ListedColormap

import scipy.io

import ssm
from ssm.util import find_permutation

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# Simulation parameters
niter = 200
T = 200
F1 = np.array([[0.5, 0.5], [0, 0.5]])  # transition matrix for model 1
F2 = np.array([[0.5, 0], [0, 0.5]])  # transition matrix for model 2
p = 2  # number of hidden states
q = 2  # number of channels
Q = np.eye(p) * 2  # state noise covariance matrix
G = np.eye(p)  # observation matrix
R = np.eye(q) * 0.1  # observation noise covariance matrix
dwell_prob = 0.95  # HMM state dwell probability
K = 2  # number of switching LDS

npr.seed(0)
cmap_limited = ListedColormap(colors[0:K])

svi_mf_percent_correct = np.zeros(niter)
svi_struct_percent_correct = np.zeros(niter)
lem_percent_correct = np.zeros(niter)

for n in range(niter):
    print(n)

    # Generate a single SSM with two AR(1) and a discrete state HMM for switching
    x = np.hstack([np.random.normal(scale=np.sqrt(1), size=(p, 1)), np.zeros((p, T-1))])
    s = np.hstack([np.random.choice([1, -1]), np.zeros(T-1, dtype=int)])
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
    y = y.T

    print("Fitting SLDS with BBVI and Mean-Field Posterior")
    slds = ssm.SLDS(q, K, p, emissions="gaussian_id")
    slds.dynamics.As[0] = F1
    slds.dynamics.As[1] = F2
    slds.dynamics.Sigmas = np.tile(Q[0, 0] * np.eye(p)[None, :, :], (K, 1, 1))
    slds.emissions.inv_etas[0][:] = np.log(R[0, 0])

    # Fit the model using BBVI with a mean field variational posterior
    q_mf_elbos, q_mf = slds.fit(y, method="bbvi",
                                variational_posterior="mf",
                                num_iters=1000)

    # Get the posterior mean of the continuous states
    q_mf_x = q_mf.mean[0]

    # Find the permutation that matches the true and inferred states
    slds.permute(find_permutation(s, slds.most_likely_states(q_mf_x, y)))
    q_mf_z = slds.most_likely_states(q_mf_x, y)

    svi_mf_percent_correct[n] = np.mean(abs(s - q_mf_z) < 0.1)

    print("Fitting SLDS with BBVI using structured variational posterior")
    slds = ssm.SLDS(q, K, p, emissions="gaussian_id")
    slds.dynamics.As[0] = F1
    slds.dynamics.As[1] = F2
    slds.dynamics.Sigmas = np.tile(Q[0, 0] * np.eye(p)[None, :, :], (K, 1, 1))
    slds.emissions.inv_etas[0][:] = np.log(R[0, 0])

    # Fit the model using SVI with a structured variational posterior
    q_struct_elbos, q_struct = slds.fit(y, method="bbvi",
                                        variational_posterior="tridiag",
                                        num_iters=1000)

    # Get the posterior mean of the continuous states
    q_struct_x = q_struct.mean[0]

    # Find the permutation that matches the true and inferred states
    slds.permute(find_permutation(s, slds.most_likely_states(q_struct_x, y)))
    q_struct_z = slds.most_likely_states(q_struct_x, y)

    svi_struct_percent_correct[n] = np.mean(abs(s - q_struct_z) < 0.1)

    print("Fitting SLDS with Laplace-EM")
    slds = ssm.SLDS(q, K, p, emissions="gaussian_id")
    slds.dynamics.As[0] = F1
    slds.dynamics.As[1] = F2
    slds.dynamics.Sigmas = np.tile(Q[0, 0] * np.eye(p)[None, :, :], (K, 1, 1))
    slds.emissions.inv_etas[0][:] = np.log(R[0, 0])

    # Fit the model using Laplace-EM with a structured variational posterior
    q_lem_elbos, q_lem = slds.fit(y, method="laplace_em",
                                  variational_posterior="structured_meanfield",
                                  num_iters=100, alpha=0.0)

    # Get the posterior mean of the continuous states
    q_lem_x = q_lem.mean_continuous_states[0]

    # Find the permutation that matches the true and inferred states
    slds.permute(find_permutation(s, slds.most_likely_states(q_lem_x, y)))
    q_lem_z = slds.most_likely_states(q_lem_x, y)

    lem_percent_correct[n] = np.mean(abs(s - q_lem_z) < 0.1)

    # # Plot the true and inferred states
    # titles = ["True", "Laplace-EM", "SVI with Structured MF", "SVI with MF"]
    # states_list = [s, q_lem_z, q_struct_z, q_mf_z]
    # fig, axs = plt.subplots(4, 1, figsize=(8, 6))
    # for (i, ax, states) in zip(range(len(axs)), axs, states_list):
    #     ax.imshow(states[None, :], aspect="auto", cmap=cmap_limited)
    #     ax.set_yticks([])
    #     ax.set_title(titles[i])
    #     if i < (len(axs) - 1):
    #         ax.set_xticks([])
    #
    # plt.suptitle("True and Inferred States for Different Fitting Methods", va="baseline")
    # plt.tight_layout()

save_dict = {'svi_mf_percent_correct': svi_mf_percent_correct,
             'svi_struct_percent_correct': svi_struct_percent_correct,
             'lem_percent_correct': lem_percent_correct}
scipy.io.savemat('Experiment4_svi_lem_results.mat', save_dict)
