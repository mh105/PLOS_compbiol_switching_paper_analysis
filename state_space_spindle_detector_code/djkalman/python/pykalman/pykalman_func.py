# Kalman filtering and smoothing functions in Python

# This is a collection of Kalman filtering and smoothing functions originally coded in MATLAB now ported to Python:
# - kalman() function tracks the kalman.m under <standard>
# - djkalman() function tracks the djkalman.m under <standard>
# - djkalman_conv_torch() function is an exact copy from development version under <convergent/djkalman_conv_torch/src>

# Authors: Alex He, Proloy Das, Ran Liu
# Last edit: 01/09/2022

import torch
import numpy as np


def logdet(A):
    """Computes logdet using Schur complement.
    Non-torch version for regular usage"""
    log_det = np.log(A[0, 0])
    for i in range(0, A.shape[0]-1):
        A = A[1:, 1:] - (A[1:, 0] @ A[0, 1:]) / A[0, 0]
        log_det += np.log(A[0, 0])
    return log_det


def logdet_torch(A):
    """Computes logdet using Schur complement"""
    log_det = torch.log(A[0, 0])
    for i in range(0, A.shape[0]-1):
        A = A[1:, 1:] - torch.matmul(torch.unsqueeze(A[1:, 0], -1),
                                     torch.unsqueeze(A[0, 1:], 0))/A[0, 0]
        log_det += torch.log(A[0, 0])
    return log_det


def kalman(F, Q, mu0, Q0, G, R, y, R_weights=None):
    """
    This is the classical Kalman filter and fixed-interval smoother.
    Multivariate observation data y is supported. This implementation runs
    the kalman filtering and smoothing on CPU. This function has been heavily
    optimized for matrix computations in exchange for more intense memory use
    and arcane syntax.

    Reference:
        Shumway, R. H., & Stoffer, D. S. (1982). An approach to time
        series smoothing and forecasting using the EM algorithm.
        Journal of time series analysis, 3(4), 253-264.

        De Jong, P., & Mackinnon, M. J. (1988). Covariances for
        smoothed estimates in state space models. Biometrika, 75(3),
        601-602.

        Jazwinski, A. H. (2007). Stochastic processes and filtering
        theory. Courier Corporation.

    Authors: Alex He, Proloy Das; Last edit: 12/28/2021

    :param F: transition matrix
    :param Q: state noise covariance matrix
    :param mu0: initial state mean vector
    :param Q0: initial state covariance matrix
    :param G: observation matrix
    :param R: observation noise covariance matrix
    :param y: observed data (can be multivariate)
    :param R_weights: time-varying weights on the observation
            noise covariance (default: uniform unit weights)
    :return: x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp (nan)
    """
    # Vector dimensions
    q, T = y.shape
    p = mu0.shape[0]

    if R_weights is None:
        R = R[:, :, None] * np.ones((q, q, T))
    else:  # (index 1 corresponds to t=1, etc.)
        R = R[:, :, None] * np.reshape(np.tile(R_weights, (q ** 2, 1)), (q, q, T))

    # Kalman filtering (forward pass)
    x_t_tmin1 = np.zeros((p, T+1))  # (index 1 corresponds to t=0, etc.)
    P_t_tmin1 = np.zeros((p, p, T+1))
    K_t = np.zeros((p, q))
    x_t_t = np.zeros((p, T+1))
    P_t_t = np.zeros((p, p, T+1))
    logL = np.zeros(T)  # (index 1 corresponds to t=1)
    qlog2pi = q * np.log(2 * np.pi)

    # Initialize
    x_t_t[:, 0] = np.squeeze(mu0, axis=1)  # x_0_0
    P_t_t[:, :, 0] = Q0  # P_0_0

    # Recursion of forward pass
    for ii in range(1, T+1):  # t=1 -> t=T
        # One-step prediction
        x_t_tmin1[:, ii] = F @ x_t_t[:, ii-1]
        P_t_tmin1[:, :, ii] = F @ P_t_t[:, :, ii-1] @ F.T + Q

        # Update
        GP = G @ P_t_tmin1[:, :, ii]
        Sigma = GP @ G.T + R[:, :, ii-1]
        invSigma = np.linalg.inv(Sigma)
        dy = y[:, ii-1] - G @ x_t_tmin1[:, ii]
        K_t = GP.T @ invSigma
        x_t_t[:, ii] = x_t_tmin1[:, ii] + K_t @ dy
        P_t_t[:, :, ii] = P_t_tmin1[:, :, ii] - K_t @ GP

        # Innovation form of the log likelihood
        logL[ii-1] = -(qlog2pi + logdet(Sigma) + dy.T @ invSigma @ dy) / 2

    # Kalman smoothing (backward pass)
    x_t_n = np.zeros((p, T+1))  # (index 1 corresponds to t=0, etc.)
    P_t_n = np.zeros((p, p, T+1))
    P_t_tmin1_n = np.zeros((p, p, T+1))  # cross-covariance between t and t-1
    fy_t_interp = float('nan')  # interpolated conditional density is not available in classical kalman filtering

    # Initialize
    x_t_n[:, -1] = x_t_t[:, -1]  # x_T_T
    P_t_n[:, :, -1] = P_t_t[:, :, -1]  # P_T_T

    # Recursion of backward pass
    for ii in range(T, 0, -1):  # t=T -> t=1
        J_t = P_t_t[:, :, ii-1] @ F.T @ np.linalg.inv(P_t_tmin1[:, :, ii])
        JP = J_t @ P_t_n[:, :, ii]
        x_t_n[:, ii-1] = x_t_t[:, ii-1] + J_t @ (x_t_n[:, ii] - x_t_tmin1[:, ii])
        P_t_n[:, :, ii-1] = P_t_t[:, :, ii-1] + (JP - J_t @ P_t_tmin1[:, :, ii]) @ J_t.T
        P_t_tmin1_n[:, :, ii] = JP.T  # Cov(t,t-1) proved in De Jong & Mackinnon (1988)

    return x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp


def djkalman(F, Q, mu0, Q0, G, R, y, R_weights=None):
    """
    This is the [De Jong 1989] Kalman filter and fixed-interval smoother.
    Multivariate observation data y is supported. This implementation runs
    the kalman filtering and smoothing on CPU. This function has been heavily
    optimized for matrix computations in exchange for more intense memory use
    and arcane syntax. Note the filtered estimates are skipped since not used
    during recursion. If you need filtered estimate, refer to the equations
    derived in the MATLAB human-readable version (under <archive>) and add to
    the for loop.

    Since De Jong Kalman filtering and smoothing are not defined at t=0, we
    repeat the estimates at t=1 to extend to t=0. This is useful to allow
    updating initial state and covariance estimates in the M step.

    Reference:
        De Jong, P. (1989). Smoothing and interpolation with the
        state-space model. Journal of the American Statistical
        Association, 84(408), 1085-1088.

    Authors: Alex He, Proloy Das; Last edit: 12/28/2021

    :param F: transition matrix
    :param Q: state noise covariance matrix
    :param mu0: initial state mean vector
    :param Q0: initial state covariance matrix
    :param G: observation matrix
    :param R: observation noise covariance matrix
    :param y: observed data (can be multivariate)
    :param R_weights: time-varying weights on the observation
            noise covariance (default: uniform unit weights)
    :return: x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp
    """
    # Vector dimensions
    q, T = y.shape
    p = mu0.shape[0]

    if R_weights is None:
        R = R[:, :, None] * np.ones((q, q, T))
    else:  # (index 1 corresponds to t=1, etc.)
        R = R[:, :, None] * np.reshape(np.tile(R_weights, (q ** 2, 1)), (q, q, T))

    # Kalman filtering (forward pass)
    x_t_tmin1 = np.zeros((p, T+2))  # (index 1 corresponds to t=0, etc.)
    P_t_tmin1 = np.zeros((p, p, T+2))
    K_t = np.zeros((p, q, T+1))  # note that this is different from the classical Kalman gain by pre-multiplying with F
    e_t = np.zeros((q, T+1))
    invD_t = np.zeros((q, q, T+1))
    L_t = np.zeros((p, p, T+1))
    logL = np.zeros(T)  # (index 1 corresponds to t=1)
    qlog2pi = q * np.log(2 * np.pi)

    # Initialize
    x_t_t = mu0  # x_0_0, initialized only to keep return outputs consistent
    P_t_t = Q0  # P_0_0, initialized only to keep return outputs consistent
    x_t_tmin1[:, 1] = F @ np.squeeze(x_t_t, axis=1)  # x_1_0 (W_0*beta in De Jong 1989)
    P_t_tmin1[:, :, 1] = F @ P_t_t @ F.T + Q  # P_1_0 (V_0 in De Jong 1989)

    # Recursion of forward pass
    for ii in range(1, T+1):  # t=1 -> t=T
        # Intermediate vectors
        e_t[:, ii] = y[:, ii-1] - G @ x_t_tmin1[:, ii]  # same as dy in classical kalman
        D_t = G @ P_t_tmin1[:, :, ii] @ G.T + R[:, :, ii-1]  # same as Sigma in classical kalman
        invD_t[:, :, ii] = np.linalg.inv(D_t)
        FP = F @ P_t_tmin1[:, :, ii]
        K_t[:, :, ii] = FP @ G.T @ invD_t[:, :, ii]
        L_t[:, :, ii] = F - K_t[:, :, ii] @ G

        # One-step prediction for the next time point
        x_t_tmin1[:, ii+1] = F @ x_t_tmin1[:, ii] + K_t[:, :, ii] @ e_t[:, ii]
        P_t_tmin1[:, :, ii+1] = FP @ L_t[:, :, ii].T + Q

        # Innovation form of the log likelihood
        logL[ii-1] = -(qlog2pi + logdet(D_t) + e_t[:, ii].T @ invD_t[:, :, ii] @ e_t[:, ii]) / 2

    # remove the extra t=T+1 time point created
    x_t_tmin1 = x_t_tmin1[:, :-1]
    P_t_tmin1 = P_t_tmin1[:, :, :-1]

    # Kalman smoothing (backward pass)
    r_t = np.zeros((p, T+1))  # (index 1 corresponds to t=0, etc.)
    R_t = np.zeros((p, p, T+1))
    x_t_n = np.zeros((p, T+1))
    P_t_n = np.zeros((p, p, T+1))
    P_t_tmin1_n = np.zeros((p, p, T+1))  # cross-covariance between t and t-1
    Ip = np.eye(p)

    # Recursion of backward pass - fixed-interval smoothing
    for ii in range(T, 0, -1):  # t=T -> t=1
        # Intermediate vectors
        GD = G.T @ invD_t[:, :, ii]
        r_t[:, ii-1] = GD @ e_t[:, ii] + L_t[:, :, ii].T @ r_t[:, ii]
        R_t[:, :, ii-1] = GD @ G + L_t[:, :, ii].T @ R_t[:, :, ii] @ L_t[:, :, ii]

        # Smoothed estimates
        x_t_n[:, ii] = x_t_tmin1[:, ii] + P_t_tmin1[:, :, ii] @ r_t[:, ii-1]
        RP = R_t[:, :, ii-1] @ P_t_tmin1[:, :, ii]
        P_t_n[:, :, ii] = P_t_tmin1[:, :, ii] @ (Ip - RP)
        # Cov(t,t-1) is derived using Theorem 1 and Lemma 1, m = t, s = t+1
        P_t_tmin1_n[:, :, ii] = (Ip - RP.T) @ L_t[:, :, ii-1] @ P_t_tmin1[:, :, ii-1].T

    # Set the cross-covariance estimate at t=1. Cov(t=1,t=0) can be computed exactly using J_0.
    # But we use P_t=1_n instead to avoid inverting conditional state noise covariance.
    P_t_tmin1_n[:, :, 1] = P_t_n[:, :, 1]

    # Repeat t=1 to extend the smoothed estimates to t=0
    x_t_n[:, 0] = x_t_n[:, 1]
    P_t_n[:, :, 0] = P_t_n[:, :, 1]

    # Interpolated conditional density of y_t
    # fy_t_interp = normpdf(y_t | y_1,...,y_t-1,y_t+1,y_T)
    fy_t_interp = np.zeros(T)  # (index 1 corresponds to t=1, etc.)
    for ii in range(0, T):  # t=1 -> t=T
        n_t = invD_t[:, :, ii+1] @ e_t[:, ii+1] - K_t[:, :, ii+1].T @ r_t[:, ii+1]
        N_t = invD_t[:, :, ii+1] + K_t[:, :, ii+1].T @ R_t[:, :, ii+1] @ K_t[:, :, ii+1]
        # See De Jong 1989 Section 5, note that -logdet(N_t) is NOT a typo
        fy_t_interp[ii] = np.exp(-(qlog2pi - logdet(N_t) + n_t.T @ np.linalg.inv(N_t) @ n_t) / 2)

    return x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp


def djkalman_conv_torch(F, Q, mu0, Q0, G, R, y, conv_steps=100):
    """A verbatim port from MATLAB to Pytorch, based upon djkalman
    implementation by Alex He

    State space model is defined as follows:
        x(t) = F*x(t-1)+eta(t)   eta ~ N(0,Q) (state or transition equation)
        y(t) = G*x(t)+eps(t)     eps ~ N(0,R) (observation or measurement equation)

    djkalman_conv_torch provides a pytorch based implementation (for gpu) that
    computes the one-step prediction and the smoothed estimate, as well as
    their covariance matrices. The function uses forward and backward
    recursions, and uses a convergent approach to compute steady state version
    of the Kalman gain (and hence the covariance matrices) for reducing
    runtime.

    Author: Ran Liu <rliu20@mgh.harvard.edu> Last edit: Alex He 12-28-2021

    Input:
    -----
    F: Nx x Nx matrix
        a time-invariant transition matrix in transition equation.
    Q: Nx x Nx matrix
        time-invariant variance matrix for the error in transition equation.
    mu0: Nx x 1
        initial state vector.
    Q0: Nx x Nx
        covariance matrix of an initial state vector.
    G: Ny x Nx matrix
        a time-invariant measurement matrix in measurement equation.
    R: Ny x Ny matrix
        time-invariant variance matrix for the error in measurement equation.
    y: Ny x T matrix
        containing data (y(1), ... , y(T)).
    conv_steps: int (default 100)
        Kalman gain is updated up to this many steps

    Output:
    -------
    x_t_n: Nx x T matrix
        smoothed state vectors.
    P_t_n: Nx x Nx matrix
        SS covariance matrices of smoothed state vectors.
    P_t_tmin1_n: Nx x Nx matrix
        SS cross-covariance (lag 1) matrices of smoothed state vectors.
    logL: 1 x T vector (float)
        value of the log likelihood function of the SSM at each time point
        under assumption that observation noise eps(t) is normally distributed.
    K_t: Nx x Nx matrix
        SS Kalman gain.
    x_t_tmin1: Nx x T matrix
        one-step predicted state vectors.
    P_t_tmin1: Nx x Nx matrix
        SS mean square error of predicted state vectors.
    """
    # Vector dimensions
    q, T = y.shape
    p = mu0.shape[0]

    # Kalman filtering (forward pass)
    x_t_tmin1 = torch.zeros(p, T+2, dtype=torch.float32).cuda()  # (index 1 corresponds to t=0, etc.)
    K_t = torch.zeros(p, q, dtype=torch.float32).cuda()
    e_t = torch.zeros(q, T+1, dtype=torch.float32).cuda()
    # noinspection PyUnusedLocal
    D_t = torch.zeros(q, q, dtype=torch.float32).cuda()
    L_t = torch.zeros(p, p, dtype=torch.float32).cuda()
    logL = torch.zeros(T, dtype=torch.float32).cuda()  # (index 1 corresponds to t=1)
    qlog2pi = q * torch.log(torch.as_tensor(2 * torch.pi, dtype=torch.float32).cuda())

    # Initialize
    x_t_tmin1[:, 1] = torch.matmul(F, mu0)[:, 0]  # x_1_0 (W_0*beta in De Jong 1989)
    P_t_tmin1 = torch.matmul(torch.matmul(F, Q0), torch.transpose(F, 0, 1)) + Q  # P_1_0 (V_0 in De Jong 1989)

    # Recursion of forward pass until convergence of predicted covariance matrix
    for ii in range(1, conv_steps+1):  # t=1 -> t=conv_steps
        # Intermediate vectors
        e_t[:, ii] = y[:, ii-1] - torch.matmul(G, x_t_tmin1[:, ii])
        D_t = torch.matmul(torch.matmul(G, P_t_tmin1),
                           torch.transpose(G, 0, 1)) + R
        invD_t = torch.inverse(D_t)
        FP = torch.matmul(F, P_t_tmin1)
        K_t = torch.matmul(torch.matmul(FP, torch.transpose(G, 0, 1)), invD_t)
        L_t = F - torch.matmul(K_t, G)

        # One-step prediction for the next time point
        x_t_tmin1[:, ii+1] = (torch.matmul(F, x_t_tmin1[:, ii])
                              + torch.matmul(K_t, e_t[:, ii]))
        P_t_tmin1 = torch.matmul(FP, torch.transpose(L_t, 0, 1)) + Q

        # Innovation form of the log likelihood
        log_det_Dt = logdet_torch(D_t)  # logdet of prediction error covariance also converges
        logL[ii-1] = -(qlog2pi + log_det_Dt + torch.matmul(
            torch.matmul(torch.unsqueeze(e_t[:, ii], 0), invD_t),
            torch.unsqueeze(e_t[:, ii], -1))) / 2

    # Recursion of forward pass for the remaining time steps
    for ii in range(conv_steps+1, T+1):  # t=conv_steps+1 -> t=T
        e_t[:, ii] = y[:, ii-1] - torch.matmul(G, x_t_tmin1[:, ii])
        x_t_tmin1[:, ii+1] = (torch.matmul(F, x_t_tmin1[:, ii])
                              + torch.matmul(K_t, e_t[:, ii]))
        # noinspection PyUnboundLocalVariable
        logL[ii-1] = -(qlog2pi + log_det_Dt + torch.matmul(
            torch.matmul(torch.unsqueeze(e_t[:, ii], 0), invD_t),
            torch.unsqueeze(e_t[:, ii], -1))) / 2

    x_t_tmin1 = x_t_tmin1[:, :-1]  # Remove the extra t=T+1 time point created

    # Kalman smoothing (backward pass)
    r_t = torch.zeros(p, dtype=torch.float32).cuda()
    x_t_n = torch.zeros(p, T+1, dtype=torch.float32).cuda()  # (index 1 corresponds to t=0, etc.)
    Ip = torch.eye(p, dtype=torch.float32).cuda()

    # Run R_t recursion until convergence
    GD = torch.matmul(torch.transpose(G, 0, 1), invD_t)
    R_t = torch.zeros(p, p, dtype=torch.float32).cuda()  # dlyap is slow on GPU, find convergent R_t empirically
    GDG = torch.matmul(GD, G)

    for k in range(0, conv_steps):  # t=T -> t=T-conv_steps+1
        R_t = GDG + torch.matmul(torch.matmul(torch.transpose(L_t, 0, 1), R_t), L_t)

    RP = torch.matmul(R_t, P_t_tmin1)
    P_t_n = torch.matmul(P_t_tmin1, (Ip - RP))
    P_t_tmin1_n = torch.matmul(
        torch.matmul((Ip - torch.transpose(RP, 0, 1)), L_t),
        torch.transpose(P_t_tmin1, 0, 1))  # derived using Theorem 1 and Lemma 1, m = t, s = t+1

    # Recursion of backward pass: fixed-interval smoothing
    for ii in range(T, 0, -1):  # t=T -> t=1
        r_t = (torch.matmul(GD, e_t[:, ii])
               + torch.matmul(torch.transpose(L_t, 0, 1), r_t))
        x_t_n[:, ii] = x_t_tmin1[:, ii] + torch.matmul(P_t_tmin1, r_t)

    x_t_n[:, 0] = x_t_n[:, 1]  # Repeat t=1 to extend smoothed estimates to t=0

    return x_t_n, P_t_n, P_t_tmin1_n, logL, K_t, x_t_tmin1, P_t_tmin1
