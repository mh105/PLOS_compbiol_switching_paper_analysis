# Author: Ran Liu <rliu20@mgh.harvard.edu> Last edit: Alex He 12-28-2021
import torch


def logdet_torch(A):
    """Computes logdet using Schur complement"""
    log_det = torch.log(A[0, 0])
    for i in range(0, A.shape[0]-1):
        A = A[1:, 1:] - torch.matmul(torch.unsqueeze(A[1:, 0], -1),
                                     torch.unsqueeze(A[0, 1:], 0))/A[0, 0]
        log_det += torch.log(A[0, 0])
    return log_det


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
