"""
Define a special class for Experiment 4 in order to do
coupled parameter estimation in a bi-variate AR(1) setting
used for Granger Causality-like VB switching learning

This should not be used in general, only for the method
paper simulation analysis
"""

from codetiming import Timer
from sorcery import dict_of
import numpy as np
from somata.switching import VBSwitchModel, switching
from somata.exact_inference import viterbi, run_em


class BVAR1VBS(VBSwitchModel):

    def __init__(self, ssm_array, y=None):
        super().__init__(ssm_array, y=y)

    def learn(self, y=None, dwell_prob=0.99, A=None, keep_param=(),
              maxE_iter=100, maxVB_iter=100, h_t_thresh=1e-6, q_t_thresh=1e-6,
              shared_R=False, shared_comp=False, priors_all='auto',
              plot_E=False, verbose=True, return_dict=None, original=False, show_pbar=True):
        old_settings = np.seterr(divide='ignore', over='ignore', invalid='ignore')
        t = Timer(name="time taken", text="{name}: {seconds:.2f} seconds")
        t.start()

        """ Initialization """
        self.A = abs(np.eye(self.K, dtype=np.float64) - np.ones((self.K, self.K), dtype=np.float64)
                     * (1-dwell_prob) / (self.K-1)) if A is None else A
        self.logL_bound = []
        if original:
            shared_R = True
            shared_comp = False
            priors_all = None
            self.h_t_m = np.ones((self.K, self.T), dtype=np.float64) / self.K  # initialize with equal responsibilities
        else:
            self.h_t_m = float('inf')
        # Initialize priors for SSM if 'auto' is specified
        priors_all = [x.initialize_priors() for x in self.ssm_array] if priors_all == 'auto' else priors_all

        e_kwargs = {'original': original, 'maxE_iter': maxE_iter, 'q_t_thresh': q_t_thresh, 'plot_E': plot_E}
        m_kwargs = {'priors_all': priors_all, 'keep_param': keep_param,
                    'shared_R': shared_R, 'shared_comp': shared_comp, 'original': original}

        """ EM iterations """
        VB_iter, _ = run_em(self, y=y, e_kwargs=e_kwargs, m_kwargs=m_kwargs,
                            max_iter=maxVB_iter, stop_thresh=h_t_thresh, show_pbar=show_pbar)

        """ Switching linear segmentation """
        # Provide different segmentation methods (soft, semi-hard (VB h_t_m), hard)
        h_t_m_soft, fy_t = switching(self.ssm_array, method='ab pari', A=self.A)  # soft segmentation
        _, h_t_m_hard = viterbi(self.A, fy_t)  # apply Viterbi on parallel interpolated density for hard segmentation

        """ EOF """
        if verbose:
            print('BVAR1VBS.learn()')
            t.stop()
            print('iterations:', str(VB_iter))

        np.seterr(**old_settings)

        if return_dict is None:
            pass
        elif return_dict:
            return dict_of(self.h_t_m, h_t_m_soft, h_t_m_hard,
                           self.q_t_m, self.A, self.ssm_array, VB_iter, self.logL_bound)
        else:
            return self.h_t_m, h_t_m_soft, h_t_m_hard, \
                   self.q_t_m, self.A, self.ssm_array, VB_iter, self.logL_bound

    def m_step(self, y=None, x_t_n_all=None, P_t_n_all=None, P_t_tmin1_n_all=None, h_t_tmin1_m=None,
               priors_all=None, keep_param=(), shared_R=False, shared_comp=False, original=False):
        """ Generalized M step for VB learning, exposed for run_em() """
        # M.1 ML estimates of SSM parameters with weighted data
        _m1(self.ssm_array, x_t_n_all, P_t_n_all, P_t_tmin1_n_all, self.h_t_m, y=y,
            priors_all=priors_all, keep_param=keep_param, shared_R=shared_R, shared_comp=shared_comp,
            original=original)

        # M.2 ML estimate of transition matrix of HMM
        self.A, _ = _m2(self.h_t_m, h_t_tmin1_m)


def _m1(ssm_array, x_t_n_all, P_t_n_all, P_t_tmin1_n_all, h_t_m, y=None, priors_all=None,
        keep_param=(), shared_R=False, shared_comp=False, original=False):
    """ Special M step definitions for Bi-variate AR1 process """
    y = ssm_array[0].y if y is None else y
    K = len(ssm_array)  # number of models
    T = y.shape[1]  # number of time points
    priors_all = [[None] * x.ncomp for x in ssm_array] if priors_all is None else priors_all

    """ Get sums of squares for each model """
    R_ss = np.zeros((y.shape[0], y.shape[0], K), dtype=np.float64)
    A, B, C = ([], [], [])
    for m in range(K):
        m_results = ssm_array[m].m_estimate(y=y, x_t_n=x_t_n_all[m], P_t_n=P_t_n_all[m], P_t_tmin1_n=P_t_tmin1_n_all[m],
                                            h_t=h_t_m[m, :], priors=priors_all[m], force_ABC=True,
                                            keep_param=('F', 'Q', 'mu0', 'Q0', 'R', 'G'), return_dict=True)
        # Store sum variables for joint estimations
        R_ss[:, :, m] = m_results['R_ss']
        A.append(m_results['A'])
        B.append(m_results['B'])
        C.append(m_results['C'])

    """ Apply correct update equations, assume ssm_array[0] has F[0,1] == 0 """
    if original is None:  # separate estimation for each model << not using this for now as original can also couple
        """ Update the transition matrix F """
        theta_1 = B[0][0, 0] / A[0][0, 0]
        # phi = B[0][1, :] @ np.linalg.inv(A[0])
        # ssm_array[0].F = np.array([[theta_1, 0.], phi])  # assume ssm_array[0] has F[0,1] == 0
        phi = B[0][1, 1] / A[0][1, 1]
        ssm_array[0].F = np.array([[theta_1, 0.], [0., phi]])

        # ssm_array[1].F = B[1] @ np.linalg.inv(A[1])
        theta = B[1][0, :] @ np.linalg.inv(A[1])
        phi = B[1][1, 1] / A[1][1, 1]
        ssm_array[0].F = np.array([theta, [0., phi]])

        """ Update state noise covariance matrix Q """
        Q_ss_0 = C[0] - B[0] @ ssm_array[0].F.T - ssm_array[0].F @ B[0].T + ssm_array[0].F @ A[0] @ ssm_array[0].F.T
        sigma2_0_0 = Q_ss_0[0, 0] / T
        sigma2_0_1 = Q_ss_0[1, 1] / T
        ssm_array[0].Q = np.array([[sigma2_0_0, 0.], [0, sigma2_0_1]])

        Q_ss_1 = C[1] - B[1] @ ssm_array[1].F.T - ssm_array[1].F @ B[1].T + ssm_array[1].F @ A[1] @ ssm_array[1].F.T
        sigma2_1_0 = Q_ss_1[0, 0] / T
        sigma2_1_1 = Q_ss_1[1, 1] / T
        ssm_array[1].Q = np.array([[sigma2_1_0, 0.], [0, sigma2_1_1]])

        """ Update observation noise R """
        lambda2_0_0 = R_ss[0, 0, 0] / h_t_m[0, :].sum()
        lambda2_0_1 = R_ss[1, 1, 0] / h_t_m[0, :].sum()
        ssm_array[0].R = np.array([[lambda2_0_0, 0.], [0, lambda2_0_1]])

        lambda2_1_0 = R_ss[0, 0, 1] / h_t_m[1, :].sum()
        lambda2_1_1 = R_ss[1, 1, 1] / h_t_m[1, :].sum()
        ssm_array[1].R = np.array([[lambda2_1_0, 0.], [0, lambda2_1_1]])

    else:  # do coupled estimation
        """ Update the transition matrix F """
        A_temp = A[1].copy()
        A_temp[0, 0] += A[0][0, 0]
        B_temp = B[1][0, :].copy()
        B_temp[0] += B[0][0, 0]
        theta = B_temp @ np.linalg.inv(A_temp)

        """
        A_temp = A[0] + A[1]
        B_temp = B[0][1, :] + B[1][1, :]
        phi = B_temp @ np.linalg.inv(A_temp)

        ssm_array[0].F = np.array([[theta[0], 0.], phi])  # assume ssm_array[0] has F[0,1] == 0
        ssm_array[1].F = np.array([theta, phi])
        """

        phi = (B[0][1, 1] + B[1][1, 1]) / (A[0][1, 1] + A[1][1, 1])

        ssm_array[0].F = np.array([[theta[0], 0.], [0., phi]])  # assume ssm_array[0] has F[0,1] == 0
        ssm_array[1].F = np.array([theta, [0., phi]])

        """ Update state noise covariance matrix Q """
        Q_ss_0 = C[0] - B[0] @ ssm_array[0].F.T - ssm_array[0].F @ B[0].T + ssm_array[0].F @ A[0] @ ssm_array[0].F.T
        Q_ss_1 = C[1] - B[1] @ ssm_array[1].F.T - ssm_array[1].F @ B[1].T + ssm_array[1].F @ A[1] @ ssm_array[1].F.T
        if Q_ss_0[0, 0] < 0 or Q_ss_1[0, 0] < 0:  # catch negative covariance estimates
            import ipdb
            ipdb.set_trace()
        sigma2_0 = (Q_ss_0[0, 0] + Q_ss_1[0, 0]) / 2 / T
        sigma2_1 = (Q_ss_0[1, 1] + Q_ss_1[1, 1]) / 2 / T
        new_Q = np.array([[sigma2_0, 0.], [0, sigma2_1]])
        ssm_array[0].Q = new_Q.copy()
        ssm_array[1].Q = new_Q.copy()

        """ Update observation noise R """
        R_ss = R_ss.sum(axis=2)
        lambda2_0 = R_ss[0, 0] / T
        lambda2_1 = R_ss[1, 1] / T
        new_R = np.array([[lambda2_0, 0.], [0, lambda2_1]])
        ssm_array[0].R = new_R.copy()
        ssm_array[1].R = new_R.copy()

    # mu0 and Q0 are not coupled regardless of using the original algorithm or not
    # Update initial state mean mu0
    ssm_array[0].mu0 = x_t_n_all[0][:, 0][:, None]
    ssm_array[1].mu0 = x_t_n_all[1][:, 0][:, None]

    # Update initial state covariance Q0
    Q0_0 = P_t_n_all[0][:, :, 0] + x_t_n_all[0][:, 0][:, None] @ x_t_n_all[0][:, 0][:, None].T \
        - x_t_n_all[0][:, 0][:, None] @ ssm_array[0].mu0.T \
        - ssm_array[0].mu0 @ x_t_n_all[0][:, 0][:, None].T + ssm_array[0].mu0 @ ssm_array[0].mu0.T
    ssm_array[0].Q0 = np.array([[Q0_0[0, 0], 0.], [0., Q0_0[1, 1]]])

    Q0_1 = P_t_n_all[1][:, :, 0] + x_t_n_all[1][:, 0][:, None] @ x_t_n_all[1][:, 0][:, None].T \
        - x_t_n_all[1][:, 0][:, None] @ ssm_array[1].mu0.T \
        - ssm_array[1].mu0 @ x_t_n_all[1][:, 0][:, None].T + ssm_array[1].mu0 @ ssm_array[1].mu0.T
    ssm_array[1].Q0 = np.array([[Q0_1[0, 0], 0.], [0., Q0_1[1, 1]]])


def _m2(h_t_m, h_t_tmin1_m):
    """ ML estimation of HMM parameters for the M.2 step """
    # M step - MLE of parameters {A, p1}
    p1 = h_t_m[:, 0]  # updated p1
    edges = h_t_tmin1_m.sum(axis=2)  # sum over time points
    A = edges / edges.sum(axis=0)  # updated A
    return A, p1
