%% (De Jong version) Kalman filtering and smoothing - Steady State Solutions
function [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1, fy_t_interp] = djkalman_ss(F, Q, mu0, Q0, G, R, y)
% This is the [De Jong 1989] Kalman filter and fixed-interval smoother.
% Multivariate observation data y is supported. This implementation runs
% the kalman filtering and smoothing on CPU. This function has been heavily
% optimized for matrix computations in exchange for more intense memory use
% and arcane syntax. Note the filtered estimates are skipped since not used
% during recursion. If need filtered estimate, refer to the equations
% derived in the human-readable version and add to the for loop.
%
% Steady state solutions are used. Interpolated density approximations are
% also computed since we have closed form steady-state solutions. 
%
% Since De Jong Kalman filtering and smoothing are not defined at t=0, we
% repeat the estimates at t=1 to extend to t=0. This is useful to allow
% updating initial state and covariance estimates in the M step.
%
% Reference:
% De Jong, P. (1989). Smoothing and interpolation with the
% state-space model. Journal of the American Statistical
% Association, 84(408), 1085-1088.
%
% Authors: Alex He, Proloy Das; Last edit: 12/28/2021

% Vector dimensions
[q, T] = size(y);
p = length(mu0);

% Kalman filtering (forward pass)
% -------------------------------------------------------------
x_t_tmin1 = zeros(p,T+2); % (index 1 corresponds to t=0, etc.)
e_t = zeros(q,T+1);
logL = zeros(1,T); % (index 1 corresponds to t=1)
Iq = eye(q);
qlog2pi = q*log(2*pi);

% initialize
x_t_t = mu0; % x_0_0, initialized only to keep outputs consistent
P_t_t = Q0; % P_0_0, initialized only to keep outputs consistent
x_t_tmin1(:,2) = F*x_t_t; % x_1_0 (W_0*beta in De Jong 1989)

% Solve a Riccati equation for steady state variables during filtering
P_t_tmin1 = idare(F',G',Q,R); % this step is incredibly slow!
D_t = G*P_t_tmin1*G' + R;
invD_t = Iq/D_t;
K_t= F*P_t_tmin1*G'*invD_t;
L_t = F - K_t*G;

% recursion of forward pass
for ii=2:T+1 % t=1 -> t=T
    e_t(:,ii) = y(:,ii-1) - G*x_t_tmin1(:,ii); % same as dy in classical kalman
    x_t_tmin1(:,ii+1) = F*x_t_tmin1(:,ii) + K_t*e_t(:,ii);
    logL(ii-1) = -1/2 * (qlog2pi + sparse_find_log_det_mex(D_t) + e_t(:,ii)'*invD_t*e_t(:,ii)); % innovation form of the log likelihood
end

x_t_tmin1(:,end) = []; % remove the extra t=T+1 time point created
% -------------------------------------------------------------

% Kalman smoothing (backward pass) - De Jong derivation avoids
% inverting the conditional state noise covariance matrix!
% -------------------------------------------------------------
r_t = zeros(p,T+1); % (index 1 corresponds to t=0, etc.)
x_t_n = zeros(p,T+1);
Ip = eye(p);

% Solve a Lyapunov equation for steady state variables during smoothing
GD = G'*invD_t;
R_t = dlyap(L_t',GD*G);
RP = R_t*P_t_tmin1;
P_t_n = P_t_tmin1*(Ip - RP);
P_t_tmin1_n = (Ip - RP')*L_t*P_t_tmin1'; % derived using Theorem 1 and Lemma 1, m = t, s = t+1

% recursion of backward pass - fixed-interval smoothing
for ii=T+1:-1:2 % t=T -> t=1
    r_t(:,ii-1) = GD*e_t(:,ii) + L_t'*r_t(:,ii);
    x_t_n(:,ii) = x_t_tmin1(:,ii) + P_t_tmin1*r_t(:,ii-1);
end

x_t_n(:,1) = x_t_n(:,2); % repeat t=1 to extend smoothed estimates to t=0
% -------------------------------------------------------------

% Optional output - interpolated conditional density of y_t
% fy_t_interp = normpdf(y_t | y_1,...,y_t-1,y_t+1,y_T)
% -------------------------------------------------------------
if nargout > 9
    fy_t_interp = zeros(1,T); % (index 1 corresponds to t=1, etc.)
    for ii=1:T % t=1 -> t=T
        n_t = invD_t*e_t(:,ii+1) - K_t'*r_t(:,ii+1);
        N_t = invD_t + K_t'*R_t*K_t;
        fy_t_interp(ii) = exp(-1/2 * (qlog2pi - sparse_find_log_det_mex(N_t) + n_t'/N_t*n_t)); % see De Jong 1989 Section 5
    end
end
% -------------------------------------------------------------
end
