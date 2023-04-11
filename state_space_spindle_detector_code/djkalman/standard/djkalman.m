%% (De Jong version) Kalman filtering and smoothing
function [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1, fy_t_interp] = djkalman(F, Q, mu0, Q0, G, R, y, R_weights)
% This is the [De Jong 1989] Kalman filter and fixed-interval smoother.
% Multivariate observation data y is supported. This implementation runs
% the kalman filtering and smoothing on CPU. This function has been heavily
% optimized for matrix computations in exchange for more intense memory use
% and arcane syntax. Note the filtered estimates are skipped since not used
% during recursion. If you need filtered estimate, refer to the equations
% derived in the human-readable version and add to the for loop.
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

if ~exist('R_weights', 'var') || all(isnan(R_weights), 'all')
    R = R .* ones(q, q, T);
else % (index 1 corresponds to t=1, etc.)
    R = R .* reshape(repmat(R_weights, q^2, 1), q, q, T);
end

% Kalman filtering (forward pass)
% -------------------------------------------------------------
x_t_tmin1 = zeros(p,T+2); % (index 1 corresponds to t=0, etc.)
P_t_tmin1 = zeros(p,p,T+2);
K_t = zeros(p,q,T+1); % note that this is different from the classical Kalman gain by pre-multiplying with F
e_t = zeros(q,T+1);
invD_t = zeros(q,q,T+1);
L_t = zeros(p,p,T+1);
logL = zeros(1,T); % (index 1 corresponds to t=1)
Iq = eye(q);
qlog2pi = q*log(2*pi);

% initialize
x_t_t = mu0; % x_0_0, initialized only to keep outputs consistent
P_t_t = Q0; % P_0_0, initialized only to keep outputs consistent
x_t_tmin1(:,2) = F*x_t_t; % x_1_0 (W_0*beta in De Jong 1989)
P_t_tmin1(:,:,2) = F*P_t_t*F' + Q; % P_1_0 (V_0 in De Jong 1989)

% recursion of forward pass
for ii=2:T+1 % t=1 -> t=T
    % intermediate vectors
    e_t(:,ii) = y(:,ii-1) - G*x_t_tmin1(:,ii); % same as dy in classical kalman
    D_t = G*P_t_tmin1(:,:,ii)*G' + R(:,:,ii-1); % same as Sigma in classical kalman 
    invD_t(:,:,ii) = Iq/D_t;
    FP = F*P_t_tmin1(:,:,ii);
    K_t(:,:,ii) = FP*G'*invD_t(:,:,ii);
    L_t(:,:,ii) = F - K_t(:,:,ii)*G;
    
    % one-step prediction for the next time point
    x_t_tmin1(:,ii+1) = F*x_t_tmin1(:,ii) + K_t(:,:,ii)*e_t(:,ii);
    P_t_tmin1(:,:,ii+1) = FP*L_t(:,:,ii)' + Q;
    
    % innovation form of the log likelihood
    logL(ii-1) = -1/2 * (qlog2pi + sparse_find_log_det_mex(D_t) + e_t(:,ii)'*invD_t(:,:,ii)*e_t(:,ii));
end

x_t_tmin1(:,end) = []; % remove the extra t=T+1 time point created
P_t_tmin1(:,:,end) = []; % remove the extra t=T+1 time point created
% -------------------------------------------------------------

% Kalman smoothing (backward pass) - De Jong derivation avoids
% inverting the conditional state noise covariance matrix!
% -------------------------------------------------------------
r_t = zeros(p,T+1); % (index 1 corresponds to t=0, etc.)
R_t = zeros(p,p,T+1);
x_t_n = zeros(p,T+1);
P_t_n = zeros(p,p,T+1);
P_t_tmin1_n = zeros(p,p,T+1); % cross-covariance between t and t-1
Ip = eye(p);

% recursion of backward pass - fixed-interval smoothing
for ii=T+1:-1:2 % t=T -> t=1
    % intermediate vectors
    GD = G'*invD_t(:,:,ii);
    r_t(:,ii-1) = GD*e_t(:,ii) + L_t(:,:,ii)'*r_t(:,ii);
    R_t(:,:,ii-1) = GD*G + L_t(:,:,ii)'*R_t(:,:,ii)*L_t(:,:,ii);
    
    % smoothed estimates
    x_t_n(:,ii) = x_t_tmin1(:,ii) + P_t_tmin1(:,:,ii)*r_t(:,ii-1);
    RP = R_t(:,:,ii-1)*P_t_tmin1(:,:,ii);
    P_t_n(:,:,ii) = P_t_tmin1(:,:,ii)*(Ip - RP);
    P_t_tmin1_n(:,:,ii) = (Ip - RP')*L_t(:,:,ii-1)*P_t_tmin1(:,:,ii-1)'; % derived using Theorem 1 and Lemma 1, m = t, s = t+1
end

% set the cross-covariance estimate at t=1
P_t_tmin1_n(:,:,2) = P_t_n(:,:,2); % Cov(t=1,t=0) can be computed exactly using J_0. But we use P_t=1_n instead to avoid inverting conditional state noise covariance.

% Repeat t=1 and extend the smoothed estimates to t=0. This is
% because De Jong version of smoothing is not defined at t=0.
% We can compute these exactly using J_0, but we skip to avoid
% inverting the conditional state noise covariance matrix.
x_t_n(:,1) = x_t_n(:,2);
P_t_n(:,:,1) = P_t_n(:,:,2);
% -------------------------------------------------------------

% Optional output - interpolated conditional density of y_t
% fy_t_interp = normpdf(y_t | y_1,...,y_t-1,y_t+1,y_T)
% -------------------------------------------------------------
if nargout > 9
    fy_t_interp = zeros(1,T); % (index 1 corresponds to t=1, etc.)
    for ii=1:T % t=1 -> t=T
        n_t = invD_t(:,:,ii+1)*e_t(:,ii+1) - K_t(:,:,ii+1)'*r_t(:,ii+1);
        N_t = invD_t(:,:,ii+1) + K_t(:,:,ii+1)'*R_t(:,:,ii+1)*K_t(:,:,ii+1);
        fy_t_interp(ii) = exp(-1/2 * (qlog2pi - sparse_find_log_det_mex(N_t) + n_t'/N_t*n_t)); % see De Jong 1989 Section 5
    end
end
% -------------------------------------------------------------
end
