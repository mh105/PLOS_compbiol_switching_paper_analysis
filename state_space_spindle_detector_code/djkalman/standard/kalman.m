%% Kalman filtering and smoothing
function [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1, fy_t_interp] = kalman(F, Q, mu0, Q0, G, R, y, R_weights)
% This is the classical Kalman filter and fixed-interval smoother.
% Multivariate observation data y is supported. This implementation runs
% the kalman filtering and smoothing on CPU. This function has been heavily
% optimized for matrix computations in exchange for more intense memory use
% and arcane syntax. 
%
% Reference:
% Shumway, R. H., & Stoffer, D. S. (1982). An approach to time
% series smoothing and forecasting using the EM algorithm.
% Journal of time series analysis, 3(4), 253-264.
%
% De Jong, P., & Mackinnon, M. J. (1988). Covariances for
% smoothed estimates in state space models. Biometrika, 75(3),
% 601-602.
%
% Jazwinski, A. H. (2007). Stochastic processes and filtering
% theory. Courier Corporation.
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
x_t_tmin1 = zeros(p,T+1); % (index 1 corresponds to t=0, etc.)
P_t_tmin1 = zeros(p,p,T+1);
x_t_t = zeros(p,T+1);
P_t_t = zeros(p,p,T+1);
logL = zeros(1,T); % (index 1 corresponds to t=1)
I = eye(q);
qlog2pi = q*log(2*pi);

% initialize
x_t_t(:,1) = mu0; % x_0_0
P_t_t(:,:,1) = Q0; % P_0_0

% recursion of forward pass
for ii=2:T+1 % t=1 -> t=T
    % one-step prediction
    x_t_tmin1(:,ii) = F*x_t_t(:,ii-1);
    P_t_tmin1(:,:,ii) = F*P_t_t(:,:,ii-1)*F' + Q;
    
    % update
    GP = G*P_t_tmin1(:,:,ii);
    Sigma = GP*G' + R(:,:,ii-1);
    invSigma = I/Sigma;
    dy = y(:,ii-1)-G*x_t_tmin1(:,ii);
    K_t = GP'*invSigma;
    x_t_t(:,ii) = x_t_tmin1(:,ii) + K_t*dy;
    P_t_t(:,:,ii) = P_t_tmin1(:,:,ii) - K_t*GP;
    
    % innovation form of the log likelihood
    logL(ii-1) = -1/2 * (qlog2pi + sparse_find_log_det_mex(Sigma) + dy'*invSigma*dy);
end
% -------------------------------------------------------------

% Kalman smoothing (backward pass)
% -------------------------------------------------------------
x_t_n = zeros(p,T+1); % (index 1 corresponds to t=0, etc.)
P_t_n = zeros(p,p,T+1);
P_t_tmin1_n = zeros(p,p,T+1); % cross-covariance between t and t-1
fy_t_interp = nan; % interpolated conditional density is not available in classical kalman filtering

% initialize
x_t_n(:,end) = x_t_t(:,end); % x_T_T
P_t_n(:,:,end) = P_t_t(:,:,end); % P_T_T

% recursion of backward pass
for ii=T+1:-1:2 % t=T -> t=1
    J_t = P_t_t(:,:,ii-1)*F'/P_t_tmin1(:,:,ii);
    JP = J_t*P_t_n(:,:,ii);
    x_t_n(:,ii-1) = x_t_t(:,ii-1) + J_t*(x_t_n(:,ii) - x_t_tmin1(:,ii));
    P_t_n(:,:,ii-1) = P_t_t(:,:,ii-1) + (JP - J_t*P_t_tmin1(:,:,ii))*J_t';
    P_t_tmin1_n(:,:,ii) = JP'; % Cov(t,t-1) proved in De Jong & Mackinnon (1988)
end
% -------------------------------------------------------------
end
