%% Kalman filtering and smoothing - Human Readable
function [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t,x_t_tmin1,P_t_tmin1, fy_t_interp] = kalman_human(F, Q, mu0, Q0, G, R, y, R_weights)
% This is the classical Kalman filter and fixed-interval smoother.
% Multivariate observation data y is supported. This implementation runs
% the kalman filtering and smoothing on CPU. This is a human-readable
% version of the code. It is not optimized for computation but good for
% understanding the algorithm.
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
% Author: Alex He; Last edit: 12/28/2021

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
K_t = zeros(p,q,T+1);
x_t_t = zeros(p,T+1);
P_t_t = zeros(p,p,T+1);
logL = zeros(1,T); % (index 1 corresponds to t=1)

% initialize
x_t_t(:,1) = mu0; % x_0_0
P_t_t(:,:,1) = Q0; % P_0_0

% recursion of forward pass
for ii=2:T+1 % t=1 -> t=T
    % one-step prediction
    x_t_tmin1(:,ii) = F*x_t_t(:,ii-1);
    P_t_tmin1(:,:,ii) = F*P_t_t(:,:,ii-1)*F' + Q;
    
    % update
    K_t(:,:,ii) = P_t_tmin1(:,:,ii)*G'/(G*P_t_tmin1(:,:,ii)*G' + R(:,:,ii-1));
    x_t_t(:,ii) = x_t_tmin1(:,ii) + K_t(:,:,ii)*(y(:,ii-1) - G*x_t_tmin1(:,ii));
    P_t_t(:,:,ii) = P_t_tmin1(:,:,ii) - K_t(:,:,ii)*G*P_t_tmin1(:,:,ii);
    
    % innovation form of the log likelihood
    logL(ii-1) = - 1/2 * (q*log(2*pi) + log(det(G*P_t_tmin1(:,:,ii)*G' + R(:,:,ii-1))) +...
        (y(:,ii-1)-G*x_t_tmin1(:,ii))'/(G*P_t_tmin1(:,:,ii)*G' + R(:,:,ii-1))*(y(:,ii-1)-G*x_t_tmin1(:,ii)));
end
% -------------------------------------------------------------

% Kalman smoothing (backward pass)
% -------------------------------------------------------------
J_t = zeros(p,p,T+1); % (index 1 corresponds to t=0, etc.)
x_t_n = zeros(p,T+1);
P_t_n = zeros(p,p,T+1);
P_t_tmin1_n = zeros(p,p,T+1); % cross-covariance between t and t-1
fy_t_interp = nan; % interpolated conditional density is not available in classical kalman filtering

% initialize
x_t_n(:,end) = x_t_t(:,end); % x_T_T
P_t_n(:,:,end) = P_t_t(:,:,end); % P_T_T

% recursion of backward pass
for ii=T+1:-1:2 % t=T -> t=1
    J_t(:,:,ii-1) = P_t_t(:,:,ii-1)*F'/(P_t_tmin1(:,:,ii));
    x_t_n(:,ii-1) = x_t_t(:,ii-1) + J_t(:,:,ii-1)*(x_t_n(:,ii) - F*x_t_t(:,ii-1));
    P_t_n(:,:,ii-1) = P_t_t(:,:,ii-1) + J_t(:,:,ii-1)*(P_t_n(:,:,ii) - P_t_tmin1(:,:,ii))*J_t(:,:,ii-1)';
    P_t_tmin1_n(:,:,ii) = P_t_n(:,:,ii)*J_t(:,:,ii-1)'; % Cov(t,t-1) proved in De Jong & Mackinnon (1988)
end
% -------------------------------------------------------------
end
