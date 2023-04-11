%% (De Jong version) Kalman filtering and smoothing - Human Readable
function [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1, fy_t_interp] = djkalman_human(F, Q, mu0, Q0, G, R, y, R_weights)
% This is the [De Jong 1989] Kalman filter and fixed-interval smoother.
% Multivariate observation data y is supported. This implementation runs
% the kalman filtering and smoothing on CPU. This is a human-readable
% version of the code. It is not optimized for computation but good for
% understanding the algorithm.
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
% Author: Alex He; Last edit: 12/25/2021

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
D_t = zeros(q,q,T+1);
L_t = zeros(p,p,T+1);
x_t_t = zeros(p,T+1);
P_t_t = zeros(p,p,T+1);
logL = zeros(1,T); % (index 1 corresponds to t=1)

% initialize
x_t_t(:,1) = mu0; % x_0_0
P_t_t(:,:,1) = Q0; % P_0_0
x_t_tmin1(:,2) = F*x_t_t(:,1); % x_1_0 (W_0*beta in De Jong 1989)
P_t_tmin1(:,:,2) = F*P_t_t(:,:,1)*F' + Q; % P_1_0 (V_0 in De Jong 1989)

% recursion of forward pass
for ii=2:T+1 % t=1 -> t=T
    % intermediate vectors
    e_t(:,ii) = y(:,ii-1) - G*x_t_tmin1(:,ii);
    D_t(:,:,ii) = G*P_t_tmin1(:,:,ii)*G' + R(:,:,ii-1);
    K_t(:,:,ii) = F*P_t_tmin1(:,:,ii)*G'/D_t(:,:,ii);
    L_t(:,:,ii) = F - K_t(:,:,ii)*G;
    
    % one-step prediction for the next time point
    x_t_tmin1(:,ii+1) = F*x_t_tmin1(:,ii) + K_t(:,:,ii)*e_t(:,ii);
    P_t_tmin1(:,:,ii+1) = F*P_t_tmin1(:,:,ii)*L_t(:,:,ii)' + Q;
    
    % filtered estimate using Theorem 1 and m = t-1, n = t = s
    x_t_t(:,ii) = x_t_tmin1(:,ii) + P_t_tmin1(:,:,ii)*G'/D_t(:,:,ii)*e_t(:,ii);
    P_t_t(:,:,ii) = P_t_tmin1(:,:,ii) - P_t_tmin1(:,:,ii)*G'/D_t(:,:,ii)*G*P_t_tmin1(:,:,ii);
    
    % innovation form of the log likelihood
    logL(ii-1) = - 1/2 * (q*log(2*pi) + log(det(G*P_t_tmin1(:,:,ii)*G' + R(:,:,ii-1))) +...
        (y(:,ii-1)-G*x_t_tmin1(:,ii))'/(G*P_t_tmin1(:,:,ii)*G' + R(:,:,ii-1))*(y(:,ii-1)-G*x_t_tmin1(:,ii)));
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

% recursion of backward pass - fixed-interval smoothing
for ii=T+1:-1:2 % t=T -> t=1
    % intermediate vectors
    r_t(:,ii-1) = G'/D_t(:,:,ii)*e_t(:,ii) + L_t(:,:,ii)'*r_t(:,ii);
    R_t(:,:,ii-1) = G'/D_t(:,:,ii)*G + L_t(:,:,ii)'*R_t(:,:,ii)*L_t(:,:,ii);
    
    % smoothed estimates
    x_t_n(:,ii) = x_t_tmin1(:,ii) + P_t_tmin1(:,:,ii)*r_t(:,ii-1);
    P_t_n(:,:,ii) = P_t_tmin1(:,:,ii) - P_t_tmin1(:,:,ii)*R_t(:,:,ii-1)*P_t_tmin1(:,:,ii);
    P_t_tmin1_n(:,:,ii) = L_t(:,:,ii-1)*P_t_tmin1(:,:,ii-1)' - P_t_tmin1(:,:,ii)'*R_t(:,:,ii-1)'*L_t(:,:,ii-1)*P_t_tmin1(:,:,ii-1)'; % derived using Theorem 1 and Lemma 1, m = t, s = t+1
end

% set the cross-covariance estimate at t=1
P_t_tmin1_n(:,:,2) = P_t_n(:,:,2); % Cov(t=1,t=0) can be computed exactly using J_0. But we use P_t=1_n instead to avoid inverting conditional state noise covariance.
% -------------------------------------------------------------

% Repeat t=1 and extend the smoothed estimates to t=0. This is
% because De Jong version of smoothing is not defined at t=0.
% We can compute these exactly using J_0, but we skip to avoid
% inverting the conditional state noise covariance matrix.
x_t_n(:,1) = x_t_n(:,2);
P_t_n(:,:,1) = P_t_n(:,:,2);

% Optional output - interpolated conditional density of y_t
% fy_t_interp = normpdf(y_t | y_1,...,y_t-1,y_t+1,y_T)
% -------------------------------------------------------------
if nargout > 9
    n_t = zeros(q,T); % (index 1 corresponds to t=1, etc.)
    N_t = zeros(q,q,T);
    fy_t_interp = zeros(1,T);
    for ii=1:T % t=1 -> t=T
        n_t(:,ii) = inv(D_t(:,:,ii+1))*e_t(:,ii+1) - K_t(:,:,ii+1)'*r_t(:,ii+1); %#ok<*MINV>
        N_t(:,:,ii) = inv(D_t(:,:,ii+1)) + K_t(:,:,ii+1)'*R_t(:,:,ii+1)*K_t(:,:,ii+1);
        fy_t_interp(ii) = exp(-q/2*log(2*pi) - 1/2*(log(1/det(N_t(:,:,ii))) + n_t(:,ii)'/N_t(:,:,ii)*n_t(:,ii))); % see De Jong 1989 Section 5
    end
end
% -------------------------------------------------------------
end
