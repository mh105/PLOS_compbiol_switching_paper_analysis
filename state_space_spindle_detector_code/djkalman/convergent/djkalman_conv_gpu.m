%% (De Jong version) Kalman filtering and smoothing - Convergent Solutions
function [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1] = djkalman_conv_gpu(F, Q, mu0, Q0, G, R, y, conv_steps)
% This is the [De Jong 1989] Kalman filter and fixed-interval smoother.
% Multivariate observation data y is supported. This implementation runs
% the kalman filtering and smoothing on GPU. This function has been heavily
% optimized for matrix computations in exchange for more intense memory use
% and arcane syntax. Note the filtered estimates are skipped since not used
% during recursion. If need filtered estimate, refer to the equations
% derived in the human-readable version and add to the for loop.
%
% Convergent solutions are used where kalman filtering is run for some time
% steps until P_t_tmin1 converges, and we switch to steady state solutions. 
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
% Authors: Alex He, Proloy Das; Last edit: 12/21/2021

if nargin < 8
    conv_steps = 25; % run kalman filtering for 25 time steps 
end

% Vector dimensions
[q, T] = size(y);
p = length(mu0);

% Convert all inputs to GPU arrays
F = gpuArray(F);
Q = gpuArray(Q);
mu0 = gpuArray(mu0);
Q0 = gpuArray(Q0);
G = gpuArray(G);
R = gpuArray(R);
y = gpuArray(y);

% Kalman filtering (forward pass)
% -------------------------------------------------------------
x_t_tmin1 = gpuArray.zeros(p,T+2); % (index 1 corresponds to t=0, etc.)
K_t = gpuArray.zeros(p,q);
e_t = gpuArray.zeros(q,T+1);
D_t = gpuArray.zeros(q,q);
invD_t = gpuArray.zeros(q,q);
FP = gpuArray.zeros(p,p);
L_t = gpuArray.zeros(p,p);
logL = gpuArray.zeros(1,T); % (index 1 corresponds to t=1)
Iq = gpuArray.eye(q);
qlog2pi = q*log(2*pi);

% initialize
x_t_t = mu0; % x_0_0, initialized only to keep outputs consistent
P_t_t = Q0; % P_0_0, initialized only to keep outputs consistent
x_t_tmin1(:,2) = F*x_t_t; % x_1_0 (W_0*beta in De Jong 1989)
P_t_tmin1 = F*P_t_t*F' + Q; % P_1_0 (V_0 in De Jong 1989)

% recursion of forward pass until convergence of predicted covariance matrix
for ii=2:conv_steps+1 % t=1 -> t=conv_steps
    % intermediate vectors
    e_t(:,ii) = y(:,ii-1) - G*x_t_tmin1(:,ii); % same as dy in classical kalman
    D_t = G*P_t_tmin1*G' + R; % same as Sigma in classical kalman
    invD_t = Iq/D_t;
    FP = F*P_t_tmin1;
    K_t = FP*G'*invD_t;
    L_t = F - K_t*G;
    
    % one-step prediction for the next time point
    x_t_tmin1(:,ii+1) = F*x_t_tmin1(:,ii) + K_t*e_t(:,ii);
    P_t_tmin1 = FP*L_t' + Q;
    
    % innovation form of the log likelihood
    logL(ii-1) = -1/2 * (qlog2pi + log(det(D_t)) + e_t(:,ii)'*invD_t*e_t(:,ii)); 
end

% recursion of forward pass for the remaining time steps 
for ii=conv_steps+2:T+1 % t=conv_steps+1 -> t=T
    e_t(:,ii) = y(:,ii-1) - G*x_t_tmin1(:,ii); % same as dy in classical kalman
    x_t_tmin1(:,ii+1) = F*x_t_tmin1(:,ii) + K_t*e_t(:,ii);
    logL(ii-1) = -1/2 * (qlog2pi + log(det(D_t)) + e_t(:,ii)'*invD_t*e_t(:,ii)); % innovation form of the log likelihood
end

x_t_tmin1(:,end) = []; % remove the extra t=T+1 time point created
% -------------------------------------------------------------

% Kalman smoothing (backward pass) - De Jong derivation avoids
% inverting the conditional state noise covariance matrix!
% -------------------------------------------------------------
r_t = gpuArray.zeros(p,1);
x_t_n = gpuArray.zeros(p,T+1); % (index 1 corresponds to t=0, etc.)
Ip = gpuArray.eye(p);

% Solve a Lyapunov equation for steady state variables during smoothing
GD = G'*invD_t;
R_t = gpuArray(dlyap(gather(L_t'),gather(GD*G))); % convert back to CPU for solving Lyapunov
RP = R_t*P_t_tmin1;
P_t_n = P_t_tmin1*(Ip - RP);
P_t_tmin1_n = (Ip - RP')*L_t*P_t_tmin1'; % derived using Theorem 1 and Lemma 1, m = t, s = t+1

% recursion of backward pass - fixed-interval smoothing
for ii=T+1:-1:2 % t=T -> t=1
    r_t = GD*e_t(:,ii) + L_t'*r_t;
    x_t_n(:,ii) = x_t_tmin1(:,ii) + P_t_tmin1*r_t;
end

x_t_n(:,1) = x_t_n(:,2); % repeat t=1 to extend smoothed estimates to t=0
% -------------------------------------------------------------
end
