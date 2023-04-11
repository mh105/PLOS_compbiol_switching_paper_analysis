% Convergent kalman filtering and smoothing example script
% Last edit: 12/07/2021 Alex He 

% addpath to the compiled sparse_find_log_det mex files (Linux and Mac OS
% are supported)
addpath('./standard')

% load data
load('./runtime_testing/runtime_inputs.mat')

% cpu version 
tic
[x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1] = djkalman_conv(F, Q, mu0, Q0, G, R, y);
toc

% Save results
save('./runtime_testing/runtime_results.mat', 'x_t_n', 'P_t_n', 'P_t_tmin1_n', 'logL', 'x_t_t', 'P_t_t', 'K_t', 'x_t_tmin1', 'P_t_tmin1')


% gpu version
tic
[x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1] = djkalman_conv_gpu(F, Q, mu0, Q0, G, R, y);
toc

clear
load('./runtime_testing/runtime_results.mat')
K_t_matlab = K_t;
logL_matlab = logL;
P_t_n_matlab = P_t_n;
P_t_tmin1_matlab = P_t_tmin1;
P_t_tmin1_n_matlab = P_t_tmin1_n;
x_t_n_matlab = x_t_n;
x_t_tmin1_matlab = x_t_tmin1;

load('./runtime_testing/torch_results.mat')
figure
plot(1:200, logL)