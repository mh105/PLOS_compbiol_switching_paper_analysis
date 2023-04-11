%% Script used to compile djkalman_conv
load('runtime_inputs.mat')
[x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1] = djkalman_conv_compile(F, Q, mu0, Q0, G, R, y, 25);