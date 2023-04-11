%% Runtime testing script
% This script runs kalman filtering and smoothing on an example data taken
% from Gladia's source localization simulation using ico3 cortical
% resolution. These runtime results provide the benchmark to understand the
% relative speed performance of different Kalman filtering and smoothing
% implementations in realistic source localization applications.
%
% Author: Alex He; Last edit: 12/07/2021

% Change the current folder to the folder of this m-file.
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
clearvars tmp

% addpath to code folders
addpath('../standard')
addpath('../convergent')
addpath('../archived')

close all; clear all; clc

%% Load example data
load('runtime_inputs.mat')
% y - 64 channel data for 200 time points
% F - transition matrix
% Q - state noise covariance matrix
% mu0 - initial state mean
% Q0 - initial state covariance matrix
% G - observation matrix
% R - observation noise covariance

%% Elvira's standard kalman filtering and smoothing [~30min]
prior.Kalman = 'standard';
k_version = prior.Kalman;
test_folder = 'runtime_testing';
mkdir(test_folder)
tmp_save_folder = [test_folder, '/tmp'];

tic
[SS_pp, Xf, Xf0, Xp] = sparse_kalmanfilt_20210420(R, Q0, F, G, mu0, length(mu0), prior, Q, size(y,2), k_version, y, [], tmp_save_folder);
toc

tic
[A, logL, Pst, Xs, Xs0, A1, A2, A3, B, B1, B2] = sparse_smoother_20210318(R, G, F, length(mu0), prior, SS_pp, size(y,2), k_version, Xf, Xf0, Xp, y, tmp_save_folder);
toc

% delete the folder created during kalman filtering
rmdir(test_folder, 's');

%% Elvira's Steady state kalman filtering and smoothing [~3min but doesn't work due to singular matrix]
prior.Kalman = 'SS';
k_version = prior.Kalman;

tic
[SS_pp, Xf, Xf0, Xp] = sparse_kalmanfilt_20210420(R, Q0, F, G, mu0, length(mu0), prior, Q, size(y,2), k_version, y, [], []);
toc

tic
[A, logL, Pst, Xs, Xs0, A1, A2, A3, B, B1, B2] = sparse_smoother_20210318(R, G, F, length(mu0), prior, SS_pp, size(y,2), k_version, Xf, Xf0, Xp, y, []);
toc

%% Gladia's standard-SS method for kalman filtering and smoothing [~25s, works well]
prior.Kalman = 'standard-SS';
k_version = prior.Kalman;

tic
[SS_pp, Xf, Xf0, Xp] = sparse_kalmanfilt_20210420(R, Q0, F, G, mu0, length(mu0), prior, Q, size(y,2), k_version, y, [], []);
toc

tic
[A, logL, Pst, Xs, Xs0, A1, A2, A3, B, B1, B2] = sparse_smoother_20210318(R, G, F, length(mu0), prior, SS_pp, size(y,2), k_version, Xf, Xf0, Xp, y, []);
toc

%% Classical kalman filtering and smoothing [too slow and takes too much RAM]
% classical kalman filtering on CPU without optimization
tic
[x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t,x_t_tmin1,P_t_tmin1, fy_t_interp] = kalman_human(F, Q, mu0, Q0, G, R, y, nan);
toc

% classical kalman filtering on CPU with optimization
tic
[x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1, fy_t_interp] = kalman(F, Q, mu0, Q0, G, R, y, nan);
toc

%% De Jong kalman filtering and smoothing [too slow and takes too much RAM]
% de jong kalman filtering on CPU without optimization
tic
[x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1, fy_t_interp] = djkalman_human(F, Q, mu0, Q0, G, R, y, nan);
toc

% de jong kalman filtering on CPU with optimization
tic
[x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1, fy_t_interp] = djkalman(F, Q, mu0, Q0, G, R, y, nan);
toc

%% Steady-state De Jong kalman filtering and smoothing [too slow to solve Riccati ~2hrs]
tic
[x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1] = djkalman_ss(F, Q, mu0, Q0, G, R, y);
toc

%% Convergent De Jong kalman filtering and smoothing [~20s, works great!]
tic
[x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1] = djkalman_conv(F, Q, mu0, Q0, G, R, y);
toc

% gpu version
tic
[x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1] = djkalman_conv_gpu(F, Q, mu0, Q0, G, R, y);
toc

% compiled version
tic
[x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1] = djkalman_conv_mex(F, Q, mu0, Q0, G, R, y, 25);
toc



