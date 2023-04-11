%% Experiment 7: Spindle prediction experiment
close all
clear all
clc

% Change the current folder to the folder of this m-file.
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
clearvars tmp

% addpath to the iterative_oscillator folder - which includes ssss as submodule
addpath(genpath('state_space_spindle_detector_code'))
addpath(genpath('iterative_oscillator'))

%% Load data
load('young_sleep_EEG_data.mat') % channel C3

% MT parameters
spectrogram_parameters.frequency_max = Fs/2;
spectrogram_parameters.taper_params = [2, 3];
spectrogram_parameters.window_params = [1, 0.05];
spectrogram_parameters.min_NFFT = 2^10;
spectrogram_parameters.detrend = 'constant';
spectrogram_parameters.weighting = 'unity';
spectrogram_parameters.ploton = false;

%% Select a segment
% segment1 = [14897, 14927]; % 1st
% segment1 = [8311, 8341]; % 2nd
segment1 = [26203, 26233]; % 3rd

[~, idx] = min(abs(t - segment1(1)));
eeg1 = EEG(idx:idx+Fs*diff(segment1))';
t1 = 0:1/Fs:diff(segment1);

% compute the multitaper spectrogram for the segment
[ spect1, stimes1, sfreqs1 ] = multitaper_spectrogram_mex(eeg1, Fs, ...
    [0 min([Fs/2 spectrogram_parameters.frequency_max])], ...
    spectrogram_parameters.taper_params, spectrogram_parameters.window_params, ...
    spectrogram_parameters.min_NFFT, spectrogram_parameters.detrend, ...
    spectrogram_parameters.weighting, spectrogram_parameters.ploton);

%% Inspect this segment first
% generate a summary plot
figure
ax = figdesign(2,3, 'merge', {[1,2], [4,5]}, 'margin', [.1 .1 .1 .05 .1]);
set(gcf, 'Position', [0.5009 0.2611 0.4409 0.5739])
for ii = 1:length(ax);  axes(ax(ii)); title(num2str(ii)); end
for ii = [2,4]; axes(ax(ii)); current_ax = gca; current_ax.Position = [current_ax.Position(1)-0.05, current_ax.Position(2), current_ax.Position(3)+0.05, current_ax.Position(4)]; end
linkaxes(ax(1:3), 'x')

axes(ax(1))
imagesc(stimes1, sfreqs1(sfreqs1<=30), pow2db(spect1(sfreqs1<=30,:)));
colormap jet;
axis xy;
climscale;
[lab,c] = topcolorbar(.1, 0.01, 0.006);
set(lab, 'Position', [c.Limits(1)-diff(c.Limits)/3,0,0])
ylabel('Frequency (Hz)')
title('N2 Sleep Spindles')
set(gca, 'FontSize', 20)
set(lab, 'FontSize', 14)

axes(ax(2))
imagesc(stimes1, sfreqs1, pow2db(spect1));
colormap jet;
axis xy;
climscale;
clb = colorbar;
ylabel(clb, 'Power (dB)')
xlabel('Time (s)')
ylabel('Frequency (Hz)')
title('Frequency up to Nyquist')
set(gca, 'FontSize', 14)

axes(ax(3))
plot(t1, eeg1)
hold on
plot(xlim, [0,0], 'k--')
xlabel('Time (s)')
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 20)

axes(ax(4))
spectrum1 = nanmean(spect1, 2);
plot(sfreqs1, pow2db(spectrum1), 'LineWidth', 1.5)
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
title('Multitaper Spectrum [2,3 | 1,0.05]')
set(gca, 'FontSize', 14)

%% Fit oscillator model with a fixed number of oscillations
% specify the index of data to use
%startID is vector of starting indexes for parallelizing the process, endID
%is the vector of ending indexes (In units of samples)
startID=1;
endID=length(eeg1);

% fit with both slow and spindle oscillations
init_params.f_init=[2, 13]; % initial center frequency (Hz)
init_params.a_init=0.98*ones(1,length(init_params.f_init)); % initial radius
init_params.sigma2_init=ones(1,length(init_params.f_init)); % initial variance
init_params.R=estimate_R(eeg1, Fs, 30, 'cov'); %initial measurement variance
init_params.NNharm=[1, 1]; %harmonics for each oscillation
fname='segment2_N2'; % a filename attached to the oscillator model structure

noise_osc=[]; %This will fix the frequencies of 4th 5th and 6th oscillation (47, 60, 78 Hz). Refers to the order in f_init. (Leave empty to avoid using)

vmp_param.kappa=ones(1,2) * (length(eeg1) * 10); %concentration parameter for Von Mises Prior
mu_f=init_params.f_init; %center frequency of Von Mises Prior in Hz
vmp_param.mu = (mu_f*2*pi)/Fs;
vmp_param.osc_num = [1, 2]; %oscillation with prior (leave empty to avoid using von mises prior)

em_it=50; %number of EM iterations
plot_on=0; %Do we want to plot the outcome?
auto_on=0; %Turn off automatic parameters

% Parameters for iterative model
prior_on="no_prior"; %prior on state noise covariance
updateR=true; %update observation noise in EM algorithm?

% Run the pre-specified oscillator model fitting (not iterative model)
modelOsc = ssp_decomp(eeg1,Fs,startID(1),endID(1),em_it,eps,init_params,vmp_param,noise_osc,fname,plot_on,auto_on,prior_on,updateR,true);

%This function does not save the output. You need to save separately.
time_ind=1; %this is the window number
[f_y, S_ar]=plot_summary_component(modelOsc,time_ind,[1]);

%% Switching inference
init_R = max(modelOsc.res{1,1}.model_prams.R, init_params.R * (modelOsc.res{1,1}.model_prams.Q(3,3) / modelOsc.res{1,1}.model_prams.R));

o1 = ssm([],modelOsc.res{1,1}.model_prams.Phi,modelOsc.res{1,1}.model_prams.Q,...
    zeros(size(modelOsc.res{1,1}.model_prams.mu)),modelOsc.res{1,1}.model_prams.sigma,...
    {[1,0,1,0],[1,0,0,0]}, init_R, eeg1, Fs);

%% Full VB learning first to train the models
[Mprob, fy_t, obj_array, A, VB_iter, Mprob_soft, Mprob_hard,...
    logL_bound, x_t_n_all, P_t_n_all] = VBlearn(o1, 'shared_ctype', {[1,1]}, 'maxVB_iter', 5, 'norm_qt_m', true);

%% Run prediction in real time by providing more samples incrementally
spindle_prediction = zeros(size(t1));
lag = 5; % 5s

for ii = 2:length(eeg1)
    disp(['Time point: ', num2str(ii)])
    new_obj_array = obj_array;
    new_obj_array(1).y = eeg1(max(1, ii-lag*Fs):ii);
    new_obj_array(2).y = eeg1(max(1, ii-lag*Fs):ii);
    
    new_Mprob = VBlearn(new_obj_array, 'A', A, 'maxVB_iter', 1, 'verbose', false, 'norm_qt_m', true);
    
    spindle_prediction(ii) = new_Mprob(1,end);
end

%% plotting the results of VB learning
figure
ax = figdesign(8,1, 'merge', {[1,2]}, 'margin', [.05 .1 .1 .05 .05]);
set(gcf, 'Position', [1 0.0810 0.3000 0.7382])
for ii = 1:length(ax);  axes(ax(ii)); title(num2str(ii)); end
linkaxes(ax, 'x')
axes(ax(1))
imagesc(stimes1, sfreqs1(sfreqs1<=30), pow2db(spect1(sfreqs1<=30,:)));
colormap jet;
axis xy;
climscale;
[lab,c] = topcolorbar(.1, 0.01, 0.01);
set(lab, 'Position', [c.Limits(1)-diff(c.Limits)/2.5,0,0])
ylabel('Frequency (Hz)')
title(['Spectrogram'])
set(gca, 'FontSize', 14)
set(lab, 'FontSize', 14)

axes(ax(2))
plot(t1, eeg1)
hold on
ylim([-70, 70])
plot(xlim, [0,0], 'k--')
title('Original signal')
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 14)

[filtered_eeg, d] = bandpass(eeg1, [10,16], Fs); % FIR filtered signal
axes(ax(3))
plot(t1, filtered_eeg, '-')
ylim([-20, 20])
title('Filtered 10-16 Hz')
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 14)

axes(ax(4))
stairs(t1, Mprob(1,:), 'LineWidth',2)
hold on
plot(xlim, [0.5,0.5], 'k--')
% scatter(t1(Mprob(1,2:end)>0.05), ones(size(t1(Mprob(1,2:end)>0.05)))*0.5, 'm')
title('Model Probability for Slow + Spindle')
set(gca, 'FontSize', 14)
ylim([0,1])

axes(ax(5))
spindle_comp = x_t_n_all{1}(3,2:end);
spindle_comp(Mprob(1,:) < 0.5) = 0;
spindle_std = sqrt(squeeze(P_t_n_all{1}(3,3,2:end)))';
spindle_std(Mprob(1,:) < 0.5) = 0;
shadedErrorBar(t1, spindle_comp, 2*spindle_std, {'Color', [0 0.4470 0.7410]});
ylim([-20, 20])
[a, w] = get_rot_aw(obj_array(1).F(3:4,3:4));
spindle_freq = w * Fs / (2 * pi);
title(['Spindle Component Frequency: ', num2str(spindle_freq), ' Hz'])
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 14)

axes(ax(6))
stairs(t1, spindle_prediction(1,:), 'LineWidth',2)
hold on
plot(xlim, [0.5,0.5], 'k--')
title('Predictive Probability for Slow + Spindle')
set(gca, 'FontSize', 14)
ylim([0,1])

axes(ax(7))
stairs(t1, fy_t(2,:), 'LineWidth',1)
title('q_t^m density of y for Slow only')
set(gca, 'FontSize', 14)
