%% Switching state-space models for sleep spindle detection and analysis
% Version: 07/29/2022 Alex He - switched to using the new ssm class 
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

%% Load example full-night data
load('young_sleep_EEG_data.mat') % channel C3

% MT parameters
spectrogram_parameters.frequency_max = Fs/2;
spectrogram_parameters.taper_params = [2, 3];
spectrogram_parameters.window_params = [1, 0.05];
spectrogram_parameters.min_NFFT = 2^10;
spectrogram_parameters.detrend = 'constant';
spectrogram_parameters.weighting = 'unity';
spectrogram_parameters.ploton = false;

%% MTM spectrogram visualization of full-night recording
while false
    % compute spectrogram
    [ spect, stimes, sfreqs ] = multitaper_spectrogram_mex(single(EEG'), Fs, ...
        [0 min([Fs/2 spectrogram_parameters.frequency_max])], ...
        spectrogram_parameters.taper_params, spectrogram_parameters.window_params, ...
        spectrogram_parameters.min_NFFT, spectrogram_parameters.detrend, ...
        spectrogram_parameters.weighting, spectrogram_parameters.ploton);
    
    plot_freqs = sfreqs<=30; % adjust if want to check up to Nyquist
    
    % plot spectrograms
    figure
    ax = figdesign(4,1, 'merge', {[2,3]});
    colormap jet;
    linkaxes(ax,'x');
    axes(ax(1))
    hypnoplot(t, stages);
    title('Hypnogram After Artifact Detection')
    set(gca, 'FontSize', 16)
    axes(ax(2))
    imagesc(stimes, sfreqs(plot_freqs), nanpow2db(spect(plot_freqs,:)));
    axis xy;
    climscale;
    peak_freq_range = [8, 17];
    hline(peak_freq_range(1));
    hline(peak_freq_range(2));
    title('Channel C3 Spectrogram')
    ylabel('Frequency (Hz)')
    set(gca, 'FontSize', 16)
    axes(ax(3))
    plot(t, EEG)
    xlabel('Time (s)')
    set(gca, 'FontSize', 16)
end

%% Extract a 10s segment of N2 data with spindles
segment1 = [19573, 20173];

% let's take a 10s data
segment1 = [6996, 7006];
[~, idx] = min(abs(t - segment1(1)));
eeg1 = EEG(idx:idx+Fs*diff(segment1));
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
endID=Fs*diff(segment1)+1;

% fit with both slow and spindle oscillations
init_params.f_init=[2, 13]; % initial center frequency (Hz)
init_params.a_init=0.98*ones(1,length(init_params.f_init)); % initial radius
init_params.sigma2_init=ones(1,length(init_params.f_init)); % initial variance
init_params.R=initialize_R(eeg1, Fs, 30, false); %initial measurement variance
init_params.NNharm=[1, 1]; %harmonics for each oscillation
fname='segment2_N2'; % a filename attached to the oscillator model structure

noise_osc=[]; %This will fix the frequencies of 4th 5th and 6th oscillation (47, 60, 78 Hz). Refers to the order in f_init. (Leave empty to avoid using)

vmp_param.kappa=[10000, 10000]; %concentration parameter for Von Mises Prior
mu_f=init_params.f_init; %center frequency of Von Mises Prior in Hz
vmp_param.mu = (mu_f*2*pi)/Fs;
vmp_param.osc_num = [1, 2]; %oscillation with prior (leave empty to avoid using von mises prior)

em_it=400; %number of EM iterations
plot_on=0; %Do we want to plot the outcome?
auto_on=0; %Turn off automatic parameters

% Parameters for iterative model
prior_on="no_prior"; %prior on state noise covariance
updateR=true; %update observation noise in EM algorithm?

% Run the pre-specified oscillator model fitting (not iterative model)
modelOsc = ssp_decomp(eeg1,Fs,startID(1),endID(1),em_it,eps,init_params,vmp_param,noise_osc,fname,plot_on,auto_on,prior_on,updateR,true);

%This function does not save the output. You need to save separately.
time_ind=1; %this is the window number
% [f_y, S_ar]=plot_summary_component(modelOsc,time_ind,[1]);

% Extract spindle component 
x_t_n_tot=modelOsc.res{1,time_ind}.x_t_n(:,2:end);
spindle_comp = x_t_n_tot(3,:); % real component of second oscillator

%% How does the component compare against FIR filtered signal
while false
    x_t_n_tot=modelOsc.res{1,time_ind}.x_t_n(:,2:end);
    spindle_comp = x_t_n_tot(3,:); % real component of second oscillator
    
    [filtered_eeg, d] = bandpass(eeg1, [10,16], Fs); % FIR filtered signal
    
    figure;
    ax1=subplot(2,1,1);
    plot(t1, spindle_comp)
    title('Spindle Component')
    set(gca,'FontSize',14)
    ax2=subplot(2,1,2);
    plot(t1, filtered_eeg)
    title('FIR filtered 10-16Hz')
    xlabel('Time (s)')
    set(gca,'FontSize',14)
end

%% Switching between two alternative models
% using the EM fitted parameters, we can construct two alternative models,
% which are essentially two different observation matrices with one
% observing just the slow, and the other observing both slow and alpha.

% we need to re-run the Kalman filtering with switching dynamics to
% estimate the filtered model probabilities for the two alterantive models

o1 = ssm([],modelOsc.res{1,1}.model_prams.Phi,modelOsc.res{1,1}.model_prams.Q,...
    zeros(size(modelOsc.res{1,1}.model_prams.mu)),modelOsc.res{1,1}.model_prams.sigma,...
    {[1,0,1,0],[1,0,0,0]}, modelOsc.res{1,1}.model_prams.R, eeg1', Fs);

%% Traditional switching methods 
[Mprob, fy_t] = switching(o1,'method','IMM','dwellp',0.99);

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
title(['10s segment, Manual Set Q = ', num2str(o1.Q(3,3))])
set(gca, 'FontSize', 14)
set(lab, 'FontSize', 14)

axes(ax(2))
plot(t1, eeg1)
hold on
plot(xlim, [0,0], 'k--')
title('Original signal')
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 14)

axes(ax(3))
plot(t1, spindle_comp, '-')
title('Spindle Component')
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 14)

axes(ax(4))
stairs(t1, Mprob(1,:), 'LineWidth',2)
hold on
plot(xlim, [0.5,0.5], 'k--')
% scatter(t1(Mprob(1,2:end)>0.05), ones(size(t1(Mprob(1,2:end)>0.05)))*0.5, 'm')
title('Filtered Model Probability for Slow + Spindle')
set(gca, 'FontSize', 14)
ylim([0,1])

axes(ax(5))
stairs(t1, Mprob(2,:), 'LineWidth',2)
hold on
plot(xlim, [0.5,0.5], 'k--')
title('Filtered Model Probability for Slow only')
set(gca, 'FontSize', 14)
ylim([0,1])

axes(ax(6))
stairs(t1, fy_t(1,:), 'LineWidth',1)
title('Conditional density of y for Slow + Spindle')
set(gca, 'FontSize', 14)

axes(ax(7))
stairs(t1, fy_t(2,:), 'LineWidth',1)
title('Conditional density of y for Slow only')
set(gca, 'FontSize', 14)

%% Full VB learning (Ghahramani & Hinton (2000))
[Mprob, fy_t, obj_array, A, VB_iter, Mprob_soft, Mprob_hard, logL_bound] = VBlearn(o1, 'shared_ctype', {[1,1]}, 'norm_qt_m', false);

[Mprob, fy_t, obj_array, A, VB_iter, logL_bound] = VBlearn_original(o1, 'plot_Estep', false);

[Mprob, fy_t] = switching(obj_array,'method','ab pari','A',A); % soft segmentation

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
title(['10s segment'])
set(gca, 'FontSize', 14)
set(lab, 'FontSize', 14)

axes(ax(2))
plot(t1, eeg1)
hold on
plot(xlim, [0,0], 'k--')
title('Original signal')
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 14)

axes(ax(3))
plot(t1, spindle_comp, '-')
title('Spindle Component')
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 14)

axes(ax(4))
stairs(t1, Mprob(1,:), 'LineWidth',2)
hold on
plot(xlim, [0.5,0.5], 'k--')
% scatter(t1(Mprob(1,2:end)>0.05), ones(size(t1(Mprob(1,2:end)>0.05)))*0.5, 'm')
title('Filtered Model Probability for Slow + Spindle')
set(gca, 'FontSize', 14)
ylim([0,1])

axes(ax(5))
stairs(t1, Mprob(2,:), 'LineWidth',2)
hold on
plot(xlim, [0.5,0.5], 'k--')
title('Filtered Model Probability for Slow only')
set(gca, 'FontSize', 14)
ylim([0,1])

axes(ax(6))
stairs(t1, fy_t(1,:), 'LineWidth',1)
title('Conditional density of y for Slow + Spindle')
set(gca, 'FontSize', 14)

axes(ax(7))
stairs(t1, fy_t(2,:), 'LineWidth',1)
title('Conditional density of y for Slow only')
set(gca, 'FontSize', 14)

%% plotting to show different segmentation methods 
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
title(['VB learning (improved)'])
set(gca, 'FontSize', 14)
set(lab, 'FontSize', 14)

axes(ax(2))
plot(t1, eeg1)
hold on
plot(xlim, [0,0], 'k--')
title('Original signal')
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 14)

axes(ax(3))
plot(t1, spindle_comp, '-')
title('Spindle Component')
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 14)

axes(ax(4))
l1=stairs(t1, Mprob(1,:), 'LineWidth',2);
hold on
l2=plot(t1, Mprob_soft(1,:), 'LineWidth',2);
l3=plot(t1, Mprob_hard(1,:), '--', 'LineWidth',2);
plot(xlim, [0.5,0.5], 'k--')
title('Segmentation results for Slow + Spindle')
legend([l1,l2,l3], 'minKL', 'soft', 'viterbi')
set(gca, 'FontSize', 14)
ylim([0,1])

axes(ax(5))
l1=stairs(t1, Mprob(2,:), 'LineWidth',2);
hold on
l2=plot(t1, Mprob_soft(2,:), 'LineWidth',2);
l3=plot(t1, Mprob_hard(2,:), '--', 'LineWidth',2);
plot(xlim, [0.5,0.5], 'k--')
title('Segmentation results for Slow only')
legend([l1,l2,l3], 'minKL', 'soft', 'viterbi')
set(gca, 'FontSize', 14)
ylim([0,1])

axes(ax(6))
stairs(t1, fy_t(1,:), 'LineWidth',1)
title('Converged qt_m for Slow + Spindle')
set(gca, 'FontSize', 14)

axes(ax(7))
stairs(t1, fy_t(2,:), 'LineWidth',1)
title('Converged qt_m for Slow only')
set(gca, 'FontSize', 14)

%% PLOS paper figure: short example figure
figure
ax = figdesign(9,1, 'merge', {[1,2]}, 'margin', [.05 .1 .1 .05 .05]);
set(gcf, 'Position', [1 0.0810 0.3000 0.7382])
for ii = 1:length(ax);  axes(ax(ii)); title(num2str(ii)); end
linkaxes(ax, 'x')

% Spectrogram
axes(ax(1))
imagesc(stimes1, sfreqs1(sfreqs1<=30), pow2db(spect1(sfreqs1<=30,:)));
colormap jet;
axis xy;
climscale;
[lab,c] = topcolorbar(.1, 0.01, 0.01);
set(lab, 'Position', [c.Limits(1)-diff(c.Limits)/2.5,0,0])
ylabel('Frequency (Hz)')
title(['10s segment'])
set(gca, 'FontSize', 14)
set(lab, 'FontSize', 14)

% Original time series 
axes(ax(2))
plot(t1, eeg1)
hold on
plot(xlim, [0,0], 'k--')
title('Original signal')
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 14)

% Filtered signal 
[filtered_eeg, d] = bandpass(eeg1, [10,16], Fs); % FIR filtered signal
axes(ax(3))
plot(t1, filtered_eeg, '-')
title('FIR filtered 10-16 Hz')
ylabel('Voltage (\muV)')
set(gca, 'FontSize', 14)

% Static switching 
[Mprob, fy_t] = switching(o1,'method','static','dwellp',0.99);
axes(ax(4))
stairs(t1, Mprob(1,:), 'k', 'LineWidth',2)
hold on
plot(xlim, [0.5,0.5], 'k--')
% scatter(t1(Mprob(1,2:end)>0.05), ones(size(t1(Mprob(1,2:end)>0.05)))*0.5, 'm')
title('Static switching')
set(gca, 'FontSize', 14)
ylim([0,1])

% IMM 
[Mprob, fy_t] = switching(o1,'method','IMM','dwellp',0.99);
axes(ax(5))
stairs(t1, Mprob(1,:), 'k', 'LineWidth',2)
hold on
plot(xlim, [0.5,0.5], 'k--')
title('IMM')
set(gca, 'FontSize', 14)
ylim([0,1])

% S&S 1991
[Mprob, fy_t] = switching(o1,'method','1991','dwellp',0.99);
axes(ax(6))
stairs(t1, Mprob(1,:), 'k', 'LineWidth',2)
hold on
plot(xlim, [0.5,0.5], 'k--')
title('S&S 1991')
set(gca, 'FontSize', 14)
ylim([0,1])

% Annealed variational learning 
[Mprob, fy_t, obj_array, A, VB_iter, logL_bound] = VBlearn_original(o1, 'plot_Estep', false);
axes(ax(7))
stairs(t1, Mprob(1,:), 'Color', [0.8500 0.3250 0.0980], 'LineWidth',2)
hold on
plot(xlim, [0.5,0.5], 'k--')
title('Annealed')
set(gca, 'FontSize', 14)
ylim([0,1])

% Interpolated density variational learning 
[Mprob, fy_t, obj_array, A, VB_iter, Mprob_soft, Mprob_hard, logL_bound] = VBlearn(o1, 'shared_ctype', {[1,1]});
axes(ax(8))
stairs(t1, Mprob(1,:), 'Color', [0 0.4470 0.7410], 'LineWidth',2)
hold on
plot(xlim, [0.5,0.5], 'k--')
title('Interpolated')
set(gca, 'FontSize', 14)
ylim([0,1])

%%
% EOF