%% Experiment 6 - Oscillator model simulation
% To test the performance of switching with oscillator models on a few
% parameters:
%   - Dimensionality of the SSM
%   - Numbers of switching SSMs

close all; clear all; clc

addpath(genpath('state_space_spindle_detector_code'))

%% Figure to visualize the spectrum of the underlying oscillators 
% Load the simulated time series 
load('Experiment6_simulation_plot_data.mat')
Fs = 100;

figure
hold on
freqs = [1, 10, 20, 30, 40];
l = zeros(size(freqs));
for ii = 1:size(sim_xs,1)
    [spect, stimes, sfreqs] = multitaper_spectrogram_mex(sim_xs(ii,:), Fs, [], [4 6], [4 2], [],'off',[],false,false);
    l(ii) = plot(sfreqs, pow2db(mean(spect, 2)), 'LineWidth', 2);
    plot([freqs(ii), freqs(ii)], ylim, 'k--', 'LineWidth', 1)
end
legend(l, {'1 Hz', '10 Hz', '20 Hz', '30 Hz', '40 Hz'}, 'Location', 'Best')
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
set(gca, 'FontSize', 16)

%% Visualize segmentation performance
% Load segmentation accuracy data
load('Experiment6_simulation_results.mat')

num_oscillators = 2:5;
figure
hold on

shadedErrorBar(num_oscillators, mean(improved_percent_correct), std(improved_percent_correct)./sqrt(size(improved_percent_correct, 1)), {'Color', [0 0.4470 0.7410]});
plot(num_oscillators, mean(improved_percent_correct), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0 0.4470 0.7410])

shadedErrorBar(num_oscillators, mean(original_percent_correct), std(original_percent_correct)./sqrt(size(original_percent_correct, 1)), {'Color', [0.8500 0.3250 0.0980]}, 1);
plot(num_oscillators, mean(original_percent_correct), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.8500 0.3250 0.0980])

plot(num_oscillators, 1./(2.^num_oscillators-1), 'k--', 'LineWidth', 3)

xlim([1.5, 5.5])
xticks(1:5)
ylim([0, 0.9])
yticks(0:0.1:0.9)
xlabel('Number of Oscillators')
ylabel('Percent Correct Segmentation')
set(gca,'FontSize',20)

%% Show an example switching variable with 5 oscillators to show the difficulty 

figure;
subplot(1,2,1)
hold on
plot(s, 'k-', 'LineWidth', 2.5)
xlim([0, length(s)])
ylim([0, 2^5-1])
xlabel('Time Points')
ylabel('Switching variable')
set(gca,'FontSize',20)

[~, improved] = max(Mprob);
[~, original] = max(Mprob_original);

subplot(1,2,2)
hold on
plot(original-1, 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980])
plot(improved-1, 'LineWidth', 2, 'Color', [0 0.4470 0.7410])
xlim([0, length(s)])
ylim([0, 2^5-1])
xlabel('Time Points')
ylabel('Switching variable')
set(gca,'FontSize',20)
