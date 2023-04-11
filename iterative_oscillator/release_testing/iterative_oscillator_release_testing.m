%% Iterative Oscillator Algorithm release testing script (alpha oscillation)
% This is a script made to test each release of the iterative oscillator
% algorithm on 59 resting state recordings from sleep study and 35 subjects
% from anesthesia study. After running, a folder named with the test date
% will be created, containing a few subfolders: 
%   - a folder named "iteration_results" will be saved with the default
%   outputs from the iterative_model() function for each subject 
%   - a folder named "decomposed_spectra" will be saved with figures of
%   recording multitaper spectrogram and decomposed spectra using the
%   fitted oscillators. Manually inspect these plots to check the
%   performance of the algorithm.
% 
% Current testing data sets solely focus on alpha oscillations, which is
% one of the simplest possible neural oscillation in EEG. We need to make
% sure the algorithm can extract alpha oscillations accurately. 
%
% Last edit: Alex He 11/17/2021
close all; clear all; clc

% Change the current folder to the folder of this m-file.
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
clearvars tmp

% addpath to the iterative oscillator algorithm code folders 
addpath('../')
addpath(genpath('../alex_helper_functions'))
addpath(genpath('../chronux-helpers'))
addpath(genpath('../ssss'))

%% Load data sets
load('release_testing_datasets.mat')
% Data set descriptions:
%
% 1) 31 subjects from Sleep HD-EEG study. 5 young adults and 26 elderly
% subjects. 28 of these subjects have two nights of resting state
% recordings with fresh cap application in between. Electrode Z10
% (occipital lead) is extracted under approximate Laplacian referencing.
% The data have been downsampled to 100 Hz and filtered 0.1-50 Hz. Within
% each cell structure, the first entry is 5 min of eyes-open recording, and
% the second entry is 5 min of eyes-closed reocrding. For this release
% testing, a variable number of durations will be examined only from the
% eyes-closed recording. 
%
% 2) 35 subjects from Rodriogo's anesthesia patient recordings within OR. A
% single electrode is taken from a frontal lead with Sedline layout, under
% average-mastoid referencing. The data have been downsampled to 80 Hz and
% filtered 0.1-40 Hz (unsure about this). Each recording is 3-min long.  

%% Algorithm testing parameter setting 
% Specify the recording durations that will be tested
test_duration_list = [10, 30, 60]; % seconds

% State noise covariance prior
priors = {'no_prior'};

%% Begin testing 
% create a folder to save testing results 
startdatetime = datestr(datetime);
save_fd = fullfile(pwd, ['release_testing_results ', strrep(startdatetime,':','_')]);
if ~exist(save_fd, 'dir') % make the directory if non-existant
    mkdir(save_fd) 
    mkdir(fullfile(save_fd, 'iteration_results'))
    mkdir(fullfile(save_fd, 'decomposed_spectra'))
else
    error('release testing folder already exists. Cannot save results!')
end

tic
for test_duration = test_duration_list
    %% Data set 1: 31 subjects with 2 nights of resting state recordings 
    for ii = 1:size(ds1,2)
        for n = 1:2
            disp(['   Processing [ds1] subject # ', num2str(ii), ' night ', num2str(n)])
            
            % load the resting state recording data 
            y_cell = ds1(ii).(['night',num2str(n),'_signal']);
                        
            if ~isempty(y_cell) % some subjects don't have night 2 data 
                % grab the eyes-closed recording
                y = y_cell{2};
                
                % truncate to the desired duration
                y = y(1:test_duration*ds1_Fs);
                
                base_path = fullfile(save_fd, 'iteration_results', ['duration_', num2str(test_duration),'s_ds1'],...
                    ['Subject_', num2str(ii), '_night_', num2str(n), '_']);
                
                fig_name = fullfile(save_fd, 'decomposed_spectra', ...
                    ['duration_', num2str(test_duration),'s_ds1_',...
                    'Subject_', num2str(ii), '_night_', num2str(n) '_decomposed_spectra.png']);
                
                run_iterative_testing(ii, y, ds1_Fs, priors, base_path, fig_name);
                
            end
        end
    end
    disp(['test duration ', num2str(test_duration), 's, ds1, time elapsed since onset = ', num2str(toc), 's'])
    
    %% Data set 2: 35 subjects with 3 min anesthesia recordings
    for ii = 1:size(ds2,1)
        disp(['   Processing [ds2] subject # ', num2str(ii)])
        
        % load the anesthesia recording data
        y = ds2(ii,:);
        
        if ~all(y == 0) % some subjects don't have data
            % truncate to the desired duration
            y = y(1:test_duration*ds2_Fs);
            
            base_path = fullfile(save_fd, 'iteration_results', ['duration_', num2str(test_duration),'s_ds2'],...
                ['Subject_', num2str(ii), '_']);
            
            fig_name = fullfile(save_fd, 'decomposed_spectra', ...
                ['duration_', num2str(test_duration),'s_ds2_',...
                'Subject_', num2str(ii), '_decomposed_spectra.png']);
            
            run_iterative_testing(ii, y, ds2_Fs, priors, base_path, fig_name);
            
        end
    end
    disp(['test duration ', num2str(test_duration), 's, ds2, time elapsed since onset = ', num2str(toc), 's'])
    
end
disp('[Completed] release testing finished for all test durations.')
toc

%% testing helper function  
function [] = run_iterative_testing(ii, y, Fs, priors, base_path, fig_name)
%% Pre-specified parameters 
% Multitaper spectrogram parameters
mtm_params = {[3, 4], [3, 0.01], 2^10, 'constant', 'unity', false, false};

% Spectrum plot cutoff frequency
freq_cutoff = 30; % adjust if want to check up to Nyquist

%% MTM analysis
% plot spectrogram and average mean spectrum
[spect, stimes, sfreqs] = multitaper_spectrogram_mex(y, Fs, [0, Fs/2], mtm_params{:});

f = figure;
ax = figdesign(1,2, 'margins',[.1 .15 .1 .1 .1]);
if contains(matlabroot, 'Applications')
    set(gcf, 'Position', [0.3541 0.3271 0.3713 0.3421])
else
    set(gcf, 'Position', [0.1490 0.2343 0.7443 0.5444])
end

axes(ax(1)) % MTM spectrogram
imagesc(stimes, sfreqs(sfreqs<=freq_cutoff), pow2db(spect(sfreqs<=freq_cutoff,:)));
colormap jet;
axis xy;
climscale;
c = colorbar;
ylabel('Frequency (Hz)')
xlabel('Time (s)')
set(gca, 'FontSize', 20)
title(['Subject ', num2str(ii)], 'FontSize', 30)

axes(ax(2))
hold on % MTM spectrum
h = plot(sfreqs, pow2db(mean(spect,2)), 'k-', 'LineWidth', 2);
legend_elements = [h];
plot([8,8],ylim, 'k--', 'LineWidth', 1)
plot([12,12],ylim, 'k--', 'LineWidth', 1)
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
set(gca, 'FontSize', 20)
xticks([0,8,10,12,20,30])
xlim([0,freq_cutoff])

%% Run Amanda's iterative oscillator model decomposition of the data
% Parameters for the iterative model (Advanced Parameters to Change)
param.data.y = y; %This is the "recorded" data (must be real) and should be (number of subjects x number of time points)
param.useFittedParam=true; %Should we use fitted parameters from iteration i as the initial parameters in iteration i+1
param.updateR=true; %Should we update Observation Noise Variance on each EM iteration?
param.data.win_no=1; %keep as 1 window for now, may test more windows in future
param.data.win_length=length(y)/Fs; %window length in seconds
param.data.Fs=Fs;
param.save_fpath=base_path;
min_round=1; %Optional: Set the minimum number of oscillations you want to investigate
param.osc_limit=5;%Maximum number of oscillations (stop from getting stuck adding infinite oscillations)
param.prior_on=priors{1};
param.fig_fold=[priors{1} '_AR2/'];

%% >>> Iterative Oscillator Model Algorithm <<<
iterative_model(param, min_round, [], [], [], false);

%% Identifying Minimum AIC model and Plot decomposed spectra
sub = 1; % this is always 1 in this testing analysis 
osc_num_overall = AIC_selection(param.save_fpath, priors, sub, false);

% load the saved modelOsc struct with selected model order
if osc_num_overall(2,sub)>1
    load([base_path priors{1} '_AR' num2str(osc_num_overall(1,sub)) '/sub' num2str(sub) '_r' num2str(osc_num_overall(2,sub)-1) '.mat'])
else
    load([base_path priors{1} '_AR' num2str(osc_num_overall(1,sub)) '/sub' num2str(sub) '_orig.mat'])
end

% grab the smoothed estimates of hidden states and transition matrix F
x = modelOsc.res{1, 1}.x_t_n(:,2:end);
assert(size(x,2) == size(param.data.y,2), 'Different dimensions of data and hidden states.')
F = modelOsc.res{1, 1}.model_prams.Phi;

legend_text = {'recorded EEG'};
axes(ax(2))
hold on
set(gca,'ColorOrderIndex',1)

for j = 1:size(x,1)/2 % iterate through all identified oscillators
    [spect, stimes, sfreqs] = multitaper_spectrogram_mex(x(j*2-1,:), Fs, [0, Fs/2], mtm_params{:});
    
    h = plot(sfreqs, pow2db(mean(spect,2)), 'LineWidth', 2);
    
    % calculate the center frequency to put on legends
    [a,w] = get_rotmat_pam(F(j*2-1:j*2,j*2-1:j*2));
    center_freq = abs(w)/(2*pi)*Fs; % keep the reported frequency positive
    legend_text = [legend_text, [sprintf('%.2f', center_freq),'Hz']];
    legend_elements = [legend_elements, h];
end

legend(legend_elements, legend_text, 'Location', 'northeast')

% save the figure in a folder for later browsing
saveas(f, fig_name);
close(f)

end
