%% Running a single instance of the Iterative Oscillation Model
%by Amanda M Beck 3/22/21

addpath chronux-helpers %add selected helper functions from chronux (small alterations)

%If using Van Der Pol 
clearvars -except y_t

%% Basic Parameters to Change %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%base = '/autofs/cluster';
%path to save data: (base + base_path)
base_path='/autofs/cluster/purdonlab/users/ambeck/AR_model/propofol_matlab/ar_modeling/Matsuda_Hugo/SSP/iterative_osc_model/iter_test3/';

%supported priors:
%{'no_prior','laplace'}, only 'inv_gamma' recommended 

param.data.y=repmat(y_t(1:800,2)',2,1); %This is the "recorded" data (must be real) and should be (number of subjects x number of time points)
priors={'inv_gamma'};
Fs=80; %Sampling Frequency in Hz
window_len=10; %window length in seconds
run_AR1=false; %Do you want to compare AR1 and AR2 slow components? 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%% Parameters for the iterative model (Advanced Parameters to Change)

%param.addpath_tools=false;
%param.mount=false; %Are you mounting the drive? (ex. VPN, mount drive, run code on own computer)
param.useFittedParam=true; %Should we use fitted parameters from iteration i as the initial parameters in iteration i+1
param.updateR=true; %Should we update Observation Noise Variance on each EM iteration? 
param.data.win_no=1; %keep as 1 window for now, may test more windows in future
param.data.win_length=window_len;
param.data.Fs=Fs;
param.save_fpath=base_path;
min_round=3; %Optional: Set the minimum number of oscillations
param.osc_limit=7;%Maximum number of oscillations (stop from getting stuck adding infinite oscillations)
%you want to investigate

%Von Mises Prior parameters:
osc_num=[];
kappa=[];
mu_f=[]; %if these three parameters are empty, function will use usual parameters, override by defining vmp_param.kappa (concentration parameter) and vmp_param.mu (center frequency (Hz))
vmp_off=false; %turn VMP off to see what it's doing (NOT RECOMMENDED FOR DATA ANALYSIS)

for ii=1:length(priors)
    if run_AR1
        eval(['mkdir ' base_path priors{ii} '_AR1']);
        param.fig_fold=[ priors{ii} '_AR1/'];
        param.prior_on=priors{ii};
        modelOsc=iterative_model(param, min_round, kappa, mu_f, osc_num,true); %Run iterative model with AR1 slow oscillation
    end
    eval(['mkdir ' base_path priors{ii} '_AR2']);
    param.fig_fold=[ priors{ii} '_AR2/'];
    param.prior_on=priors{ii};
    modelOsc=iterative_model(param, min_round, kappa, mu_f, osc_num, false); %Run iterative model with AR2 bslow oscillation
end

%% Identifying Minimum AIC Model and Graphing AIC
osc_num_overall = AIC_selection(param.save_fpath, priors, 2, run_AR1);

%% Plot selected model
close all

for ii=1:length(priors)
    for sub=1:size(osc_num_overall,2)
        display(['Subject #' num2str(sub) ' ' priors{ii}]);
        if osc_num_overall(2,sub) ==1
            load([ base_path priors{ii} '_AR' num2str(osc_num_overall(1,sub)) '/sub' num2str(sub) '_orig.mat'])
            display(['Base: AR ' num2str(osc_num_overall(1,sub)) ', Number of Oscillators: 1']);
        else
            load([ base_path priors{ii} '_AR' num2str(osc_num_overall(1,sub)) '/sub' num2str(sub) '_r' num2str(osc_num_overall(2,sub)-1) '.mat'])
            display(['Base: AR ' num2str(osc_num_overall(1,sub)) ', Number of Oscillators: ' num2str(osc_num_overall(2,sub))]);
        end
        figure
        [f_y, S_ar]=plot_summary_component(modelOsc,1,[1]); %The last argument refers to which figure to plot. 1= spectrum, 2=spectrogrgam, 3=time series 
        sgtitle(['Subject #' num2str(sub) ' AR ' num2str(osc_num_overall(1,sub))]);
        
    end
end
%% Local Functions

function graph_all(x_mat,specparams)
    figure
    for ii=1:size(x_mat,1)
        [S, f]=mtspectrumc(x_mat(ii,:),specparams);
        plot(f,20*log10(S)); hold on;
    end
end

