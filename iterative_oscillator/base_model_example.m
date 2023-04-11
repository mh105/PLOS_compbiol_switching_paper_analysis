%% Running a single instance of the Base Oscillation Model
%by Amanda M Beck 3/22/21

addpath chronux-helpers %add selected helper functions from chronux (small alterations)

%% Loading some example data (replace with your data)
load('/autofs/cluster/purdonlab/users/ambeck/AR_model/propofol_matlab/ar_modeling/Elderly/prop_extract_data.mat');
save_fpath='/autofs/cluster/purdonlab/users/ambeck/AR_model/propofol_matlab/ar_modeling/Matsuda_Hugo/data/aging/modelselection/test3'
%run /autofs/eris/purdongp/projects/Elderly/Ageogram/prop_age_and_conc.m
[lead time_pts patient_no]=size(prop_data);

saved_data=squeeze(prop_data(3,:,:));
saved_data=resample(saved_data,8,25);
%% Set Up the Model Parameters

Fs_new=80; %sampling frequency (Hz)
win_sec=10; %window length (sec)
data_no=1:12; %number of windows

startID=(data_no-1)*win_sec*Fs_new+1;%startID is vector of starting indexes for parallelizing the process,
endID=data_no*win_sec*Fs_new;% endID is the vector of ending indexes

init_params.f_init=[0 12]; % initial center frequency (Hz)
init_params.a_init=0.98*ones(1,length(init_params.f_init)); % initial radius
init_params.sigma2_init=ones(1,length(init_params.f_init)); % initial variance

init_params.NNharm=[1 1]; %harmonics for each oscillation
fname='noise_test';

noise_osc=[1]; %This will fix the frequencies of the 1st oscillation (0 Hz). Refers to the order in f_init. (Leave empty to avoid using), 
%Fix frequency with 0 Hz initial to set up an AR1

vmp_param.kappa=[3000 3000]; %concentration parameter for Von Mises Prior, same order as mu
mu_f=[0 12]; %center frequency of Von Mises Prior in Hz, kappa and mu correspond
vmp_param.mu = (mu_f*2*pi)/Fs_new;
vmp_param.osc_num = [1 2]; %oscillation with prior (leave empty to avoid using von mises prior)


em_it=400; %number of iterations
plot_on=0; %Do we want to plot the outcome?
auto_on=0; %Turn off automatic parameters

% Parameters for iterative model
prior_on="inv_gamma"; %prior on state noise covariance
updateR=true; %update observation noise in EM algorithm?

%% Run the Model and Examine the results
for sub=6%1:patient_no
    init_params.R=initialize_R(saved_data(:,sub),Fs_new,30,false); %initial measurement variance, selected from 100 Hz and higher
    
    [modelOsc]=ssp_decomp(saved_data(:,sub),Fs_new,startID(1),endID(1),em_it, eps,init_params,vmp_param,noise_osc,fname,plot_on,auto_on, prior_on, updateR);

    %View the results
    %This function does not save the output. You need to save separately.
    time_ind=1; %this is the window number
    [f_y, S_ar]=plot_summary_component(modelOsc,time_ind,[1 3]); %The last argument refers to which figure to plot. 1= spectrum, 2=spectrogrgam, 3=time series 

end
