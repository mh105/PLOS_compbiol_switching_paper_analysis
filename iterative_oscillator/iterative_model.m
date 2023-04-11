function [modelOsc]= iterative_model(param, min_round,kappa,mu_f,osc_num,AR1)
% param should have these fields:
% param.save_fpath: string
% param.fig_fold: string
% param.addpath_tools: boolean
% param.mount: boolean
% param.updateR: boolean
% param.useFittedParam: boolean
% param.data.y: matrix (subject x data)
% param.data.Fs: scalar (Hz)
% param.data.win_length: scalar (sec)
% param.data.win_no: vector of window numbers (non-overlapping)
% param.data.min_round: min number of osc tested (scalar) (do not have if
% no min)
% param.vmp_prior.kappa
% param.vmp_prior.mu_f
% param.vmp_prior.noise_osc
% param.prior_on: string
% AR1: boolean, should we start with AR1?

% if param.mount
%     base_dir='/Volumes';
% else
%     base_dir='/autofs/cluster';
% end

save_fpath=[param.save_fpath];
fig_fold=param.fig_fold;
subject_fd_path = [save_fpath fig_fold]; % create subject folder if non-existent
if ~exist(subject_fd_path, 'dir')
    mkdir(subject_fd_path);
end

%addpath([base_dir '/purdonlab/users/ambeck/AR_model/propofol_matlab/ar_modeling/Matsuda_Hugo/SSP/ssp_decomp/'])

% if param.addpath_tools
%     addpath(genpath([base_dir '/purdonlab/users/ambeck/AR_model/tools']))
% end

if isempty(min_round)
    min_round=1;
end

Fs=param.data.Fs;
data_no=param.data.win_no;

startID= round((data_no-1)*Fs*param.data.win_length)+1;
endID= round(data_no*Fs*param.data.win_length);

%% need to do this for f_init
if isempty(kappa)
    vmp_param.kappa=[3000]; %concentration parameter for Von Mises Prior
else
    vmp_param.kappa=kappa;
end

if isempty(mu_f)
    if AR1
        vmp_param.mu=0;
    else
        vmp_param.mu = ([1]*2*pi)/Fs; %center frequency of Von Mises Prior in Hz
    end
else
    vmp_param.mu = (mu_f*2*pi)/Fs;
end

if isempty(osc_num)
    vmp_param.osc_num = [1]; %oscillation with prior (leave empty to avoid using von mises prior)
end

%%
em_it=400; %number of iterations
plot_on=0; %Do we want to plot the outcome?
auto_on=0; %Turn off automatic parameters
fname='test';

% specparams.Fs=Fs;
% specparams.tapers =[10 19];
prior_on=param.prior_on;

ar_order=7; %Order of AR fit to innovations

for sub=1:size(param.data.y,1)
    
    AIC_all=[];
    no_params=[];
    if AR1
        init_params.f_init=[0]; % initial center frequency (Hz)
        noise_osc=[1];
    else
        init_params.f_init=[1];
        noise_osc=[];
    end
    
    init_params.a_init=(0.98^3)*ones(1,length(init_params.f_init)); % initial radius
    init_params.sigma2_init=ones(1,length(init_params.f_init)); % initial variance
    init_params.R=initialize_R(param.data.y(sub,:)',Fs,30,false); %initial measurement variance
    disp(['Initial R = ', num2str(init_params.R)])
    init_params.NNharm=[1]; %harmonics for each oscillation
    %     close all
    
    [modelOsc]=ssp_decomp(param.data.y(sub,:)',Fs,startID,endID,em_it, eps,init_params,vmp_param,noise_osc,fname,plot_on,auto_on,prior_on, param.updateR);
    save([subject_fd_path 'sub' num2str(sub) '_orig.mat'],'modelOsc', 'vmp_param', 'init_params', 'noise_osc'); %save first model
    [AIC_temp, pn_temp]=AIC_calc(modelOsc,1); %calculate AIC
    AIC_all=[AIC_all AIC_temp]; %keep AIC for laterer
    no_params=[no_params pn_temp]; %track number of parameters (for AIC)
    
    [y_pred,y_smooth,y_filt,y]=find_innovations(modelOsc,1); %find y estimates with kalman filter
    y_tot=length(y);
    innov_pred=y-y_pred(1:y_tot)'; %Actual Innovations
    
    num_params=4;
    round_num=1;
    if param.useFittedParam
        Phi=modelOsc.res{1,1}.model_prams.Phi;
        [keep_a, fitted_w] = get_rotmat_pam(Phi);
        Q=diag(modelOsc.res{1,1}.model_prams.Q)';
        keep_sigma=Q(1:2:end);
    end
    
    while round_num<param.osc_limit && (AIC_all(end)<AIC_all(max(1,end-1)) || round_num<=min_round) %Keep adding oscillations if AIC is decreasing or we haven't reached the minimum number of oscillations
        %         display(['Round Num: ' num2str(round_num)])
        %         display(['Min Round: ' num2str(min_round)])
        
        fig_name=[subject_fd_path 'sub' num2str(sub) '_r' num2str(round_num) '_AR_fit_order' num2str(ar_order) '.fig'];
        [AIC_params.add_freq, AIC_params.add_rad, AIC_params.freqs2, AIC_params.r_pole2] = ar_init(innov_pred,Fs,fig_name,ar_order); %currently using largest pole
        
        if param.useFittedParam
            init_params.f_init=[fitted_w*Fs/(2*pi) AIC_params.add_freq];
            init_params.a_init=[keep_a AIC_params.add_rad];
            init_params.sigma2_init=[keep_sigma 1];
        else
            init_params.f_init=[init_params.f_init AIC_params.add_freq]; % initial center frequency (Hz)
            init_params.a_init=(0.98^3)*ones(1,length(init_params.f_init)); % initial radius
            init_params.sigma2_init=ones(1,length(init_params.f_init)); % initial variance
        end
        
        init_params.NNharm=[init_params.NNharm 1]; %harmonics for each oscillation (set to 1 for all)
        
        if AR1 %Will we fit an AR1?
            noise_osc=[1];
            init_params.f_init(1)=0; %make sure that base is AR1 (0 Hz)
        end
        
        vmp_param.kappa=vmp_param.kappa(1)*ones(1,length(init_params.f_init)); %concentration parameter for Von Mises Prior
        vmp_param.mu = (init_params.f_init*2*pi)/Fs; %center von mises prior at initial frequency
        vmp_param.osc_num = [1:length(init_params.f_init)];
        clear modelOsc
        
        [modelOsc]=ssp_decomp(param.data.y(sub,:)',Fs,startID,endID,em_it, eps,init_params,vmp_param,noise_osc,fname,plot_on,auto_on, prior_on, param.updateR);
        save([subject_fd_path 'sub' num2str(sub) '_r' num2str(round_num) '.mat'],'modelOsc', 'vmp_param', 'init_params', 'noise_osc','AIC_params');
        [AIC_temp, pn_temp]=AIC_calc(modelOsc,1);
        AIC_all=[AIC_all AIC_temp];
        no_params=[no_params pn_temp];
        num_params=num_params+3;
        
        [y_pred,y_smooth,y_filt,y]=find_innovations(modelOsc,1); %will need to fix this for more windows
        y_tot=length(y);
        innov_pred=y-y_pred(1:y_tot)';
        
        round_num=round_num+1;
        
        if param.useFittedParam
            Phi=modelOsc.res{1,1}.model_prams.Phi;
            oscnum=size(Phi,1)/2;
            keep_a=zeros(1,oscnum);
            fitted_w=zeros(1,oscnum);
            for oscii=1:oscnum
                osca=oscii*2-1;
                [keep_a(oscii), fitted_w(oscii)] = get_rotmat_pam(Phi(osca:osca+1,osca:osca+1));
            end
            Q=diag(modelOsc.res{1,1}.model_prams.Q)';
            keep_sigma=Q(1:2:end);
        end
        
    end
    
    save([subject_fd_path 'sub' num2str(sub) '_AIC.mat'],'AIC_all','no_params');
    %disp([save_fpath fig_fold 'sub' num2str(sub) '_AIC.mat'])
    disp(['Subject ' num2str(sub) ' done.'])
    
end

end