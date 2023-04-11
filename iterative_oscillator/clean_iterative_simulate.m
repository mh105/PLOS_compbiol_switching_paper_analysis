%% Running a series of simulations on the Iterative Oscillation Model for a series of SNR levels and alpha center frequencies
%by Amanda M Beck 3/22/21

addpath chronux-helpers %add selected helper functions from chronux (small alterations)

%Parameters to Change
base_path='/purdonlab/users/ambeck/AR_model/propofol_matlab/ar_modeling/Matsuda_Hugo/data/aging/model_selection/Figs_030921_orig/';
hzs={'alpha6', 'alpha8','alpha10','alpha12','alpha14'};
alpha_hz=[6,8,10,12,14]; %Center of Alpha Oscillations that correspond to folders (hzs)
alpha_scaling = [0.1 0.25 0.5 0.75 1 1.25 1.5]; %Scaling factor on alpha state noise standard deviation

%priors={'no_prior','laplace','jeff','inv_gamma','approx_zeronorm'};
%%possible priors
priors={'inv_gamma'};
number_iter=50;
Fs=80; %Sampling Frequency in Hz
window_len=10; %window length in seconds
specparams.tapers =[5 9]; %Control tapers for investigating simulated data
run_AR1=True; %Do you want to compare AR1 and AR2 slow components? 


%% Run over all iterations
T=window_len*Fs;
specparams.Fs=Fs;
    
for ni=1:number_iter %run through a number of iterations
    %Create the proper folder structure
    topfold=['iteration' num2str(ni)];
    eval([ 'mkdir ' topfold]);
    eval([' cd ' topfold]);  
    for ii=1:length(hzs)
        eval(['mkdir ' hzs{ii}])
        eval(['cd ' hzs{ii}])
        for jj=1:length(priors)
            if run_AR1
                eval(['mkdir ' priors{jj} '_AR1'])
            end
            eval(['mkdir ' priors{jj} '_AR2'])
        end
        eval(['cd ..'])
    end

    %Simulate Oscillations
    sig_meas=sqrt(8); %Standard Deviation of Observation Noise (sig=sigma)
    noise_meas=normrnd(0,sig_meas,[1,T]); %Observation Noise series

    %Slow Oscillation
    sig_slow=sqrt(5);%Standard Deviation of Slow State Noise (sig=sigma)
    Phis=get_rot_mat(0.97,.725*2*pi/Fs); %Matsuda Transition Matrix for r and omega
    noise_slow1=normrnd(0,sig_slow,[1,T]); %real noise component
    noise_slow2=normrnd(0,sig_slow,[1,T]); %imaginary noise component
    noise_slow=[noise_slow1; noise_slow2];
    x_slow_c=zeros(2,T+1);
    for ii=2:T+1
        %x_slow_c = "x_slow complex"
        x_slow_c(:,ii)=noise_slow(:,ii-1)+Phis*x_slow_c(:,ii-1); %Propogate forward through the matsuda model to create the signal
    end
    x_slow1=x_slow_c(1,2:end)+i*x_slow_c(2,2:end); %Save real and imaginary parts as a complex number
    
    %Alpha Oscillations
    for num_num=1:length(alpha_hz) %Run through one center frequency at a time
        sig_alpha=sqrt(5.8)*alpha_scaling; %Standard Deviation of Alpha State Noise, testing different scales/SNRs
        x_alpha1=zeros(length(sig_alpha),T);
        Phia=get_rot_mat(0.94, alpha_hz(num_num)*2*pi/Fs);
        for ii=1:length(sig_alpha)
            sig_alpha(ii);
            noise_alpha1=normrnd(0,sig_alpha(ii),[1,T]); %real noise component
            noise_alpha2=normrnd(0,sig_alpha(ii),[1,T]); %imaginary noise component
            noise_alpha=[noise_alpha1; noise_alpha2];
            x_alpha_c=zeros(2,T+1);
            for jj=2:T+1
                %x_alpha_c="x_alpha_complex"
                x_alpha_c(:,jj)=noise_alpha(:,jj-1)+Phia*x_alpha_c(:,jj-1); %Propogate forward through the matsuda model to create the signal
            end
            x_alpha1(ii,:)=x_alpha_c(1,2:end)+i*x_alpha_c(2,2:end); %Save real and imaginary parts as a complex number
        end

        % Produce "Recorded" Data
        x_all=real(x_slow1+x_alpha1+noise_meas); %adding all states and noise to find y or "recorded" data
        % Graph "Recorrded" Data
        graph_all(x_all,specparams); %look at what the SNR scaling is like

        %Save Simulation parameters and time series for later reference
        sim_param.slow.param=a_slow;
        sim_param.slow.sig=sig_slow;
        sim_param.slow.x=x_slow1;

        sim_param.alpha.param=a_alpha;
        sim_param.alpha.sig=sig_alpha;
        sim_param.alpha.x=x_alpha1;

        sim_param.noise.sig=sig_meas;
        sim_param.noise.x=noise_meas;
        sim_param.y=x_all;

        param.save_fpath=[base_path topfold '/' hzs{num_num} '/'];
        save(['/autofs/cluster' param.save_fpath 'simulated_data.mat'],'sim_param');
        

        %% Parameters for the iterative model
        param.addpath_tools=false;
        param.mount=false; %Are you mounting the drive? (ex. VPN, mount drive, run code on own computer)
        param.useFittedParam=true; %Should we use fitted parameters from iteration i as the initial parameters in iteration i+1
        param.updateR=true; %Should we update Observation Noise Variance on each EM iteration? 
        param.data.y=x_all; %This is the "recorded" data (must be real)
        param.data.win_no=1; %keep as 1 window for now, may test more windows in future
        param.data.win_length=window_len;
        param.data.Fs=Fs;
        min_round=1; %Optional: Set the minimum number of oscillations
        %you want to investigate

        %Von Mises Prior parameters:
        osc_num=[];
        kappa=[];
        mu_f=[]; %if these three parameters are empty, function will use usual parameters, override by defining vmp_param.kappa (concentration parameter) and vmp_param.mu (center frequency (Hz))
        vmp_off=false; %turn VMP off to see what it's doing (NOT RECOMMENDED FOR DATA ANALYSIS)


        for ii=1:length(priors)
            if run_AR1
                param.fig_fold=[ priors{ii} '_AR1/'];
                param.prior_on=priors{ii};
                modelOsc=iterative_model(param, min_round, kappa, mu_f, osc_num,true); %Run iterative model with AR1 slow oscillation
            end
            param.fig_fold=[ priors{ii} '_AR2/'];
            param.prior_on=priors{ii};
            modelOsc=iterative_model(param, min_round, kappa, mu_f, osc_num,false); %Run iterative model with AR2 bslow oscillation
        end
    end
    eval(['cd ..'])

    %% Identifying Minimum AIC Model and Graphing AIC
  
    num_sub=length(alpha_scaling); %Number of "subjects" in the folder, iterative model loops over SNR levels, but next part needs to loop explicitly
    for jj=1:length(hzs)
        for ii=1:length(priors)
            if run_AR1
                %AR1 only stats
                osc_num_AR1=zeros(1,num_sub);
                AICw1_mat=zeros(num_sub,7);
            end
            %Overall Stats
            osc_num_overall=zeros(2,num_sub);
            AICw_mat=zeros(num_sub,7); %this 7 is just making enough room for all fitted oscillations (up to 7)
            %AR2 only stats
            osc_num_AR2=zeros(1,num_sub);
            AICw2_mat=zeros(num_sub,7);

            for sub=1:num_sub
                base= ['/autofs/cluster' base_path 'iteration' num2str(ni) '/' hzs{jj} '/' priors{ii}];
                AIC2= load([base '_AR2/sub' num2str(sub) '_AIC.mat']);
                [AR2_min osc_num_AR2(sub)]=min(AIC2.AIC_all);
                
                if run_AR1
                    AIC1= load([base '_AR1/sub' num2str(sub) '_AIC.mat']);
                    AIC_all=[AIC1.AIC_all AIC2.AIC_all];
                    osc_base=[1:length(AIC1.AIC_all) 1:length(AIC2.AIC_all)];
                    ar1_cutoff=length(AIC1.AIC_all);
                    [min_AIC min_ind]=min(AIC_all);
                    if min_ind <= ar1_cutoff
                        osc_num_overall(1,sub)=1;
                    else
                        osc_num_overall(1,sub)=2;
                    end
                    osc_num_overall(2,sub)=osc_base(min_ind);
                    [AR1_min osc_num_AR1(sub)]=min(AIC1.AIC_all);
                
                
                    deltas1=AIC1.AIC_all-min_AIC;
                    deltas2=AIC2.AIC_all-min_AIC;
                    AIC_w1=exp(-deltas1/2)/sum(exp(-deltas1/2)); %AIC weights for base AR1
                    AIC_w2=exp(-deltas2/2)/sum(exp(-deltas2/2)); %AIC weights for base AR2
                    AICw_mat(sub,1:length([AIC_w1 AIC_w2]))=[AIC_w1 AIC_w2];
                    
                    %Graph AIC over models
                    figure
                    graph_AIC(AIC1.AIC_all,deltas1,AIC_w1); %make AIC graph
                    graph_AIC(AIC2.AIC_all,deltas2,AIC_w2);
                    legend('AR1','AR2')
                    savefig(gcf,[base '_AR1/sub' num2str(sub) '_AIC.fig']);
                    savefig(gcf,[base '_AR2/sub' num2str(sub) '_AIC.fig']); 
                else
                    %Set up AR2 stats
                    [min_AIC min_ind]=min(AIC2.AIC_all);
                    deltas2=AIC2.AIC_all-min_AIC;
                    AIC_w2=exp(-deltas2/2)/sum(exp(-deltas2/2));
                    
                    %Set up overall stats to be compatible with choosing
                    %between AR1 and AR2
                    osc_num_overall(1,sub)=2;
                    osc_num_overall(2,sub)=min_ind;
                    AICw_mat=AICw2;
                    
                    %Graph AIC over models
                    figure
                    graph_AIC(AIC2.AIC_all,deltas2,AIC_w2); %plot AR2 only
                    savefig(gcf,[base '_AR2/sub' num2str(sub) '_AIC.fig']); 
                end
            end
            
            close all
            if run_AR1
                save([ base '_AR1/selected_osc_num.mat'],'osc_num_overall','osc_num_AR1','osc_num_AR2','AICw_mat','AICw1_mat','AICw2_mat');
            end
            save([ base '_AR2/selected_osc_num.mat'],'osc_num_overall','osc_num_AR2','AICw_mat','AICw2_mat');
     
        end
        end
end

display('All Scripts done')


%% Local Functions
function graph_AIC(AIC_all,deltas,AIC_w)
    subplot(3,1,1)
    plot([1:length(AIC_w)],AIC_all,'-*'); title('AIC values');hold on
    subplot(3,1,2);
    plot([1:length(AIC_w)],deltas,'-*'); title('AIC deltas'); hold on
    subplot(3,1,3);
    plot([1:length(AIC_w)],AIC_w,'-*'); title('AIC weights');hold on
end


function graph_all(x_mat,specparams)
    figure
    for ii=1:size(x_mat,1)
        [S, f]=mtspectrumc(x_mat(ii,:),specparams);
        plot(f,20*log10(S)); hold on;
    end
end

