function [f_y, S_ar]=plot_summary_component(modelOsc,time_ind,fig_tags)

%addpath(genpath('../osc_harm_decomp'))
addpath(genpath('../spectral_analysis'))


%% Gather parameters and compute PSD
staPointId = modelOsc.startPointId;
endPointId = modelOsc.endPointId;
Nwind  = length(staPointId);
Fs     = modelOsc.Fs;

NNharm=modelOsc.init_params.NNharm;
Nosc     = sum(NNharm);    % Total number of oscillation
Nosc_ind = length(NNharm); % Number of indpt oscillations

specparams.Fs     = Fs;
%specparams.tapers = [60 119]; for 2 min data
specparams.tapers =[10 19]; %for 10 sec data
tpoints=mean(staPointId-endPointId);
nfft   =max(2^(nextpow2(tpoints)),tpoints)/2+1;

f_tot=zeros(Nosc,Nwind);
a_tot=zeros(Nosc,Nwind);
q_tot=zeros(Nosc,Nwind);
R_tot=zeros(1,Nwind);
f_y=0.01:0.01:120;

s_tot_param=zeros(length(f_y),Nwind);
s_param    =zeros(length(f_y),Nwind,Nosc);
s_mt       =zeros(nfft,Nwind);
s_mt_tot   =zeros(nfft,Nwind,Nosc);



for tt=1:Nwind
    Phi_tmp=modelOsc.res{1,tt}.model_prams.Phi;
    Q_tmp=modelOsc.res{1,tt}.model_prams.Q;
    R_tmp=modelOsc.res{1,tt}.model_prams.R;
    
    for nosc=1:Nosc
        Phi_n= Phi_tmp(((nosc-1)*2+1):2*nosc,((nosc-1)*2+1):2*nosc);
        [a_tmp,w_tmp]=get_rotmat_pam(Phi_n);
        a_tot(nosc,tt)=a_tmp;
        f_tot(nosc,tt)=w_tmp*Fs/(2*pi);
        q_tot(nosc,tt)=Q_tmp((nosc-1)*2+1,(nosc-1)*2+1);
        
        [s_i_tmp, fi]=mtspectrumc(modelOsc.res{1,tt}.x_t_n((nosc-1)*2+1,2:end),specparams);

        s_mt_tot(:,tt,nosc) =s_i_tmp;
    end
    
    R_tot(1,tt)=R_tmp;
    [H_tot, H_i]=get_theoretical_psd(f_y,Fs,f_tot(:,tt)',a_tot(:,tt)',q_tot(:,tt)');  % q_tot was previously missing transpose
    
    s_tot_param(:,tt)=H_tot';
    s_param(:,tt,:)=H_i';
    
    
    
    [s_mt_tmp,fmt]=mtspectrumc(modelOsc.res{1,tt}.y,specparams);
    s_mt(:,tt)=s_mt_tmp;
end





%% Scatter plot fitted model parameter for each window
colorCur = [0         0.4470    0.7410;
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840];

% 
% figure
% subplot(2,2,1); hold on
% 
% for nosc=1:Nosc_ind
%     for nosc_r=1:NNharm(1,nosc)
%         curId= ((sum(NNharm(1,1:nosc)) - NNharm(1,nosc)) + nosc_r) ;
%         scatter(staPointId/(60*Fs), f_tot(curId,:), 20,colorCur(nosc,:))
%     end
% end
% title('Freq'); xlabel('[min]')
% 
% subplot(2,2,2); hold on
% for nosc=1:Nosc_ind
%     for nosc_r=1:NNharm(1,nosc)
%         curId= ((sum(NNharm(1,1:nosc)) - NNharm(1,nosc)) + nosc_r) ;
%         scatter(staPointId/(60*Fs), a_tot(curId,:), 20,colorCur(nosc,:))
%     end
% end
% title('a'); xlabel('[min]')
% 
% subplot(2,2,3); hold on
% for nosc=1:Nosc_ind
%     for nosc_r=1:NNharm(1,nosc)
%         curId= ((sum(NNharm(1,1:nosc)) - NNharm(1,nosc)) + nosc_r) ;
%         scatter(staPointId/(60*Fs), q_tot(curId,:), 20,colorCur(nosc,:))
%     end
% end
% title('\sigma^2'); xlabel('[min]')
% 
% subplot(2,2,4); hold on
% scatter(staPointId/(60*Fs), R_tot(1,:), 20,'k')
% title('Freq'); xlabel('[min]')
% title('R'); xlabel('[min]')


%% Plot multitpaer and parametric Spectra

% Color scale
% cmax=50;
% cmin=-60;
% 
% figure
% p=subplot(2,1,1);
% imagesc(staPointId/(60*Fs), f_y, 10*log10(s_tot_param))
% colormap(jet)
% axis xy;
% ylabel('Freq [Hz]')
% ylim([0 50])
% title('Parametric PSD [dB]')
% xlabel(['Time [min]'])
% caxis([cmin cmax])
% colorbar
% 
% p2=subplot(2,1,2);
% imagesc(staPointId/(60*Fs), fmt, 10*log10(s_mt))
% colormap(jet)
% axis xy;
% ylabel('Freq [Hz]')
% ylim([0 30])
% title('Multitaper [dB]')
% xlabel(['Time [min]'])
% caxis([cmin cmax])
% colorbar
% 
% linkaxes([p,p2])
% ylim([0 30])

%% Plot Components and PSD

if any(fig_tags==1)
    figure; 
    ax1=subplot(2,1,1);hold on
    plot(fmt, 10*log10(s_mt(:,time_ind)   ),'k')
    plot( f_y, 10*log10(s_tot_param(:,time_ind)),':k')
    % for jj=1:Nosc_ind
    % plot(f_y,10*log10(H_i(jj,:)));
    % end

    % 
    %legend()
    for ii =1:size(s_param,3)
        if ii<=7
            plot( f_y, 10*log10(s_param(:,time_ind,ii)), 'Color', colorCur(ii,:), 'linewidth', 2)
        else
            plot( f_y, 10*log10(s_param(:,time_ind,ii)), 'linewidth', 2)
        end
        %plot( fmt, 10*log10(s_mt_tot(:,tt,ii)), 'Color', colorCur(ii,:), 'linewidth', 2)
        

    end
    xlabel('Frequency (Hz)');
    ylabel('Power (dB)');
    title('Spectra of AR Components');
    set(gca,'FontSize',16)
    xlim([0 Fs/2]);
    hold off;

    ax2=subplot(2,1,2); hold on
    plot(fmt, 10*log10(s_mt(:,time_ind)),'k');
    for ii =1:size(s_param,3)
        if ii<= 7
            plot( fi, 10*log10(s_mt_tot(:,time_ind,ii)), 'Color', colorCur(ii,:), 'linewidth', 2)
        else
            plot( fi, 10*log10(s_mt_tot(:,time_ind,ii)), 'linewidth', 2)
        end
        %plot( fmt, 10*log10(s_mt_tot(:,tt,ii)), 'Color', colorCur(ii,:), 'linewidth', 2)

    end
    xlabel('Frequency (Hz)');
    ylabel('Power (dB)');
    title('Spectra of Estimated Oscillations');
    set(gca,'FontSize',16)
    xlim([0 Fs/2]);
    hold off
    
    linkaxes([ax1,ax2],'x')
end

S_ar=squeeze(s_param(:,time_ind,:));
%S_ar=[squeeze(s_tot_param(:,1)) S_ar];



%Color scale
cmax=50;
cmin=-60;

if any(fig_tags==2)
    figure
    p=subplot(2,1,1);
    imagesc(staPointId/(60*Fs), f_y, 10*log10(s_tot_param))
    colormap(jet)
    axis xy;
    ylabel('Freq [Hz]')
    ylim([0 50])

    title('Parametric PSD [dB]')
    xlabel(['Time [min]'])
    caxis([cmin cmax])
    colorbar

    p2=subplot(2,1,2);
    imagesc(staPointId/(60*Fs), fmt, 10*log10(s_mt))
    colormap(jet)
    axis xy;
    ylabel('Freq [Hz]')
    ylim([0 30])
    title('Multitaper [dB]')
    xlabel(['Time [min]'])
    caxis([cmin cmax])
    colorbar

    linkaxes([p,p2])
    %ylim([0 30])
end



%% Plot temporal signal
% 10/10/18 Amanda adjusted the x axis to reflect seconds

sumHarmn=1; % Plot all harmonics separately or together

x_t_n_tot=zeros(2*Nosc,(endPointId(time_ind)-staPointId(time_ind)+1));
%y_tot=zeros(1,(endPointId(end)-staPointId(1)));
y_tot=zeros(1,endPointId(time_ind)-staPointId(time_ind)+1);
%ta=(0:endPointId(end)-staPointId(1))/Fs;
ta=(staPointId(time_ind):endPointId(time_ind))/Fs;
size(ta);

for tt=time_ind
    x_t_n_tot=modelOsc.res{1,tt}.x_t_n(:,2:end);
    %x_t_n_tot(:, (staPointId(tt):endPointId(tt))-staPointId(1)+1)=x_t_n_cur;
    %y_tot(1,(staPointId(tt):endPointId(tt))-staPointId(1)+1)=modelOsc.res{1,tt}.y';
    y_tot(1,:)=modelOsc.res{1,tt}.y';
end

size(y_tot);
if any(fig_tags==3)
    figure;
    p=subplot(Nosc_ind+2,1,1);hold on
    plot(ta, y_tot, 'color','k', 'LineWidth',2)
    title('Original Time Series')
    size(sum(x_t_n_tot(1:2:end,:),1))
    plot(ta, sum(x_t_n_tot(1:2:end,:),1), 'color','g')
    legend('Y real','x_t sum');
    set(gca,'FontSize',14)
    size(x_t_n_tot);
    size(ta);

    for nosc=1:Nosc_ind
        pcur=subplot(Nosc_ind+2,1,nosc+1);hold on
        set(gca,'FontSize',14)
        
        if sumHarmn
            curId= 2*((sum(NNharm(1,1:nosc)) - NNharm(1,nosc)) + 1) -1;
            curR=curId:2:curId+2*(NNharm(1,nosc)-1);
            x_r_sum1= sum(x_t_n_tot ( curR,:),1);
            x_r_sum2= sum(x_t_n_tot( curR+1,:),1);

            plot(ta, x_r_sum1, 'linewidth',2, 'color','b')
            %plot(ta, x_r_sum2, 'linewidth',2, 'color',[0.6 0.6 1])


        else

            for nosc_r=1:NNharm(1,nosc)
                curId= 2*((sum(NNharm(1,1:nosc)) - NNharm(1,nosc)) + nosc_r) -1;

                plot(ta, x_t_n_tot(curId,:), 'linewidth',2, 'color','b')
                plot(ta, x_t_n_tot(curId+1,:), 'linewidth',2, 'color',[0.6 0.6 1])

            end


        end
        legend('Real Signal','Imaginary Signal');
        title(['Oscillation ' num2str(nosc)]);

        p=[p, pcur];
    end

    pcur=subplot(Nosc_ind+2,1,Nosc_ind+2);
    p=[p, pcur];
    if Nosc_ind>1
        signal1=sum(x_t_n_tot(1:2:4,:),1);
        size(signal1)
        plot(ta, signal1)
    %resid=y_tot(2:end)-sum(x_t_n_tot(1:2:end,:),1)%modelOsc.res{1,time_ind}.model_prams.Phi*x_t_n_tot(:,1:(end-1));
    %resid=y_tot
    %plot(ta(2:end),resid,'linewidth',2,'color','b');
        title('Signal (Slow + Alpha)')
        set(gca,'FontSize',14)
    end

    linkaxes(p)
end

end
