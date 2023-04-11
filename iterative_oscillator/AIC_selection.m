function [osc_num_overall]=AIC_selection(base_path, priors, num_sub, run_AR1)
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

        base= [ base_path priors{ii}];

        for sub=1:num_sub
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
                AICw_mat=AIC_w2;

                %Graph AIC over models
                f = figure;
                graph_AIC(AIC2.AIC_all,deltas2,AIC_w2); %plot AR2 only
                savefig(gcf,[base '_AR2/sub' num2str(sub) '_AIC.fig']); 
                close(f)
            end
            
            if run_AR1
                save([ base '_AR1/selected_osc_num.mat'],'osc_num_overall','osc_num_AR1','osc_num_AR2','AICw_mat','AICw1_mat','AICw2_mat');
            end
            save([ base '_AR2/selected_osc_num.mat'],'osc_num_overall','osc_num_AR2','AICw_mat','AICw2_mat');

        end
    end

end