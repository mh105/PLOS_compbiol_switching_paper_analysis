%% Experiment 2 - VB learning comparisons >>> NEW UPDATES FOR REVIEWERS
% Simulations to compare the improved VB learning algorithm with the
% original one on learning segmental SSM parameters. Performance is
% evaluated at multiple levels:
% - correct segmentation
% - parameter estimation
% - free energy bounds on log likelihood (observed data fitting)
close all; clear all; clc

addpath(genpath('state_space_spindle_detector_code'))

% Some changes:
% - sample parameters from the same wider distributions
% - test parameter learning with varying sample sizes
% - check parameter convergence

%% Run simulations
% true and fixed simulation parameters
iter = 200;
T = 3200;
T_used_list = [100, 200, 400, 800, 1600, 3200];
F1 = 0.9; std1 = sqrt(2);
F2 = 0.7;  std2 = sqrt(10);
R = 0.1; % observation noise covariance
dwell_prob = 0.95; % HMM state dwell probability

nT_used = length(T_used_list);
original_percent_correct = zeros(iter,nT_used);
improved_percent_correct = zeros(iter,nT_used);
random_percent_correct = zeros(iter,nT_used);
static_percent_correct = zeros(iter,nT_used);
IMM_percent_correct = zeros(iter,nT_used);
learnt_F1 = zeros(2,iter,nT_used);
learnt_F2 = zeros(2,iter,nT_used);
learnt_var1 = zeros(2,iter,nT_used);
learnt_var2 = zeros(2,iter,nT_used);
learnt_R = zeros(2,iter,nT_used);
learnt_dwellp = zeros(2,iter,nT_used);
logL_bounds = zeros(2,iter,nT_used);

ssm_models = cell(iter,nT_used);
curr_dwell_list = zeros(1,iter);
s_list = zeros(T,iter);

%% begin iterations
for n = 1:iter
    %%
    disp(['Iteration # ', num2str(n)])
    
    % generate two AR(1) SSMs and a discrete state HMM for switching
    x1 = [randn(1)*std1, zeros(1,T-1)];
    x2 = [randn(1)*std2, zeros(1,T-1)];
    s = [datasample([1,-1],1), zeros(1,T-1)];
    for ii = 2:T
        x1(ii) = F1*x1(ii-1) + randn(1)*std1;
        x2(ii) = F2*x2(ii-1) + randn(1)*std2;
        if rand(1) >= dwell_prob
            s(ii) = s(ii-1) * -1; % switch
        else
            s(ii) = s(ii-1); % stay
        end
    end
    s(s==-1) = 0;
    
    % generate observed data
    y = x1.*(s==1) + x2.*(s==0) + randn(1,T).*sqrt(R);
    
    % introduce uncertainties in parameters to initialize VB learning
    curr_F1 = rand(1)*0.4+0.6; % 0.6-1.0
    curr_F2 = rand(1)*0.4+0.6; % 0.6-1.0
    curr_std1 = sqrt(rand(1)*14+1); % 1-15
    curr_std2 = sqrt(rand(1)*14+1); % 1-15
    curr_R = rand(1)*0.19+0.01; % 0.01-0.2
    curr_dwell_prob = rand(1)*0.09+0.9; % 0.9-0.99
    
    % store the info from the current iteration
    curr_dwell_list(n) = curr_dwell_prob;
    s_list(:,n) = s;
    
    for t = 1:nT_used
        % Use a varying data length
        T_used = T_used_list(t);
        y_used = y(1:T_used);
        s_used = s(1:T_used);
        disp(['     T length: ', num2str(T_used)])
        
        % create ssm objects
        o1 = ssm({arn(1)},curr_F1,curr_std1^2,0,curr_std1^2,1,curr_R,y_used);
        o2 = ssm({arn(1)},curr_F2,curr_std2^2,0,curr_std2^2,1,curr_R,y_used);
        
        % store the ssm objects for later processing and plotting
        ssm_models{n,t} = [o1, o2];
        
        % Run VB learning - with M steps
        [Mprob, ~, obj_array, A, ~, logL_bound] = VBlearn_original([o1, o2], 'dwellp', curr_dwell_prob, 'verbose', false);
        Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
        % figure out the label of two models
        if obj_array(1).Q < obj_array(2).Q
            first_idx = 1;
            second_idx = 2;
        else
            first_idx = 2;
            second_idx = 1;
        end
        original_percent_correct(n,t) = mean(abs(Mprob(first_idx,:) - s_used) <= 0.1);
        
        learnt_F1(1,n,t) = obj_array(first_idx).F;
        learnt_F2(1,n,t) = obj_array(second_idx).F;
        learnt_var1(1,n,t) = obj_array(first_idx).Q;
        learnt_var2(1,n,t) = obj_array(second_idx).Q;
        learnt_R(1,n,t) = obj_array(1).R;
        learnt_dwellp(1,n,t) = trace(A)/2;
        logL_bounds(1,n,t) = logL_bound(end);
        
        [Mprob, ~, obj_array, A, ~, ~, ~, logL_bound_new] = VBlearn([o1, o2], 'dwellp', curr_dwell_prob, 'verbose', false);
        Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
        % figure out the label of two models
        if obj_array(1).Q < obj_array(2).Q
            first_idx = 1;
            second_idx = 2;
        else
            first_idx = 2;
            second_idx = 1;
        end
        improved_percent_correct(n,t) = mean(abs(Mprob(first_idx,:) - s_used) <= 0.1);
        
        learnt_F1(2,n,t) = obj_array(first_idx).F;
        learnt_F2(2,n,t) = obj_array(second_idx).F;
        learnt_var1(2,n,t) = obj_array(first_idx).Q;
        learnt_var2(2,n,t) = obj_array(second_idx).Q;
        learnt_R(2,n,t) = obj_array(1).R;
        learnt_dwellp(2,n,t) = trace(A)/2;
        logL_bounds(2,n,t) = logL_bound_new(end);
        
        %%
        % Run comparison inference algorithms
        Mprob = rand(1,T_used); % random segmentation
        Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
        random_percent_correct(n,t) = mean(abs(Mprob - s_used) <= 0.1);
        
        % figure out the label of two models
        if o1.Q < o2.Q
            first_idx = 1;
        else
            first_idx = 2;
        end
        
        Mprob = switching([o1, o2],'method','static','dwellp',curr_dwell_prob);
        Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
        static_percent_correct(n,t) = mean(abs(Mprob(first_idx,:) - s_used) <= 0.1);
        
        Mprob = switching([o1, o2],'method','IMM','dwellp',curr_dwell_prob);
        Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
        IMM_percent_correct(n,t) = mean(abs(Mprob(first_idx,:) - s_used) <= 0.1);
        
    end
end

%% Save simulation results
% save('Experiment2_review_simulation_results.mat')
load('Experiment2_review_simulation_results.mat')

%% Robustness characterization 1: segmentation accuracy

select_t = 2;

figure
subplot(5,1,1)
histogram(random_percent_correct(:,select_t),0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['Random Segmentation: mean = ', sprintf('%.3f', mean(random_percent_correct(:,select_t)))])
xticks([0:0.1:1]) %#ok<*NBRAK>
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(random_percent_correct(:,select_t)), mean(random_percent_correct(:,select_t))], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,2)
histogram(static_percent_correct(:,select_t),0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['Static Switching Method: mean = ', sprintf('%.3f', mean(static_percent_correct(:,select_t)))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(static_percent_correct(:,select_t)), mean(static_percent_correct(:,select_t))], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,3)
histogram(IMM_percent_correct(:,select_t),0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['IMM Switching Method: mean = ', sprintf('%.3f', mean(IMM_percent_correct(:,select_t)))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(IMM_percent_correct(:,select_t)), mean(IMM_percent_correct(:,select_t))], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,4)
histogram(original_percent_correct(:,select_t),0:0.02:1, 'FaceColor', [0.8500 0.3250 0.0980])
title(['Original VB learning with annealing: mean = ', sprintf('%.3f', mean(original_percent_correct(:,select_t)))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(original_percent_correct(:,select_t)), mean(original_percent_correct(:,select_t))], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,5)
histogram(improved_percent_correct(:,select_t),0:0.02:1, 'FaceColor', [0 0.4470 0.7410])
title(['Improved VB learning: mean = ', sprintf('%.3f', mean(improved_percent_correct(:,select_t)))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(improved_percent_correct(:,select_t)), mean(improved_percent_correct(:,select_t))], ylim, 'r--', 'LineWidth', 1)

xlabel('Percent Correct Segmentation')

%% Robustness characterization 2: parameter estimation
figure
subplot(2,2,1)
hold on
swarmchart(ones(1,size(learnt_F1,2)), learnt_F1(1,:,select_t), [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(1,size(learnt_F1,2))*2, learnt_F1(2,:,select_t), [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(1,size(learnt_F2,2))*3.5, learnt_F2(1,:,select_t), [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(1,size(learnt_F2,2))*4.5, learnt_F2(2,:,select_t), [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5, 5])
ylim([0.49, 1.01])
plot(xlim, [0.6, 0.6], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, [1.0, 1.0], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([0.5,2.6], ones(1,2)*F1, '-', 'Color', 'k', 'LineWidth', 3)
plot([2.8,5], ones(1,2)*F2, '-', 'Color', 'k', 'LineWidth', 3)
xticks([1,2,3.5,4.5])
xticklabels({'Ori F1', 'Imp F1', 'Ori F2', 'Imp F2'})
xlabel('\bf F \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Estimated Parameters')
set(gca,'FontSize',16)
title('SSM Transition Matrix F', 'FontSize', 30)

subplot(2,2,2)
hold on
swarmchart(ones(1,size(learnt_var1,2)), learnt_var1(1,:,select_t), [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(1,size(learnt_var1,2))*2, learnt_var1(2,:,select_t), [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(1,size(learnt_var2,2))*3.5, learnt_var2(1,:,select_t), [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(1,size(learnt_var2,2))*4.5, learnt_var2(2,:,select_t), [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5, 5])
ylim([0, 17])
plot(xlim, [1, 1], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, [15, 15], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([0.5,2.6], ones(1,2)*std1^2, '-', 'Color', 'k', 'LineWidth', 3)
plot([2.8,5], ones(1,2)*std2^2, '-', 'Color', 'k', 'LineWidth', 3)
xticks([1,2,3.5,4.5])
xticklabels({'Ori Q1', 'Imp Q1', 'Ori Q2', 'Imp Q2'})
xlabel('\bf Q \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Estimated Parameters')
set(gca,'FontSize',16)
title('SSM State Noise Covariance Q', 'FontSize', 30)

subplot(2,2,3)
hold on
swarmchart(ones(1,size(learnt_R,2)), learnt_R(1,:,select_t), [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(1,size(learnt_R,2))*2, learnt_R(2,:,select_t), [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5, 2.5])
ylim([-0.1, 0.4])
plot(xlim, ones(1,2)*(0.2), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, ones(1,2)*(0.01), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, ones(1,2)*R, '-', 'Color', 'k', 'LineWidth', 3)
xticks([1:2])
xticklabels({'Ori R', 'Imp R'})
xlabel('\bf R \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Estimated R')
set(gca,'FontSize',16)
title('SSM Observation Noise Covariance R', 'FontSize', 30)

subplot(2,2,4)
hold on
swarmchart(ones(1,size(learnt_dwellp,2)), learnt_dwellp(1,:,select_t), [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(1,size(learnt_dwellp,2))*2, learnt_dwellp(2,:,select_t), [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5, 2.5])
ylim([0.8, 1])
plot(xlim, ones(1,2)*(0.99), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, ones(1,2)*(0.9), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, ones(1,2)*dwell_prob, '-', 'Color', 'k', 'LineWidth', 3)
xticks([1:2])
xticklabels({'Ori P(Stay)', 'Imp P(Stay)'})
xlabel('\bf P(Stay) \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Estimated P(Stay)')
set(gca,'FontSize',16)
title('HMM Transition Stay Probability P(Stay)', 'FontSize', 30)

%% Plot how accuracy changes with increasing data length
figure
hold on
shadedErrorBar(T_used_list, mean(random_percent_correct), std(random_percent_correct)./sqrt(size(random_percent_correct, 1)), {'Color', [0.5 0.5 0.5]});
plot(T_used_list, mean(random_percent_correct), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.5 0.5 0.5])

shadedErrorBar(T_used_list, mean(static_percent_correct), std(static_percent_correct)./sqrt(size(static_percent_correct, 1)), {'Color', [0.5 0.5 0.5]});
plot(T_used_list, mean(static_percent_correct), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.5 0.5 0.5])

shadedErrorBar(T_used_list, mean(IMM_percent_correct), std(IMM_percent_correct)./sqrt(size(IMM_percent_correct, 1)), {'Color', [0.5 0.5 0.5]});
plot(T_used_list, mean(IMM_percent_correct), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.5 0.5 0.5])

shadedErrorBar(T_used_list, mean(original_percent_correct), std(original_percent_correct)./sqrt(size(original_percent_correct, 1)), {'Color', [0.8500 0.3250 0.0980]}, 1);
plot(T_used_list, mean(original_percent_correct), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.8500 0.3250 0.0980])

shadedErrorBar(T_used_list, mean(improved_percent_correct), std(improved_percent_correct)./sqrt(size(improved_percent_correct, 1)), {'Color', [0 0.4470 0.7410]});
plot(T_used_list, mean(improved_percent_correct), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0 0.4470 0.7410])

xlim([90, 4000])
ylim([0.6, 0.9])
yticks(0.6:0.05:0.9)
set(gca, 'XScale', 'log')
xticks(T_used_list)
ax = gca;
ax.XMinorTick = 'off';
xlabel('Number of Time Points')
ylabel('Percent Correct Segmentation')
set(gca,'FontSize',20)

%% Visualize the convergence of parameters during EM learning

% generate two AR(1) SSMs and a discrete state HMM for switching
x1 = [randn(1)*std1, zeros(1,T-1)];
x2 = [randn(1)*std2, zeros(1,T-1)];
s = [datasample([1,-1],1), zeros(1,T-1)];
for ii = 2:T
    x1(ii) = F1*x1(ii-1) + randn(1)*std1;
    x2(ii) = F2*x2(ii-1) + randn(1)*std2;
    if rand(1) >= dwell_prob
        s(ii) = s(ii-1) * -1; % switch
    else
        s(ii) = s(ii-1); % stay
    end
end
s(s==-1) = 0;

% generate observed data
y = x1.*(s==1) + x2.*(s==0) + randn(1,T).*sqrt(R);

% introduce uncertainties in parameters to initialize VB learning
curr_F1 = rand(1)*0.4+0.6; % 0.6-1.0
curr_F2 = rand(1)*0.4+0.6; % 0.6-1.0
curr_std1 = sqrt(rand(1)*14+1); % 1-15
curr_std2 = sqrt(rand(1)*14+1); % 1-15
curr_R = rand(1)*0.19+0.01; % 0.01-0.2
curr_dwell_prob = rand(1)*0.09+0.9; % 0.9-0.99

figure
ax1 = subplot(2,2,1);
ax2 = subplot(2,2,2);
ax3 = subplot(2,2,3);
ax4 = subplot(2,2,4);

maxEM = 10;

for t = [2,6]
    % Use a varying data length
    T_used = T_used_list(t);
    y_used = y(1:T_used);
    s_used = s(1:T_used);
    disp(['     T length: ', num2str(T_used)])
    
    % create ssm objects
    o1 = ssm({arn(1)},curr_F1,curr_std1^2,0,curr_std1^2,1,curr_R,y_used);
    o2 = ssm({arn(1)},curr_F2,curr_std2^2,0,curr_std2^2,1,curr_R,y_used);
    
    % Run VB learning - with M steps
    [Mprob, ~, obj_array, ~, ~, logL_bound, em_store] = VBlearn_original([o1, o2], 'dwellp', curr_dwell_prob, 'verbose', true, 'ht_thresh', 0, 'maxVB_iter', maxEM);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    % figure out the label of two models
    if obj_array(1).Q < obj_array(2).Q
        first_idx = 1;
        second_idx = 2;
    else
        first_idx = 2;
        second_idx = 1;
    end
    disp(['>>> VI-A accuracy: ', num2str(mean(abs(Mprob(first_idx,:) - s_used) <= 0.1))])
    
    % Get evolution of parameters
    F1s = cellfun(@(x) x{1}(first_idx).F, em_store);
    F2s = cellfun(@(x) x{1}(second_idx).F, em_store);
    Q1s = cellfun(@(x) x{1}(first_idx).Q, em_store);
    Q2s = cellfun(@(x) x{1}(second_idx).Q, em_store);
    Rs = cellfun(@(x) x{1}(first_idx).R, em_store);
    
    if t == 2
        axes(ax1)
    else
        axes(ax3)
    end
    hold on
    plot(0:length(logL_bound), F1s, '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.8500 0.3250 0.0980])
    plot(0:length(logL_bound), F2s, '-x', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.8500 0.3250 0.0980])
    
    if t == 2
        axes(ax2)
    else
        axes(ax4)
    end
    hold on
    plot(0:length(logL_bound), Q1s, '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.8500 0.3250 0.0980])
    plot(0:length(logL_bound), Q2s, '-x', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.8500 0.3250 0.0980])
    
    [Mprob, ~, obj_array, ~, ~, ~, ~, logL_bound_new, ~, ~, em_store] = VBlearn([o1, o2], 'dwellp', curr_dwell_prob, 'verbose', true, 'ht_thresh', 0, 'maxVB_iter', maxEM);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    % figure out the label of two models
    if obj_array(1).Q < obj_array(2).Q
        first_idx = 1;
        second_idx = 2;
    else
        first_idx = 2;
        second_idx = 1;
    end
    disp(['>>> VI-I accuracy: ', num2str(mean(abs(Mprob(first_idx,:) - s_used) <= 0.1))])
    
    % Get evolution of parameters
    F1s = cellfun(@(x) x{1}(first_idx).F, em_store);
    F2s = cellfun(@(x) x{1}(second_idx).F, em_store);
    Q1s = cellfun(@(x) x{1}(first_idx).Q, em_store);
    Q2s = cellfun(@(x) x{1}(second_idx).Q, em_store);
    
    if t == 2
        axes(ax1)
    else
        axes(ax3)
    end
    hold on
    plot(0:length(logL_bound_new), F1s, '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0 0.4470 0.7410])
    plot(0:length(logL_bound_new), F2s, '-x', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0 0.4470 0.7410])
    
    if t == 2
        axes(ax2)
    else
        axes(ax4)
    end
    hold on
    plot(0:length(logL_bound_new), Q1s, '-o', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0 0.4470 0.7410])
    plot(0:length(logL_bound_new), Q2s, '-x', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0 0.4470 0.7410])
    
end

% add benchmark lines
num_em = max([length(logL_bound),length(logL_bound_new)]);
axes(ax1)
plot(0:num_em, ones(1, num_em+1)*F1, '-ok', 'MarkerSize', 10, 'LineWidth', 1)
plot(0:num_em, ones(1, num_em+1)*F2, '-xk', 'MarkerSize', 10, 'LineWidth', 1)
ylim([0.49, 1.01])
xlabel('EM iterations')
ylabel('Estimated Parameters')
set(gca,'FontSize',16)
title('SSM Transition Matrix F', 'FontSize', 30)

axes(ax3)
plot(0:num_em, ones(1, num_em+1)*F1, '-ok', 'MarkerSize', 10, 'LineWidth', 1)
plot(0:num_em, ones(1, num_em+1)*F2, '-xk', 'MarkerSize', 10, 'LineWidth', 1)
ylim([0.49, 1.01])
xlabel('EM iterations')
ylabel('Estimated Parameters')
set(gca,'FontSize',16)
title('SSM Transition Matrix F', 'FontSize', 30)

axes(ax2)
plot(0:num_em, ones(1, num_em+1)*std1^2, '-ok', 'MarkerSize', 10, 'LineWidth', 1)
plot(0:num_em, ones(1, num_em+1)*std2^2, '-xk', 'MarkerSize', 10, 'LineWidth', 1)
ylim([0, 17])
xlabel('EM iterations')
ylabel('Estimated Parameters')
set(gca,'FontSize',16)
title('SSM State Noise Covariance Q', 'FontSize', 30)

axes(ax4)
plot(0:num_em, ones(1, num_em+1)*std1^2, '-ok', 'MarkerSize', 10, 'LineWidth', 1)
plot(0:num_em, ones(1, num_em+1)*std2^2, '-xk', 'MarkerSize', 10, 'LineWidth', 1)
ylim([0, 17])
xlabel('EM iterations')
ylabel('Estimated Parameters')
set(gca,'FontSize',16)
title('SSM State Noise Covariance Q', 'FontSize', 30)

%% Let's try to quantify parameter convergence across iterations

% run a fixed number of EM iterations
num_EM = 10;

% overwrite these tally variables
learnt_F1 = zeros(2,iter,nT_used,num_EM+1);
learnt_F2 = zeros(2,iter,nT_used,num_EM+1);
learnt_var1 = zeros(2,iter,nT_used,num_EM+1);
learnt_var2 = zeros(2,iter,nT_used,num_EM+1);
learnt_R = zeros(2,iter,nT_used,num_EM+1);
learnt_dwellp = zeros(2,iter,nT_used,num_EM+1);
logL_bounds = zeros(2,iter,nT_used,num_EM);

for n = 1:iter
    %%
    disp(['Iteration # ', num2str(n)])
    
    curr_dwell_prob = curr_dwell_list(n);
    
    for t = [2, 6]
        % Use a varying data length
        T_used = T_used_list(t);
        disp(['     T length: ', num2str(T_used)])
        
        % load stored ssm objects
        o1 = ssm_models{n,t}(1);
        o2 = ssm_models{n,t}(2);
        
        % Run VB learning - with M steps
        [~, ~, obj_array, ~, ~, logL_bound, em_store] = VBlearn_original([o1, o2], 'dwellp', curr_dwell_prob, 'verbose', false, 'ht_thresh', -inf, 'maxVB_iter', num_EM);
        % figure out the label of two models
        if obj_array(1).Q < obj_array(2).Q
            first_idx = 1;
            second_idx = 2;
        else
            first_idx = 2;
            second_idx = 1;
        end
        
        % Get evolution of parameters across EM iterations
        learnt_F1(1,n,t,:) = cellfun(@(x) x{1}(first_idx).F, em_store);
        learnt_F2(1,n,t,:) = cellfun(@(x) x{1}(second_idx).F, em_store);
        learnt_var1(1,n,t,:) = cellfun(@(x) x{1}(first_idx).Q, em_store);
        learnt_var2(1,n,t,:) = cellfun(@(x) x{1}(second_idx).Q, em_store);
        learnt_R(1,n,t,:) = cellfun(@(x) x{1}(first_idx).R, em_store);
        learnt_dwellp(1,n,t,:) = cellfun(@(x) trace(x{2})/2, em_store);
        logL_bounds(1,n,t,:) = logL_bound;
        
        [~, ~, obj_array, ~, ~, ~, ~, logL_bound_new, ~, ~, em_store] = VBlearn([o1, o2], 'dwellp', curr_dwell_prob, 'verbose', false, 'ht_thresh', -inf, 'maxVB_iter', num_EM);
        % figure out the label of two models
        if obj_array(1).Q < obj_array(2).Q
            first_idx = 1;
            second_idx = 2;
        else
            first_idx = 2;
            second_idx = 1;
        end
        
        % Get evolution of parameters across EM iterations
        learnt_F1(2,n,t,:) = cellfun(@(x) x{1}(first_idx).F, em_store);
        learnt_F2(2,n,t,:) = cellfun(@(x) x{1}(second_idx).F, em_store);
        learnt_var1(2,n,t,:) = cellfun(@(x) x{1}(first_idx).Q, em_store);
        learnt_var2(2,n,t,:) = cellfun(@(x) x{1}(second_idx).Q, em_store);
        learnt_R(2,n,t,:) = cellfun(@(x) x{1}(first_idx).R, em_store);
        learnt_dwellp(2,n,t,:) = cellfun(@(x) trace(x{2})/2, em_store);
        logL_bounds(2,n,t,:) = logL_bound_new;
        
    end
    
end

% save('Experiment2_review_parameter_convergence_results.mat')
load('Experiment2_review_parameter_convergence_results.mat')

%% Plot the mean parameter convergence curves at two different data lengths

figure

%F
subplot(2,2,1)
hold on

select_t = 2;

original_delta_F = abs(squeeze(learnt_F1(1,:,select_t,:)) - F1) / F1;
improved_delta_F = abs(squeeze(learnt_F1(2,:,select_t,:)) - F1) / F1;
align_first_point = original_delta_F(:,1);
improved_delta_F(:,1) = align_first_point;

shadedErrorBar(0:size(original_delta_F,2)-1, mean(original_delta_F), std(original_delta_F)./sqrt(size(original_delta_F, 1)), {'Color', [0.8500 0.3250 0.0980]}, 1);
plot(0:size(original_delta_F,2)-1, mean(original_delta_F), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.8500 0.3250 0.0980])

shadedErrorBar(0:size(improved_delta_F,2)-1, mean(improved_delta_F), std(improved_delta_F)./sqrt(size(improved_delta_F, 1)), {'Color', [0 0.4470 0.7410]}, 1);
plot(0:size(improved_delta_F,2)-1, mean(improved_delta_F), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0 0.4470 0.7410])

select_t = 6;

original_delta_F = abs(squeeze(learnt_F1(1,:,select_t,:)) - F1) / F1;
improved_delta_F = abs(squeeze(learnt_F1(2,:,select_t,:)) - F1) / F1;
original_delta_F(:,1) = align_first_point;
improved_delta_F(:,1) = align_first_point;

shadedErrorBar(0:size(original_delta_F,2)-1, mean(original_delta_F), std(original_delta_F)./sqrt(size(original_delta_F, 1)), {'Color', [0.6350 0.0780 0.1840]}, 1);
plot(0:size(original_delta_F,2)-1, mean(original_delta_F), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.6350 0.0780 0.1840])

shadedErrorBar(0:size(improved_delta_F,2)-1, mean(improved_delta_F), std(improved_delta_F)./sqrt(size(improved_delta_F, 1)), {'Color', [0 0.2470 0.5410]}, 1);
plot(0:size(improved_delta_F,2)-1, mean(improved_delta_F), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0 0.2470 0.5410])

% ylim([-0.01, 0.2])
ylim([-0.012, 0.25])
plot(xlim, [0,0], '--', 'Color', 'k', 'LineWidth', 1)

xlabel('EM Iteration')
ylabel('Normalized Error')
set(gca,'FontSize',16)
title('SSM Transition Matrix F', 'FontSize', 30)

%Q
subplot(2,2,2)
hold on

select_t = 2;

% original_delta_Q = abs([squeeze(learnt_var1(1,:,select_t,:)) - std1^2; squeeze(learnt_var2(1,:,select_t,:)) - std2^2]);
% improved_delta_Q = abs([squeeze(learnt_var1(2,:,select_t,:)) - std1^2; squeeze(learnt_var2(2,:,select_t,:)) - std2^2]);
original_delta_Q = abs([(squeeze(learnt_var1(1,:,select_t,:)) - std1^2) / std1^2; (squeeze(learnt_var2(1,:,select_t,:)) - std2^2) / std2^2]);
improved_delta_Q = abs([(squeeze(learnt_var1(2,:,select_t,:)) - std1^2) / std1^2; (squeeze(learnt_var2(2,:,select_t,:)) - std2^2) / std2^2]);
align_first_point = original_delta_Q(:,1);
improved_delta_Q(:,1) = align_first_point;

shadedErrorBar(0:size(original_delta_Q,2)-1, mean(original_delta_Q), std(original_delta_Q)./sqrt(size(original_delta_Q, 1)), {'Color', [0.8500 0.3250 0.0980]}, 1);
plot(0:size(original_delta_Q,2)-1, mean(original_delta_Q), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.8500 0.3250 0.0980])

shadedErrorBar(0:size(improved_delta_Q,2)-1, mean(improved_delta_Q), std(improved_delta_Q)./sqrt(size(improved_delta_Q, 1)), {'Color', [0 0.4470 0.7410]}, 1);
plot(0:size(improved_delta_Q,2)-1, mean(improved_delta_Q), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0 0.4470 0.7410])

select_t = 6;

original_delta_Q = abs([(squeeze(learnt_var1(1,:,select_t,:)) - std1^2) / std1^2; (squeeze(learnt_var2(1,:,select_t,:)) - std2^2) / std2^2]);
improved_delta_Q = abs([(squeeze(learnt_var1(2,:,select_t,:)) - std1^2) / std1^2; (squeeze(learnt_var2(2,:,select_t,:)) - std2^2) / std2^2]);
original_delta_Q(:,1) = align_first_point;
improved_delta_Q(:,1) = align_first_point;

shadedErrorBar(0:size(original_delta_Q,2)-1, mean(original_delta_Q), std(original_delta_Q)./sqrt(size(original_delta_Q, 1)), {'Color', [0.6350 0.0780 0.1840]}, 1);
plot(0:size(original_delta_Q,2)-1, mean(original_delta_Q), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.6350 0.0780 0.1840])

shadedErrorBar(0:size(improved_delta_Q,2)-1, mean(improved_delta_Q), std(improved_delta_Q)./sqrt(size(improved_delta_Q, 1)), {'Color', [0 0.2470 0.5410]}, 1);
plot(0:size(improved_delta_Q,2)-1, mean(improved_delta_Q), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0 0.2470 0.5410])

% ylim([-0.19, 3.6])
ylim([-0.065, 1.3])
plot(xlim, [0,0], '--', 'Color', 'k', 'LineWidth', 1)

xlabel('EM Iteration')
ylabel('Normalized Error')
% ylabel('Absolute Deviation from True Value')
set(gca,'FontSize',16)
title('SSM State Noise Covariance Q', 'FontSize', 30)

%R
subplot(2,2,3)
hold on

select_t = 2;

original_delta_R = abs(squeeze(learnt_R(1,:,select_t,:)) - R);
improved_delta_R = abs(squeeze(learnt_R(2,:,select_t,:)) - R);

shadedErrorBar(0:size(original_delta_R,2)-1, mean(original_delta_R), std(original_delta_R)./sqrt(size(original_delta_R, 1)), {'Color', [0.8500 0.3250 0.0980]}, 1);
l1 = plot(0:size(original_delta_R,2)-1, mean(original_delta_R), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.8500 0.3250 0.0980]);

shadedErrorBar(0:size(improved_delta_R,2)-1, mean(improved_delta_R), std(improved_delta_R)./sqrt(size(improved_delta_R, 1)), {'Color', [0 0.4470 0.7410]}, 1);
l2 = plot(0:size(improved_delta_R,2)-1, mean(improved_delta_R), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0 0.4470 0.7410]);

select_t = 6;

original_delta_R = abs(squeeze(learnt_R(1,:,select_t,:)) - R);
improved_delta_R = abs(squeeze(learnt_R(2,:,select_t,:)) - R);

shadedErrorBar(0:size(original_delta_R,2)-1, mean(original_delta_R), std(original_delta_R)./sqrt(size(original_delta_R, 1)), {'Color', [0.6350 0.0780 0.1840]}, 1);
l3 = plot(0:size(original_delta_R,2)-1, mean(original_delta_R), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.6350 0.0780 0.1840]);

shadedErrorBar(0:size(improved_delta_R,2)-1, mean(improved_delta_R), std(improved_delta_R)./sqrt(size(improved_delta_R, 1)), {'Color', [0 0.2470 0.5410]}, 1);
l4 = plot(0:size(improved_delta_R,2)-1, mean(improved_delta_R), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0 0.2470 0.5410]);

ylim([-0.004, 0.08])
plot(xlim, [0,0], '--', 'Color', 'k', 'LineWidth', 1)

legend([l1, l3, l2, l4], {'VI-A EM (T=200)', 'VI-A EM (T=3200)', 'VI-I EM (T=200)', 'VI-I EM (T=3200)'}, 'Location', 'best')

xlabel('EM Iteration')
ylabel('Absolute Deviation from True Value')
set(gca,'FontSize',16)
title('SSM Observation Noise Covariance R', 'FontSize', 30)

%dwell p
subplot(2,2,4)
hold on

select_t = 2;

original_delta_dp = abs(squeeze(learnt_dwellp(1,:,select_t,:)) - dwell_prob);
improved_delta_dp = abs(squeeze(learnt_dwellp(2,:,select_t,:)) - dwell_prob);

shadedErrorBar(0:size(original_delta_dp,2)-1, mean(original_delta_dp), std(original_delta_dp)./sqrt(size(original_delta_dp, 1)), {'Color', [0.8500 0.3250 0.0980]}, 1);
plot(0:size(original_delta_dp,2)-1, mean(original_delta_dp), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.8500 0.3250 0.0980])

shadedErrorBar(0:size(improved_delta_dp,2)-1, mean(improved_delta_dp), std(improved_delta_dp)./sqrt(size(improved_delta_dp, 1)), {'Color', [0 0.4470 0.7410]}, 1);
plot(0:size(improved_delta_dp,2)-1, mean(improved_delta_dp), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0 0.4470 0.7410])

select_t = 6;

original_delta_dp = abs(squeeze(learnt_dwellp(1,:,select_t,:)) - dwell_prob);
improved_delta_dp = abs(squeeze(learnt_dwellp(2,:,select_t,:)) - dwell_prob);

shadedErrorBar(0:size(original_delta_dp,2)-1, mean(original_delta_dp), std(original_delta_dp)./sqrt(size(original_delta_dp, 1)), {'Color', [0.6350 0.0780 0.1840]}, 1);
plot(0:size(original_delta_dp,2)-1, mean(original_delta_dp), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0.6350 0.0780 0.1840])

shadedErrorBar(0:size(improved_delta_dp,2)-1, mean(improved_delta_dp), std(improved_delta_dp)./sqrt(size(improved_delta_dp, 1)), {'Color', [0 0.2470 0.5410]}, 1);
plot(0:size(improved_delta_dp,2)-1, mean(improved_delta_dp), '.-', 'LineWidth', 2, 'MarkerSize', 40, 'Color', [0 0.2470 0.5410])

ylim([-0.008, 0.155])
plot(xlim, [0,0], '--', 'Color', 'k', 'LineWidth', 1)

xlabel('EM Iteration')
ylabel('Absolute Deviation from True Value')
set(gca,'FontSize',16)
title('HMM Transition Stay Probability P(Stay)', 'FontSize', 30)

%% Explain why is the observation noise R insensitive to updates
% this is due to relatively flat likelihood surface in the small range we
% are sampling from

% generate two AR(1) SSMs and a discrete state HMM for switching
x1 = [randn(1)*std1, zeros(1,T-1)];
x2 = [randn(1)*std2, zeros(1,T-1)];
s = [datasample([1,-1],1), zeros(1,T-1)];
for ii = 2:T
    x1(ii) = F1*x1(ii-1) + randn(1)*std1;
    x2(ii) = F2*x2(ii-1) + randn(1)*std2;
    if rand(1) >= dwell_prob
        s(ii) = s(ii-1) * -1; % switch
    else
        s(ii) = s(ii-1); % stay
    end
end
s(s==-1) = 0;

% generate observed data
y = x1.*(s==1) + x2.*(s==0) + randn(1,T).*sqrt(R);

t = 2;

% Use a varying data length
T_used = T_used_list(t);
y_used = y(1:T_used);
disp(['     T length: ', num2str(T_used)])

R_list = linspace(0.01,0.2,100);
logL_bounds = zeros(2,length(R_list));

for ii = 1:length(R_list)
    curr_R = R_list(ii);
    disp(['Current R = ', num2str(curr_R)])
    
    % use true parameters except R
    o1 = ssm({arn(1)},F1,std1^2,0,std1^2,1,curr_R,y_used);
    o2 = ssm({arn(1)},F2,std2^2,0,std2^2,1,curr_R,y_used);
    
    [~, ~, ~, ~, ~, logL_bound] = VBlearn_original([o1, o2], 'dwellp', dwell_prob, 'verbose', false, 'maxVB_iter', 1);
    logL_bounds(1,ii) = logL_bound;
    
    [~, ~, ~, ~, ~, ~, ~, logL_bound_new] = VBlearn([o1, o2], 'dwellp', dwell_prob, 'verbose', false, 'maxVB_iter', 1);
    logL_bounds(2,ii) = logL_bound_new;
end

figure
hold on
l1 = plot(R_list, logL_bounds(2,:), 'LineWidth', 2);
l2 = plot(R_list, logL_bounds(1,:), 'LineWidth', 2);
plot([R, R], ylim, 'r--', 'LineWidth', 1)
legend([l2, l1], {'VI-A', 'VI-I'}, 'Location', 'best')
xlabel('Observation Noise Covariance R')
ylabel('Free energy')
set(gca,'FontSize',16)

%% In contrast, we could do the same for Q

% generate two AR(1) SSMs and a discrete state HMM for switching
x1 = [randn(1)*std1, zeros(1,T-1)];
x2 = [randn(1)*std2, zeros(1,T-1)];
s = [datasample([1,-1],1), zeros(1,T-1)];
for ii = 2:T
    x1(ii) = F1*x1(ii-1) + randn(1)*std1;
    x2(ii) = F2*x2(ii-1) + randn(1)*std2;
    if rand(1) >= dwell_prob
        s(ii) = s(ii-1) * -1; % switch
    else
        s(ii) = s(ii-1); % stay
    end
end
s(s==-1) = 0;

% generate observed data
y = x1.*(s==1) + x2.*(s==0) + randn(1,T).*sqrt(R);

t = 2;

% Use a varying data length
T_used = T_used_list(t);
y_used = y(1:T_used);
disp(['     T length: ', num2str(T_used)])

R_list = linspace(0.01,0.2,100);
Q_list = linspace(5,15,100);
logL_bounds = zeros(2,length(Q_list));

for ii = 1:length(Q_list)
    curr_Q2 = Q_list(ii);
    disp(['Current Q = ', num2str(curr_Q2)])
    
    % use true parameters except Q2
    o1 = ssm({arn(1)},F1,std1^2,0,std1^2,1,R,y_used);
    o2 = ssm({arn(1)},F2,curr_Q2,0,curr_Q2,1,R,y_used);
    
    [~, ~, ~, ~, ~, logL_bound] = VBlearn_original([o1, o2], 'dwellp', dwell_prob, 'verbose', false, 'maxVB_iter', 1);
    logL_bounds(1,ii) = logL_bound;
    
    [~, ~, ~, ~, ~, ~, ~, logL_bound_new] = VBlearn([o1, o2], 'dwellp', dwell_prob, 'verbose', false, 'maxVB_iter', 1);
    logL_bounds(2,ii) = logL_bound_new;
end

figure
hold on
l1 = plot(Q_list, logL_bounds(2,:), 'LineWidth', 2);
l2 = plot(Q_list, logL_bounds(1,:), 'LineWidth', 2);
plot([std2^2, std2^2], ylim, 'k', 'LineWidth', 1)
plot([5, 5], ylim, '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 1)
plot([15, 15], ylim, '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 1)
xlim([4.5, 15])
legend([l2, l1], {'VI-A', 'VI-I'}, 'Location', 'best')
xlabel('State Noise Variance Q')
ylabel('Free energy')
set(gca,'FontSize',16)
