%% Experiment 2 - VB learning comparisons
% Simulations to compare the improved VB learning algorithm with the
% original one on learning segmental SSM parameters. Performance is
% evaluated at multiple levels:
% - correct segmentation
% - parameter estimation
% - free energy bounds on log likelihood (observed data fitting)
close all; clear all; clc

addpath(genpath('state_space_spindle_detector_code'))

%% Run simulations 
% true and fixed simulation parameters
iter = 200;
T = 200;
F1 = 0.9; std1 = sqrt(2);
F2 = 0.7;  std2 = sqrt(10);
R = 0.1; % observation noise covariance
dwell_prob = 0.95; % HMM state dwell probability

original_percent_correct = zeros(1,iter);
improved_percent_correct = zeros(1,iter);
random_percent_correct = zeros(1,iter);
static_percent_correct = zeros(1,iter);
IMM_percent_correct = zeros(1,iter);
learnt_F1 = zeros(2,iter);
learnt_F2 = zeros(2,iter);
learnt_var1 = zeros(2,iter);
learnt_var2 = zeros(2,iter);
learnt_R = zeros(2,iter);
learnt_dwellp = zeros(2,iter);
logL_bounds = zeros(2,iter);

ssm_models = {};
curr_dwell_list = zeros(1,iter);
s_list = zeros(T,iter);

for n = 1:iter
    %%
    disp(n)
    
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
    curr_F1 = rand(1)*0.2+0.8; % 0.8-1.0
    curr_F2 = rand(1)*0.2+0.6; % 0.6-0.8
    curr_std1 = sqrt(rand(1)*2+1); % 1-3
    curr_std2 = sqrt(rand(1)*10+5); % 5-15
    curr_R = rand(1)*0.19+0.01; % 0.01-0.2
    curr_dwell_prob = rand(1)*0.09+0.9; % 0.9-0.99
    
    % create ssm objects
    o1 = ssm({arn(1)},curr_F1,curr_std1^2,0,curr_std1^2,1,curr_R,y);
    o2 = ssm({arn(1)},curr_F2,curr_std2^2,0,curr_std2^2,1,curr_R,y);
    
    % store the ssm objects for later processing and plotting 
    ssm_models{n} = [o1, o2];
    curr_dwell_list(n) = curr_dwell_prob;
    s_list(:,n) = s;
    
    % Run VB learning - with M steps
    [Mprob, ~, obj_array, A, ~, logL_bound] = VBlearn_original([o1, o2], 'dwellp', curr_dwell_prob);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    original_percent_correct(n) = mean(abs(Mprob(1,:) - s) <= 0.1);
    
    learnt_F1(1,n) = obj_array(1).F;
    learnt_F2(1,n) = obj_array(2).F;
    learnt_var1(1,n) = obj_array(1).Q;
    learnt_var2(1,n) = obj_array(2).Q;
    learnt_R(1,n) = obj_array(1).R;
    learnt_dwellp(1,n) = trace(A)/2;
    logL_bounds(1,n) = logL_bound(end);
    
    [Mprob, ~, obj_array, A, ~, ~, ~, logL_bound_new] = VBlearn([o1, o2], 'dwellp', curr_dwell_prob);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    improved_percent_correct(n) = mean(abs(Mprob(1,:) - s) <= 0.1);
    
    learnt_F1(2,n) = obj_array(1).F;
    learnt_F2(2,n) = obj_array(2).F;
    learnt_var1(2,n) = obj_array(1).Q;
    learnt_var2(2,n) = obj_array(2).Q;
    learnt_R(2,n) = obj_array(1).R;
    learnt_dwellp(2,n) = trace(A)/2;
    logL_bounds(2,n) = logL_bound_new(end);
    
    %%
    % Run comparison inference algorithms
    Mprob = rand(1,T); % random segmentation
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    random_percent_correct(n) = mean(abs(Mprob - s) <= 0.1);
    
    Mprob = switching([o1, o2],'method','static','dwellp',dwell_prob);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    static_percent_correct(n) = mean(abs(Mprob(1,:) - s) <= 0.1);
    
    Mprob = switching([o1, o2],'method','IMM','dwellp',dwell_prob);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    IMM_percent_correct(n) = mean(abs(Mprob(1,:) - s) <= 0.1);
end

%% Save simulation results 
% save('Experiment2_simulation_results')

%% Robustness characterization 1: segmentation accuracy 
figure
subplot(5,1,1)
histogram(random_percent_correct,0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['Random Segmentation: mean = ', sprintf('%.3f', mean(random_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(random_percent_correct), mean(random_percent_correct)], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,2)
histogram(static_percent_correct,0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['Static Switching Method: mean = ', sprintf('%.3f', mean(static_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(static_percent_correct), mean(static_percent_correct)], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,3)
histogram(IMM_percent_correct,0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['IMM Switching Method: mean = ', sprintf('%.3f', mean(IMM_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(IMM_percent_correct), mean(IMM_percent_correct)], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,4)
histogram(original_percent_correct,0:0.02:1, 'FaceColor', [0.8500 0.3250 0.0980])
title(['Original VB learning with annealing: mean = ', sprintf('%.3f', mean(original_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(original_percent_correct), mean(original_percent_correct)], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,5)
histogram(improved_percent_correct,0:0.02:1, 'FaceColor', [0 0.4470 0.7410])
title(['Improved VB learning: mean = ', sprintf('%.3f', mean(improved_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(improved_percent_correct), mean(improved_percent_correct)], ylim, 'r--', 'LineWidth', 1)

xlabel('Percent Correct Segmentation')

%% Robustness characterization 2: parameter estimation 
figure
subplot(2,2,1)
hold on
swarmchart(ones(size(learnt_F1,2)), (learnt_F1(1,:) - F1)/F1, [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(size(learnt_F1,2))*2, (learnt_F1(2,:) - F1)/F1, [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(size(learnt_F2,2))*3.5, (learnt_F2(1,:) - F2)/F2, [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(size(learnt_F2,2))*4.5, (learnt_F2(2,:) - F2)/F2, [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5,5])
plot([0.5,2.5], ones(1,2)*(0.1/F1), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([0.5,2.5], ones(1,2)*(-0.1/F1), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([3,5], ones(1,2)*(0.1/F2), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([3,5], ones(1,2)*(-0.1/F2), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, [0,0], '-', 'Color', [0.5,0.5,0.5], 'LineWidth', 1)
xticks([1,2,3.5,4.5])
xticklabels({'Ori F1', 'Imp F1', 'Ori F2', 'Imp F2'})
xlabel('\bf F \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Normalized Error against True Parameter')
set(gca,'FontSize',16)
title('SSM Transition Matrix F', 'FontSize', 30)

subplot(2,2,2)
hold on
swarmchart(ones(size(learnt_var1,2)), (learnt_var1(1,:) - std1^2)/std1^2, [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(size(learnt_var1,2))*2, (learnt_var1(2,:) - std1^2)/std1^2, [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(size(learnt_var2,2))*3.5, (learnt_var2(1,:) - std2^2)/std2^2, [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(size(learnt_var2,2))*4.5, (learnt_var2(2,:) - std2^2)/std2^2, [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5,5])
plot([0.5,2.5], ones(1,2)*(1/std1^2), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([0.5,2.5], ones(1,2)*(-1/std1^2), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([3,5], ones(1,2)*(5/std2^2), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([3,5], ones(1,2)*(-5/std2^2), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, [0,0], '-', 'Color', [0.5,0.5,0.5], 'LineWidth', 1)
xticks([1,2,3.5,4.5])
xticklabels({'Ori Q1', 'Imp Q1', 'Ori Q2', 'Imp Q2'})
xlabel('\bf Q \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Normalized Error against True Parameter')
set(gca,'FontSize',16)
title('SSM State Noise Covariance Q', 'FontSize', 30)

subplot(2,2,3)
hold on
swarmchart(ones(size(learnt_R,2)), learnt_R(1,:), [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(size(learnt_R,2))*2, learnt_R(2,:), [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5,2.5])
ylim([-0.1, 0.4])
plot([0.5,1.5], ones(1,2)*(0.2), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([0.5,1.5], ones(1,2)*(0.01), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([1.5,2.5], ones(1,2)*(0.2), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([1.5,2.5], ones(1,2)*(0.01), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, [0.1,0.1], '-', 'Color', [0.5,0.5,0.5], 'LineWidth', 1)
xticks([1:2])
xticklabels({'Ori R', 'Imp R'})
xlabel('\bf R \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Estimated R')
set(gca,'FontSize',16)
title('SSM Observation Noise Covariance R', 'FontSize', 30)

subplot(2,2,4)
hold on
swarmchart(ones(size(learnt_dwellp,2)), learnt_dwellp(1,:), [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(size(learnt_dwellp,2))*2, learnt_dwellp(2,:), [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5,2.5])
plot([0.5,1.5], ones(1,2)*(0.99), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([0.5,1.5], ones(1,2)*(0.9), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([1.5,2.5], ones(1,2)*(0.99), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot([1.5,2.5], ones(1,2)*(0.9), '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, [0.95,0.95], '-', 'Color', [0.5,0.5,0.5], 'LineWidth', 1)
xticks([1:2])
ylim([0.8, 1])
xticklabels({'Ori P(Stay)', 'Imp P(Stay)'})
xlabel('\bf P(Stay) \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Estimated P(Stay)')
set(gca,'FontSize',16)
title('HMM Transition Stay Probability P(Stay)', 'FontSize', 30)

%% Robustness characterization 3: loglikelihood bounds (Free energy)
% Visualize the distributions of converged B
figure
histogram(logL_bounds(1,:) - logL_bounds(2,:), 25, 'FaceColor', [0.6350 0.0780 0.1840])
hold on
plot([0,0], ylim, 'k--', 'LineWidth', 2)
title('Differences in Converged Free Energy')
ylabel('Counts')
xlabel('B_O_r_i - B_I_m_p')
set(gca,'FontSize', 20)

% test for significant difference using paired-sample t-test
[h,p] = ttest(logL_bounds(1,:), logL_bounds(2,:));
dummy = plot([0,0],[0,0], 'w');
legend(dummy, 'p < 10^{-9}', 'Location', 'northeast')
legend boxoff

%% Demonstrate why Free Energy bounds alone are insufficient 
curr = ssm_models{40};
o1 = curr(1); o2 = curr(2);
s = s_list(:,40)';
curr_dwell_prob = curr_dwell_list(40);

% run EM for 300 iterations
[Mprob, ~, obj_array, A, ~, logL_bound] = VBlearn_original([o1, o2], 'dwellp', curr_dwell_prob, 'ht_thresh', -inf, 'maxVB_iter', 300);
[Mprob_new, ~, obj_array_new, A_new, ~, ~, ~, logL_bound_new] = VBlearn([o1, o2], 'dwellp', curr_dwell_prob, 'ht_thresh', -inf, 'maxVB_iter', 300);
Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0; mean(abs(Mprob(1,:) - s) <= 0.1);
Mprob_new(Mprob_new>=0.5)=1; Mprob_new(Mprob_new<0.5)=0; mean(abs(Mprob_new(1,:) - s) <= 0.1);

% visualize the segmentation
s_x = 1:T;
s_x(s==0) = []; s_copy = s; s(s==0) = [];

ori_x = 1:T;
ori_x(Mprob(1,:)==0) = []; ori = Mprob(1,Mprob(1,:)==1);

imp_x = 1:T;
imp_x(Mprob_new(1,:)==0) = []; imp = Mprob_new(1,Mprob_new(1,:)==1);

figure
subplot(3,1,1)
hold on
t = 1:T;
plot(t, o1.y, 'b-', 'LineWidth', 2)
[cons, inds] = consecutive(s_copy);
for ii = 1:length(inds)
    plot(t(inds{ii}), o1.y(inds{ii}), 'm-', 'LineWidth', 2)
end
ylim([-15,15])
ylabel('Simulated Signal')
% xlabel('Samples')
title('Example 1 Simulated Time Series')
set(gca,'FontSize',20)

subplot(3,1,2)
hold on
scatter(s_x, s.*2, 5, 'filled', 'MarkerEdgeColor','m', 'MarkerFaceColor','m')
scatter(ori_x, ori.*1, 5, 'filled', 'MarkerEdgeColor',[0.8500 0.3250 0.0980], 'MarkerFaceColor',[0.8500 0.3250 0.0980])
scatter(imp_x, imp.*0, 5, 'filled', 'MarkerEdgeColor',[0 0.4470 0.7410], 'MarkerFaceColor',[0 0.4470 0.7410])
ylim([-1,3])
yticks([0:2])
yticklabels({'VI(0.94)', 'VA(0.50)', 'True'})
xlabel('Samples')
title('Segmentation Results')
set(gca,'FontSize',20)

subplot(3,1,3)
hold on
plot(logL_bound, 'LineWidth',2, 'Color', [0.8500 0.3250 0.0980])
plot(logL_bound_new, 'LineWidth',2, 'Color', [0 0.4470 0.7410])
xlim([0,300])
title('Negative variational free energy')
ylabel('Free Energy')
xlabel('EM Iterations')
legend('VA', 'VI', 'Location', 'south')
set(gca,'FontSize',20)

%% Free Energy bounds can still demonstrate the robustness
curr = ssm_models{2};
o1 = curr(1); o2 = curr(2);
s = s_list(:,2)';
curr_dwell_prob = curr_dwell_list(2);

% run EM for 300 iterations
[Mprob, ~, obj_array, A, ~, logL_bound] = VBlearn_original([o1, o2], 'dwellp', curr_dwell_prob, 'ht_thresh', -inf, 'maxVB_iter', 175);
[Mprob_new, ~, obj_array_new, A_new, ~, ~, ~, logL_bound_new] = VBlearn([o1, o2], 'dwellp', curr_dwell_prob, 'ht_thresh', -inf, 'maxVB_iter', 175);
Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0; mean(abs(Mprob(1,:) - s) <= 0.1);
Mprob_new(Mprob_new>=0.5)=1; Mprob_new(Mprob_new<0.5)=0; mean(abs(Mprob_new(1,:) - s) <= 0.1);

% visualize the segmentation
s_x = 1:T;
s_x(s==0) = []; s_copy = s; s(s==0) = []; 

ori_x = 1:T;
ori_x(Mprob(1,:)==0) = []; ori = Mprob(1,Mprob(1,:)==1);

imp_x = 1:T;
imp_x(Mprob_new(1,:)==0) = []; imp = Mprob_new(1,Mprob_new(1,:)==1);

figure
subplot(3,1,1)
hold on
t = 1:T;
plot(t, o1.y, 'b-', 'LineWidth', 2)
[cons, inds] = consecutive(s_copy);
for ii = 1:length(inds)
    plot(t(inds{ii}), o1.y(inds{ii}), 'm-', 'LineWidth', 2)
end
ylim([-15,15])
ylabel('Simulated Signal')
% xlabel('Samples')
title('Example 2 Simulated Time Series')
set(gca,'FontSize',20)

subplot(3,1,2)
hold on
scatter(s_x, s.*2, 5, 'filled', 'MarkerEdgeColor','m', 'MarkerFaceColor','m')
scatter(ori_x, ori.*1, 5, 'filled', 'MarkerEdgeColor',[0.8500 0.3250 0.0980], 'MarkerFaceColor',[0.8500 0.3250 0.0980])
scatter(imp_x, imp.*0, 5, 'filled', 'MarkerEdgeColor',[0 0.4470 0.7410], 'MarkerFaceColor',[0 0.4470 0.7410])
ylim([-1,3])
yticks([0:2])
yticklabels({'VI(0.95)', 'VA(0.89)', 'True'})
xlabel('Samples')
title('Segmentation Results')
set(gca,'FontSize',20)

subplot(3,1,3)
hold on
plot(logL_bound, 'LineWidth',2, 'Color', [0.8500 0.3250 0.0980])
plot(logL_bound_new, 'LineWidth',2, 'Color', [0 0.4470 0.7410])
xlim([0,300])
title('Negative variational free energy')
ylabel('Free Energy')
xlabel('EM Iterations')
legend('VA', 'VI', 'Location', 'south')
set(gca,'FontSize',20)
