 %% Experiment 3 - VB inference comparisons (no M steps)
% Extend simulations to bivariate observed data and a different switching
% mechanism: i.e., we have a single evolving state vector, but the SSM
% parameters (specifically, an entry of the transition matrix) switch.
%
% We will again directly compare improved VB learning algorithm with the
% original one and traditional switching methods for the inference part,
% i.e. E step, and do not involve any M steps. 
close all; clear all; clc

% simulation parameters
iter = 200;
T = 200;
F1 = [0.5, 0; 0, 0.5];  
F2 = [0.5, 0.5; 0, 0.5];
p = 2;
q = 2;
Q = eye(p) * 2;
G = eye(p);
R = eye(q) * 0.1; % observation noise covariance matrix
dwell_prob = 0.95; % HMM state dwell probability 

original_percent_correct = zeros(1,iter);
improved_percent_correct = zeros(1,iter);
random_percent_correct = zeros(1,iter);
static_percent_correct = zeros(1,iter);
gpb1_percent_correct = zeros(1,iter);
gpb2_percent_correct = zeros(1,iter);
IMM_percent_correct = zeros(1,iter);

% for n = 1:iter
%     disp(n)
%     
%     % generate a single SSM with two AR(1) and a discrete state HMM for
%     % switching 
%     x = [mvnrnd(zeros(1,p), Q, 1)', zeros(2,T-1)];
%     s = [datasample([1,-1],1), zeros(1,T-1)];
%     for ii = 2:T
%         if rand(1) >= dwell_prob
%             s(ii) = s(ii-1) * -1; % switch
%         else
%             s(ii) = s(ii-1); % stay
%         end
%         if s(ii) == -1
%             x(:,ii) = F1*x(:,ii-1) +  mvnrnd(zeros(1,p), Q, 1)';
%         else
%             x(:,ii) = F2*x(:,ii-1) +  mvnrnd(zeros(1,p), Q, 1)';
%         end
%     end
%     s(s==-1) = 0;
%     
%     % generate observed data
%     y = G * x + mvnrnd(zeros(1,q), R, T)';
%     
%     % Inference is done in SOMATA since MATLAB doesn't support multivariate
% end

load('Experiment3_simulation_results.mat')

%% Plot the segmentation results 
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
title(['Original VB inference with annealing: mean = ', sprintf('%.3f', mean(original_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(original_percent_correct), mean(original_percent_correct)], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,5)
histogram(improved_percent_correct,0:0.02:1, 'FaceColor', [0 0.4470 0.7410])
title(['Improved VB inference: mean = ', sprintf('%.3f', mean(improved_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(improved_percent_correct), mean(improved_percent_correct)], ylim, 'r--', 'LineWidth', 1)

xlabel('Percent Correct Segmentation')

%% all segmentation methods using traditional switching methods 
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
histogram(gpb1_percent_correct,0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['GPB1 Switching Method: mean = ', sprintf('%.3f', mean(gpb1_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(gpb1_percent_correct), mean(gpb1_percent_correct)], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,4)
histogram(gpb2_percent_correct,0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['GPB2 Switching Method: mean = ', sprintf('%.3f', mean(gpb2_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(gpb2_percent_correct), mean(gpb2_percent_correct)], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,5)
histogram(IMM_percent_correct,0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['IMM Switching Method: mean = ', sprintf('%.3f', mean(IMM_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(IMM_percent_correct), mean(IMM_percent_correct)], ylim, 'r--', 'LineWidth', 1)

xlabel('Percent Correct Segmentation')

%% Plot an example time series 
load('Experiment3_example_simulation.mat')

% % generate a single SSM with two AR(1) and a discrete state HMM for
% % switching
% x = [mvnrnd(zeros(1,p), Q, 1)', zeros(2,T-1)];
% s = [datasample([1,-1],1), zeros(1,T-1)];
% for ii = 2:T
%     if rand(1) >= dwell_prob
%         s(ii) = s(ii-1) * -1; % switch
%     else
%         s(ii) = s(ii-1); % stay
%     end
%     if s(ii) == -1
%         x(:,ii) = F1*x(:,ii-1) +  mvnrnd(zeros(1,p), Q, 1)';
%     else
%         x(:,ii) = F2*x(:,ii-1) +  mvnrnd(zeros(1,p), Q, 1)';
%     end
% end
% s(s==-1) = 0;
% 
% % generate observed data
% y = G * x + mvnrnd(zeros(1,q), R, T)';

t = 1:T;

s_x = 1:T;
s_x(s==0) = []; 

s_copy = s; 
s_copy(s==0) = []; 

ori_x = 1:T;
ori_x(anneal(2,:)==0) = []; ori = anneal(2,anneal(2,:)==1);

imp_x = 1:T;
imp_x(interp(2,:)==0) = []; imp = interp(2,interp(2,:)==1);

ran_x = 1:T;
ran_x(rand(1,:)==0) = []; ran = rand(1,rand(1,:)==1);

sta_x = 1:T;
sta_x(static(2,:)==0) = []; sta = static(2,static(2,:)==1);

imm_x = 1:T;
imm_x(imm(2,:)==0) = []; imm_1 = imm(2,imm(2,:)==1);


figure
subplot(3,1,1)
hold on
m1 = plot(t, y(1,:), 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2);
[cons, inds] = consecutive(s);
for ii = 1:length(inds)
    m2 = plot(t(inds{ii}), y(1, inds{ii}), 'm-', 'LineWidth', 2);
end
ylim([-10,10])
ylabel('Simulated Signal')
xlabel('Samples')
legend([m2, m1], {'S_t=1', 'S_t=2'})
title('Example 1 Simulated Time Series')
set(gca,'FontSize',20)

subplot(3,1,2)
hold on
m1 = plot(t, y(2,:), 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2);
[cons, inds] = consecutive(s);
for ii = 1:length(inds)
    m2 = plot(t(inds{ii}), y(2, inds{ii}), 'm-', 'LineWidth', 2);
end
ylim([-10,10])
ylabel('Simulated Signal')
xlabel('Samples')
set(gca,'FontSize',20)

subplot(3,1,3)
hold on
scatter(s_x, s_copy.*5, 5, 'filled', 'MarkerEdgeColor', 'm', 'MarkerFaceColor',[0 0 0])
scatter(ran_x, ran.*4, 5, 'filled', 'MarkerEdgeColor',[0 0 0], 'MarkerFaceColor',[0 0 0])
scatter(sta_x, sta.*3, 5, 'filled', 'MarkerEdgeColor',[0 0 0], 'MarkerFaceColor',[0 0 0])
scatter(imm_x, imm_1.*2, 5, 'filled', 'MarkerEdgeColor',[0 0 0], 'MarkerFaceColor',[0 0 0])
scatter(ori_x, ori.*1, 5, 'filled', 'MarkerEdgeColor',[0.8500 0.3250 0.0980], 'MarkerFaceColor',[0.8500 0.3250 0.0980])
scatter(imp_x, imp.*0, 5, 'filled', 'MarkerEdgeColor',[0 0.4470 0.7410], 'MarkerFaceColor',[0 0.4470 0.7410])
ylim([-1,6])
yticks([0:5])
yticklabels({['Interp(', sprintf('%.2f', interp_acc), ')'],...
             ['Anneal(', sprintf('%.2f', anneal_acc), ')'],...
             ['IMM(', sprintf('%.2f', imm_acc), ')'],...
             ['Static(', sprintf('%.2f', static_acc), ')'],...
             ['Random(', sprintf('%.2f', rand_acc), ')'],...
             'True'})
xlabel('Samples')
title('Segmentation Results')
set(gca,'FontSize',20)
