%% Experiment 4 - VB learning comparisons
% Extend simulations to bi-variate observed data and a different switching
% mechanism: i.e., we have a single evolving state vector, but the SSM
% parameters (specifically, an entry of the transition matrix) switch.
% 
% We will again directly compare improved VB learning algorithm with the
% original one and traditional switching methods, including learning.
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

load('Experiment4_simulation_results.mat')

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
