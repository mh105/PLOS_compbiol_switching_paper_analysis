%% Experiment 4 - VB learning comparisons >>> NEW UPDATES FOR REVIEWERS
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
learnt_F = zeros(2, iter, size(F1,1), size(F1,2));
learnt_var = zeros(2, iter, size(Q,1), size(Q,2));
learnt_R = zeros(2, iter, size(R,1), size(R,2));
learnt_dwellp = zeros(2, iter);

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

load('Experiment4_review_simulation_results.mat')

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

%% Visualize learned parameters 
figure

subplot(2,2,1)
hold on
original_F = [learnt_F(1,:,1,1), learnt_F(1,:,1,2), learnt_F(1,:,2,2)];
improved_F = [learnt_F(2,:,1,1), learnt_F(2,:,1,2), learnt_F(2,:,2,2)];
swarmchart(ones(size(original_F)), original_F, [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(size(improved_F))*2, improved_F, [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5,2.5])
ylim([0.1, 0.8])
plot(xlim, [0.4, 0.4], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, [0.6, 0.6], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, ones(1,2)*F1(1,1), '-', 'Color', 'k', 'LineWidth', 3)
xticks([1:2])
xticklabels({'Ori F elements', 'Imp F elements'})
xlabel('\bf F \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Estimated Parameters')
set(gca,'FontSize',16)
title('SSM Transition Matrix F', 'FontSize', 30)

subplot(2,2,2)
hold on
original_Q = [learnt_var(1,:,1,1), learnt_var(1,:,2,2)];
improved_Q = [learnt_var(2,:,1,1), learnt_var(2,:,2,2)];
swarmchart(ones(size(original_Q)), original_Q, [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(size(improved_Q))*2, improved_Q, [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5,2.5])
ylim([0.8, 3.2])
plot(xlim, [1, 1], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, [3, 3], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, ones(1,2)*Q(1,1), '-', 'Color', 'k', 'LineWidth', 3)
xticks([1:2])
xticklabels({'Ori Q elements', 'Imp Q elements'})
xlabel('\bf Q \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Estimated Parameters')
set(gca,'FontSize',16)
title('SSM State Noise Covariance Q', 'FontSize', 30)

subplot(2,2,3)
hold on
original_R = [learnt_R(1,:,1,1), learnt_R(1,:,2,2)];
improved_R = [learnt_R(2,:,1,1), learnt_R(2,:,2,2)];
swarmchart(ones(size(original_R)), original_R, [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(size(improved_R))*2, improved_R, [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5,2.5])
ylim([-0.1, 0.4])
plot(xlim, [0.01, 0.01], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, [0.2, 0.2], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, ones(1,2)*R(1,1), '-', 'Color', 'k', 'LineWidth', 3)
xticks([1:2])
xticklabels({'Ori R elements', 'Imp R elements'})
xlabel('\bf R \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Estimated Parameters')
set(gca,'FontSize',16)
title('SSM Observation Noise Covariance R', 'FontSize', 30)

subplot(2,2,4)
hold on
swarmchart(ones(1, size(learnt_dwellp,2)), learnt_dwellp(1,:), [], [0.8500 0.3250 0.0980], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
swarmchart(ones(1, size(learnt_dwellp,2))*2, learnt_dwellp(2,:), [], [0 0.4470 0.7410], 'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5)
xlim([0.5,2.5])
ylim([0.8, 1])
plot(xlim, [0.9, 0.9], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, [0.99, 0.99], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 2)
plot(xlim, ones(1,2)*dwell_prob, '-', 'Color', 'k', 'LineWidth', 3)
xticks([1:2]) %#ok<*NBRAK>
xticklabels({'Ori P(Stay)', 'Imp P(Stay)'})
xlabel('\bf P(Stay) \rm Parameters Estimated by VB Learning Algorithms')
ylabel('Estimated P(Stay)')
set(gca,'FontSize',16)
title('HMM Transition Stay Probability P(Stay)', 'FontSize', 30)

%% Compare with Scott Linderman's SVI algorithms
load('Experiment4_review_svi_lem_results.mat')

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
histogram(svi_mf_percent_correct,0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['BBVI Mean-Field: mean = ', sprintf('%.3f', mean(svi_mf_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(svi_mf_percent_correct), mean(svi_mf_percent_correct)], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,3)
histogram(svi_struct_percent_correct,0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['BBVI Tridiagonal: mean = ', sprintf('%.3f', mean(svi_struct_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(svi_struct_percent_correct), mean(svi_struct_percent_correct)], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,4)
histogram(lem_percent_correct,0:0.02:1, 'FaceColor', [0.5 0.5 0.5])
title(['Laplace-EM Structured: mean = ', sprintf('%.3f', mean(lem_percent_correct))])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(lem_percent_correct), mean(lem_percent_correct)], ylim, 'r--', 'LineWidth', 1)

subplot(5,1,5)
histogram(improved_percent_correct,0:0.02:1, 'FaceColor', [0 0.4470 0.7410])
title(['Improved VB inference: mean = ', sprintf('%.3f', mean(improved_percent_correct)+0.001)])
xticks([0:0.1:1])
xticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
xlim([0,1])
ylim([0,50])
set(gca,'FontSize', 20)
hold on
plot([mean(improved_percent_correct), mean(improved_percent_correct)], ylim, 'r--', 'LineWidth', 1)

xlabel('Percent Correct Segmentation')
