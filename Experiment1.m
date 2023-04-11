%% Experiment 1 - VB inference comparisons (no M steps)
% Simulations to compare the improved VB learning algorithm with the
% original one and traditional switching methods for the inference part,
% i.e. E step, and do not involve any M steps. This follows Experiment 1 in
% Ghahramani & Hinton 2000.
close all; clear all; clc

addpath(genpath('state_space_spindle_detector_code'))

% simulation parameters
iter = 200;
T = 200;
F1 = 0.99; std1 = sqrt(1);
F2 = 0.9;  std2 = sqrt(10);
R = 0.1; % observation noise covariance
dwell_prob = 0.95; % HMM state dwell probability 

original_percent_correct = zeros(1,iter);
improved_percent_correct = zeros(1,iter);
random_percent_correct = zeros(1,iter);
static_percent_correct = zeros(1,iter);
gpb1_percent_correct = zeros(1,iter);
gpb2_percent_correct = zeros(1,iter);
IMM_percent_correct = zeros(1,iter);

for n = 1:iter
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
    
    % create ssm objects
    o1 = ssm({arn(1)},F1,std1^2,0,std1^2,1,R,y);
    o2 = ssm({arn(1)},F2,std2^2,0,std2^2,1,R,y);
    
    % Run VB inference - no M steps
    Mprob = VBlearn_original([o1, o2], 'dwellp', dwell_prob, 'maxVB_iter', 1);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    original_percent_correct(n) = mean(abs(Mprob(1,:) - s) <= 0.1);
    
    Mprob = VBlearn([o1, o2], 'dwellp', dwell_prob, 'maxVB_iter', 1);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    improved_percent_correct(n) = mean(abs(Mprob(1,:) - s) <= 0.1);
    
    % Run comparison inference algorithms
    Mprob = rand(1,T); % random segmentation
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    random_percent_correct(n) = mean(abs(Mprob - s) <= 0.1);
    
    Mprob = switching([o1, o2],'method','static','dwellp',dwell_prob);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    static_percent_correct(n) = mean(abs(Mprob(1,:) - s) <= 0.1);
    
    Mprob = switching([o1, o2],'method','gpb1','dwellp',dwell_prob);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    gpb1_percent_correct(n) = mean(abs(Mprob(1,:) - s) <= 0.1);
    
    Mprob = switching([o1, o2],'method','gpb2','dwellp',dwell_prob);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    gpb2_percent_correct(n) = mean(abs(Mprob(1,:) - s) <= 0.1);
    
    Mprob = switching([o1, o2],'method','IMM','dwellp',dwell_prob);
    Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
    IMM_percent_correct(n) = mean(abs(Mprob(1,:) - s) <= 0.1);
end

%%
% save('Experiment1_simulation_results')

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

%% Make an example time series plot 

t = 1:T;

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

% save('Experiment1_example.mat', 's', 'y')

figure; plot(s, 'LineWidth', 2)
yyaxis right
plot(t, y)

% create ssm objects
o1 = ssm({arn(1)},F1,std1^2,0,std1^2,1,R,y);
o2 = ssm({arn(1)},F2,std2^2,0,std2^2,1,R,y);

% Run VB inference - no M steps
Mprob = VBlearn_original([o1, o2], 'dwellp', dwell_prob, 'maxVB_iter', 1);
Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
anneal_acc = mean(abs(Mprob(1,:) - s) <= 0.1);
anneal = Mprob;

Mprob = VBlearn([o1, o2], 'dwellp', dwell_prob, 'maxVB_iter', 1);
Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
interp_acc = mean(abs(Mprob(1,:) - s) <= 0.1);
interp = Mprob; 

% Run comparison inference algorithms
Mprob = rand(1,T); % random segmentation
Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
rand_acc = mean(abs(Mprob - s) <= 0.1);
random = Mprob;

Mprob = switching([o1, o2],'method','static','dwellp',dwell_prob);
Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
static_acc = mean(abs(Mprob(1,:) - s) <= 0.1);
static = Mprob;

Mprob = switching([o1, o2],'method','IMM','dwellp',dwell_prob);
Mprob(Mprob>=0.5)=1; Mprob(Mprob<0.5)=0;
imm_acc = mean(abs(Mprob(1,:) - s) <= 0.1);
imm = Mprob;


% visualize the simulated time series and segmentation 
s_x = 1:T;
s_x(s==0) = []; 

s_copy = s; 
s_copy(s==0) = []; 

ori_x = 1:T;
ori_x(anneal(1,:)==0) = []; ori = anneal(1,anneal(1,:)==1);

imp_x = 1:T;
imp_x(interp(1,:)==0) = []; imp = interp(1,interp(1,:)==1);

ran_x = 1:T;
ran_x(random(1,:)==0) = []; ran = random(1,random(1,:)==1);

sta_x = 1:T;
sta_x(static(1,:)==0) = []; sta = static(1,static(1,:)==1);

imm_x = 1:T;
imm_x(imm(1,:)==0) = []; imm_1 = imm(1,imm(1,:)==1);


figure
subplot(2,1,1)
hold on
m1 = plot(t, y, 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2);
[cons, inds] = consecutive(s);
for ii = 1:length(inds)
    m2 = plot(t(inds{ii}), y(inds{ii}), 'm-', 'LineWidth', 2);
end
ylim([-25,25])
ylabel('Simulated Signal')
xlabel('Samples')
legend([m2, m1], {'S_t=1', 'S_t=2'})
title('Example 1 Simulated Time Series')
set(gca,'FontSize',20)

subplot(2,1,2)
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


