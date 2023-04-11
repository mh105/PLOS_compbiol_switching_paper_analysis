function [Phi_new, Q_new, R_new, mu_new, R_mode_state]=m_step(y, NNharm, x_t_n, x_t_t, x_t_tmin1, P_t_t, P_t_tmin1, P_t_n, P_t_tmin1_n, w_fond_range, noise_osc, w_init, vmp_param, prior_on, Q_init, R_init, Phi_old, Fs, updateR)
%% Maximization step for the extended State-Space oscillator decomposition (Matsuda and Komaki 2017).
% INPUTS:
%   y                 : signal for a given window
%   NNharm            : 1 x #indpt oscillations an array containing the number of harmonics
%                       for each independant oscillation
%   w_fond_range      : range to look for fundamental frequency
%                     in case of harmonic estimation
%   x_t_n, P_t_n, ... : state means and covariance estimates from E-step
%
% OUTPUTS:
%   Updated parameters

% Prepare Von Mises prior parameters 
osc_num=vmp_param.osc_num;
kappa = vmp_param.kappa;
mu_peak = vmp_param.mu;
if isempty(osc_num)
    kappa = 0; %This keeps kappa from affecting Harmonics step
end

% Prepare oscillator parameters 
T=length(y);               % length of recording 
Nosc =  sum(NNharm);       % Total number of oscillation
M=repmat([1 0], 1, Nosc);  % Define observation matrix M

% Pre-compute sums used for M-step equations: see Shumway & Stoffer 1982
A = sum(P_t_n(:,:,1:end-1),3) + x_t_n(:,1:end-1)*x_t_n(:,1:end-1)';
B = sum(P_t_tmin1_n(:,:,2:end),3) + x_t_n(:,2:end)*x_t_n(:,1:end-1)';
C = sum(P_t_n(:,:,2:end),3) + x_t_n(:,2:end)*x_t_n(:,2:end)';

% Update initial state means mu
mu_new = x_t_n(:,1);

% Update observation noise covariance R
if updateR
    R_ss=sum((y'-M*x_t_n(:,2:end)).^2)+M*sum(P_t_n(:,:,2:end),3)*M';
%     R_mode=R_init;
%     alpha=10; %T*0.01;
%     beta=R_mode*(alpha+1);
%     R_new=(R_ss+2*beta)/(T+2*(alpha+1));
    R_new = R_ss/T; % this is ML estimate without prior
else
    R_new = R_init;
end

% Update Phi and Q matrices 
Phi_new=zeros(2*Nosc);
Q_new  =zeros(2*Nosc);
R_mode_state = zeros(length(NNharm),1); % store 'inv_gamma' prior for state noise covariance 

for n_osc_ind=1:length(NNharm) % Number of independant oscillation
    
    % Update Phi and Q for an oscillation with no harmonics
    if NNharm(1,n_osc_ind)==1
        n_harm=1;
        curId = 2*( sum(NNharm(1,1:n_osc_ind)) - NNharm(1,n_osc_ind) + n_harm) -1;
        
        A_tmp = A (curId:curId+1,curId:curId+1);
        B_tmp = B (curId:curId+1,curId:curId+1);
        C_tmp = C (curId:curId+1,curId:curId+1);
        
        % Step 1: update rotation frequency radian parameter w
        if any(osc_num==n_osc_ind) %If oscillation is flagged, use Von Mises Prior (VMP)
            ind = find(osc_num==n_osc_ind);
            w_new = atan2(rt(B_tmp) + kappa(ind)*sin(mu_peak(ind)), trace(B_tmp) + kappa(ind)*cos(mu_peak(ind))); % kappa is adaptive as kappa * a_new/sigma_Q_new
        elseif any(noise_osc==n_osc_ind) %Keep Noise oscillation frequency from updating
            w_new = w_init(n_osc_ind);
        else
            w_new = atan2(rt(B_tmp),trace(B_tmp)); %No VMP
        end
        
        % Step 2: update rotation damping factor a
        a_new = (cos(w_new)*trace(B_tmp) + sin(w_new)*rt(B_tmp)) / trace(C_tmp);
        
        % Step 3: update the transition matrix block with updated w and a 
        Phi_new(curId:curId+1,curId:curId+1) = get_rot_mat(a_new, w_new);
        
        % Step 4: update state noise covariance matrix Q
        Q_ss = 1/2*(trace(C_tmp) - 2*a_new*(cos(w_new)*trace(B_tmp) + sin(w_new)*rt(B_tmp)) + a_new^2*trace(A_tmp));
        switch prior_on
            case 'no_prior'
                sigma2_Q_new = Q_ss / T;
            case 'inv_gamma' % not recommended anymore 
                %Patrick's method - an acceleration heuristic method
                R_mode_state(n_osc_ind) = P_t_t(n_osc_ind*2-1,n_osc_ind*2-1) + mean((x_t_t(n_osc_ind*2-1,2:end)-x_t_tmin1(n_osc_ind*2-1,2:end)).^2);
                alpha = 10; %T*0.01;
                beta = R_mode_state(n_osc_ind)*(alpha+1);
                sigma2_Q_new = (Q_ss+2*beta)/(T+2*(alpha+1));
            case 'jeff'
                sigma2_Q_new = Q_ss / (T+2);
            case 'approx_zeronorm'
                gamma=10;
                temp=Q_ss-gamma-T;
                sigma2_Q_new=(temp/2+sqrt(temp^2/4+2*Q_ss*gamma*(2+T/2)))/(gamma*(4+T));
            case 'laplace'
                gamma=1;
                sigma2_Q_new=(sqrt(T^2+8*gamma*Q_ss)-T)/(4*gamma);
        end
        Q_new(curId:curId+1,curId:curId+1) = eye(2) * sigma2_Q_new;
    
    % Update Phi and Q for an oscillation containing harmonics <<< WARNING: this has not been updated with adaptive VM prior 
    else 
        Nharm_cur= NNharm(1, n_osc_ind);% Number of harmonics
        
        rHarm=1:1:Nharm_cur;
        rtBr=zeros(1,Nharm_cur);
        trBr=zeros(1,Nharm_cur);
        trAr=zeros(1,Nharm_cur);
        trCr=zeros(1,Nharm_cur);
        
        for n_harm=1:Nharm_cur
            curId= 2*(sum(NNharm(1,1:n_osc_ind)) - NNharm(1,n_osc_ind) + n_harm) -1;
            A_tmp=A(curId:curId+1,curId:curId+1);
            B_tmp=B(curId:curId+1,curId:curId+1);
            C_tmp=C(curId:curId+1,curId:curId+1);
            
            trAr(1,n_harm)=trace(A_tmp);
            trBr(1,n_harm)=trace(B_tmp);
            rtBr(1,n_harm)=   rt(B_tmp);
            trCr(1,n_harm)=trace(C_tmp);
        end
        
        if any(osc_num==n_osc_ind)
            ind=find(osc_num==n_osc_ind);
            mu_i=mu_peak(ind);
            kappa_i=kappa(ind);
        else
            kappa_i=0; %since kappa_i is 0, mu_i shouldn't matter.
            mu_i=0;
        end
        
        % Define expected log likelihood G = E (logL | y1...yn)
        Gharm= @(w, sigma2r, ar) kappa_i*cos(w-mu_i) - T* sum(log(sigma2r),2) -(1/2)* sum(   sigma2r.^(-1).*(trCr-ar.^2.*trAr)   ,2);
        dGdw = @(w) -kappa_i*sin(w-mu_i) + sum( rHarm.* (  rtBr.*trBr.*cos(2.*rHarm.*w) + 0.5*(rtBr.^2-trBr.^2).*sin(2.*rHarm.*w)   ) ...
            ./  (  (trAr.*trCr-0.5*(rtBr.^2+trBr.^2))  -  ( 0.5*(trBr.^2-rtBr.^2).*cos(2.*rHarm.*w)+rtBr.*trBr.*sin(2.*rHarm.*w) )    ) );
        
        % set the interval to look for optimal w_h
        w_r_min=w_fond_range(1,n_osc_ind);
        w_r_max=w_fond_range(2,n_osc_ind);
        
        % Find all the zeros of dG/dw_heart
        w_r_opt_k=AllMin(dGdw,w_r_min,w_r_max,500);
        
        if isempty(w_r_opt_k)
            w_r_opt=0.5*(w_r_min+w_r_max);
            disp('no min')
        else
            Gmax=-Inf;
            % In case of multiple local minima, find the one which minimizes G=log(L) itself
            Gtot_tmp = zeros(1,length(w_r_opt_k));
            for k=1:length(w_r_opt_k)
                % if k==2;disp('mult');end
                w_new=w_r_opt_k(1,k);
                a_r_tmp      = max((  trBr.*cos(rHarm.*w_new)+rtBr.*sin(rHarm*w_new)  ) ./trAr, 0.01);
                sigma2_r_tmp = (1/(2*T)) .*(trCr-a_r_tmp.^2.*trAr);
                Gtmp=Gharm(w_new , sigma2_r_tmp, a_r_tmp);
                Gtot_tmp(1,k)=Gtmp;
                if Gtmp>Gmax && isreal(Gtmp)
                    Gmax=Gtmp;
                    w_r_opt=w_new;
                end
            end
            
        end
        
        a_r_new      = max((  trBr.*cos(rHarm.*w_r_opt)+rtBr.*sin(rHarm*w_r_opt)  ) ./trAr,0.01);
        sigma2_r_new = (1/(2*T)) .*(trCr-a_r_new .^2.*trAr);
        
        % Build optimal harmonic functions
        for n_harm=1:Nharm_cur
            curId= 2*( sum(NNharm(1,1:n_osc_ind)) - NNharm(1,n_osc_ind) + n_harm) -1;
            Phi_tmp=get_rot_mat(a_r_new(1,n_harm),n_harm*w_r_opt);
            
            Phi_new(curId:curId+1, curId:curId+1) = Phi_tmp;
            Q_new  (curId:curId+1, curId:curId+1) = sigma2_r_new(1,n_harm);
            Q_new  (curId:curId+1, curId:curId+1) = sigma2_r_new(1,n_harm);
        end
        
    end
    
end

end

function z=AllMin(f,xmin,xmax,N)
% Find all zero crossing from positive to negative
if (nargin<4)
    N=100;
end
dx=(xmax-xmin)/N;
x2=xmin;
y2=f(x2);
z=[];
for i=1:N
    x1=x2;
    y1=y2;
    x2=xmin+i*dx;
    y2=f(x2);
    if (y1*y2<=0) && (y1>0)
        options = optimset('Display','off');
        z=[z,fsolve(f,(x2*y1-x1*y2)/(y1-y2),options)];
    end
end
end
