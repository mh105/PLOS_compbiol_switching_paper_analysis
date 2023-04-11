function [SS_pp, Xf, Xf0, Xp] = sparse_kalmanfilt_20210420(C, E, F, G, Mu, Ns, prior, Q, T, k_version, Y, iiter, saveoutput_fnamestub)
%%  Adapted from kalmanfilt_SS function written by Elvira Pirondini (elvira.pirondini@epfl.ch)
Ix = eye(Ns);
Xf0 = Mu;

switch prior.Kalman
    case 'standard'
        % Set up file names for saving covariances
        indicator = 1;
        Pf_fname = strcat(saveoutput_fnamestub,'_','Pf_', k_version);
        save(Pf_fname, 'indicator')
        
        Pp_fname = strcat(saveoutput_fnamestub,'_','Pp_', k_version);
        save(Pp_fname, 'indicator')
        
        % Initialization
        t = 0;
        Pf = E;
        Pf_save.(strcat('Pf', num2str(t))) = single((Pf));
        save(Pf_fname, '-struct', 'Pf_save', '-regexp', num2str(t), '-append')
        clear Pf_save
        
        Xp = zeros(Ns, T);
        Xf = zeros(Ns, T);
        
        for t = 1:T
            % Prediction
            if t == 1
                Xp(:,t) = F*Xf0;                % Predicted (a priori) state estimate
            else
                Xp(:,t) = F*Xf(:,t-1);
            end
            Pp = F*Pf*F' + Q;                   % Predicted (a priori) estimate covariance
            
            % Update
            y_tilde = Y(:,t) - G*Xp(:,t);       % Innovation or measurement residual
            St = G*Pp*G' + C;                   % Innovation (or residual) covariance
            K = Pp*G'/St;                       % Optimal Kalman gain
            Xf(:,t) = Xp(:,t) + K*y_tilde;      % Updated (a posteriori) state estimate
            Pf = (Ix-K*G)*Pp;                   % Updated (a posteriori) estimate covariance
            
            % Save covariances for FIS
            Pp_save.(strcat('Pp', num2str(t))) = single(Pp);
            save(Pp_fname, '-struct', 'Pp_save', '-regexp', num2str(t), '-append')
            clear Pp_save
            
            Pf_save.(strcat('Pf', num2str(t))) = single(Pf);
            save(Pf_fname, '-struct', 'Pf_save', '-regexp', num2str(t), '-append')
            clear Pf_save
        end
        
        SS_pp = 0; % Not relevant for the standard version; just putting it here so the code output is defined.
        
    case 'SS'
        % Form Hamiltonian matrix (H)
        h11 = (Ix/F).';
        h12 = h11*G.'/C*G;
        h21 = Q*h11;
        h22 = F + Q*h11*G.'/C*G;
        H = [h11 h12; h21 h22];
        
        eigtic = tic;
        [V, D] = eig(H); % V: cols are the right evecs; D: diag mat of evals.
        disp('Finding eigenvalues takes:')
        toc(eigtic)
        
        lambda2 = abs(diag(D)); % diag(real(D)) is wrong, we shouldn't take the real components of complex eigenvalues 
        
        if any(lambda2 == 1)
            error('KalmanSSeig_Build: unit Eigenvalue in Hamiltonian matrix');
        end
        
        % Find Eigenvalues outside unit circle
        eigVecs = V(:,lambda2 > 1);
        assert(size(eigVecs,2) == Ns, 'Hamiltonian matrix is not symplectic.')
        
        
        % Prior covariance estimate
        Pp_ss = eigVecs(Ns+1:end,:)/eigVecs(1:Ns,:); % See Malik et al. 2011 / Simon 2006 / Vaughan 1970
        Pp_ss = real(Pp_ss); % this real() operator is not explicitly justified, see Vaughan 1970 for original derivation
        
        
        % Compute steady state Kalman gain
        K_ss = Pp_ss*G.'/(G*Pp_ss*G.'+C);
        Pf_ss = (Ix-K_ss*G)*Pp_ss;
        
        Xp = zeros(Ns, T);
        Xf = zeros(Ns, T);
        
        for t = 1:T
            % Prediction
            if t == 1
                Xp(:,t) = F*Xf0;
                
            else
                Xp(:,t) = F*Xf(:,t-1);
            end
            
            % Update
            Xf(:,t) = Xp(:,t) + K_ss*(Y(:,t)-G*Xp(:,t));
        end
        SS_pp.Pp = Pp_ss;
        SS_pp.Pf = Pf_ss;
        
        
%         % Save
%         save([saveoutput_fnamestub,'_kalmanvars_internal_iter',num2str(iiter),'.mat'],...
%             'K_ss','Pf_ss','Pp_ss','h11','h12','h21','h22','H','eigVecs','k','lambda2','D')
        
    case 'standard-SS'
        %%% Use the standard KF for 25 iters, then take that value of Pp Pf
        %%% as the steady-state value for further iters.
        
        % Initialization
        t = 0;
        Pf = E;
        
        %         % Set up file names for saving covariances
        %         indicator = 1;
        %         Pf_fname = strcat(saveoutput_fnamestub,'_Pf', num2str(iiter));
        %         save(Pf_fname, 'indicator')
        %
        %         Pp_fname = strcat(saveoutput_fnamestub,'_Pp', num2str(iiter));
        %         save(Pp_fname, 'indicator')
        %
        %         Pf_save.(strcat('Pf', num2str(t))) = single((Pf));
        %         save(Pf_fname, '-struct', 'Pf_save', '-regexp', num2str(t), '-append')
        %         clear Pf_save
        
        %         %%% Store -- don't; allow it to overwrite
        %         eval(['Pf_struct.Pf',num2str(t),' = single(Pf);']);
        
        Xp = zeros(Ns, T);
        Xf = zeros(Ns, T);
        
        checker_K = 0;
        for t = 1:25
            %             disp(t)
            % Prediction
            if t == 1
                Xp(:,t) = F*Xf0;                % Predicted (a priori) state estimate
            else
                Xp(:,t) = F*Xf(:,t-1);
            end
            
            Pp = F*Pf*F' + Q;                   % Predicted (a priori) estimate covariance
            
            % Update
            y_tilde = Y(:,t) - G*Xp(:,t);       % Innovation or measurement residual
            St = G*Pp*G' + C;                   % Innovation (or residual) covariance
            %             disp(['Computing standard K, iiter=',num2str(iiter),', t=',num2str(t)])
            K = Pp*G'/St;                       % Optimal Kalman gain
            checker_K = any(isnan(K(:)));
            Xf(:,t) = Xp(:,t) + K*y_tilde;      % Updated (a posteriori) state estimate
            Pf = (Ix-K*G)*Pp;                   % Updated (a posteriori) estimate covariance
            
            %             % Save covariances for debugging
            %             Pp_save.(strcat('Pp', num2str(t))) = single(Pp);
            %             save(Pp_fname, '-struct', 'Pp_save', '-regexp', num2str(t), '-append')
            %             clear Pp_save
            %
            %             Pf_save.(strcat('Pf', num2str(t))) = single(Pf);
            %             save(Pf_fname, '-struct', 'Pf_save', '-regexp', num2str(t), '-append')
            %             clear Pf_save
            
            
            %             %%% Store -- don't store; allow it to overwrite.
            %             eval(['Pp_struct.Pp',num2str(t),' = single(Pp);']);
            %             eval(['Pf_struct.Pf',num2str(t),' = single(Pf);']);
        end
        %         %%% Save
        %         save([saveoutput_fnamestub,'Pp_struct.mat'],'Pp_struct','v7.3') % Very slow
        %         save([saveoutput_fnamestub,'Pf_struct.mat'],'Pf_struct','v7.3')
        
        % Use last iter's Pp as Pp_ss
        %         disp('Computing Pp_ss')
        Pp_ss = Pp;
        
        % Compute steady state Kalman gain
        %         disp('Computing SS K')
        K_ss = Pp_ss*G.'/(G*Pp_ss*G.'+C);
        checker_Kss = any(isnan(K_ss(:)));
        Pf_ss = (Ix-K_ss*G)*Pp_ss;
        
        for t = 26:T
            %             disp(t)
            % Prediction
            if t == 1
                Xp(:,t) = F*Xf0;
            else
                Xp(:,t) = F*Xf(:,t-1);
            end
            
            % Update
            Xf(:,t) = Xp(:,t) + K_ss*(Y(:,t)-G*Xp(:,t));
        end
        
        % Steady-state values will be saved outside this function.
        SS_pp.Pp = Pp_ss;
        SS_pp.Pf = Pf_ss;
        SS_pp.K_ss = K_ss;
end

end