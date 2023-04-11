function [A, logL, Pst, Xs, Xs0, A1, A2, A3, B, B1, B2] = sparse_smoother_20210318(C, G, F, Ns, prior, SS_pp, T, k_version, Xf, Xf0, Xp, Y, saveoutput_fnamestub)
%%  Adapted from smootheremmap_SS function written by Elvira Pirondini (elvira.pirondini@epfl.ch)
Xs(:,T) = Xf(:,T);
A1 = zeros(Ns);
A2 = A1;
A3 = A1;
B = 0; % Output for debug
%%% For debug
B1 = 0; % Output for debug
B2 = 0; % Output for debug
%%%%%%
logL = 0;

switch prior.Kalman
    case 'standard'
        Pp_fname = strcat(saveoutput_fnamestub,'_','Pp_', k_version);
        Pf_fname = strcat(saveoutput_fnamestub,'_','Pf_', k_version);
        load_struct = load(Pf_fname, strcat('Pf', num2str(T)));
        Pst = double(load_struct.(strcat('Pf', num2str(T))));
        clear load_struct
        
        for t = T:-1:1
            % Load covariances
            load_struct = load(Pf_fname, strcat('Pf', num2str(t-1)));
            Pft_minus = double(load_struct.(strcat('Pf', num2str(t-1))));
            clear load_struct
            
            load_struct = load(Pp_fname, strcat('Pp', num2str(t)));
            Ppt = double(load_struct.(strcat('Pp', num2str(t))));
            clear load_struct
            
            % Backward smoother
            Jt_minus = Pft_minus*F'/Ppt;
            
            if t == 1
                Xs0 = Xf0 + Jt_minus*(Xs(:,t)-Xp(:,t));
            else
                Xs(:,t-1) = Xf(:,t-1) + Jt_minus*(Xs(:,t)-Xp(:,t));
            end
            
            Pst_minus = Pft_minus + Jt_minus*(Pst-Ppt)*Jt_minus';
            
            % One step covariance
            Plagt = Pst*Jt_minus';
            
            %             % Accumulation of A1, A2, and A3
            %             A1 = A1 + Pst + Xs(:,t)*Xs(:,t)';
            %
            %             if t == 1
            %                 A2 = A2 + Plagt + Xs(:,t)*Xs0';
            %                 A3 = A3 + Pst_minus + Xs0*Xs0';
            %
            %             else
            %                 A2 = A2 + Plagt + Xs(:,t)*Xs(:,t-1)';
            %                 A3 = A3 + Pst_minus + Xs(:,t-1)*Xs(:,t-1)';
            %             end
            %
            %             if strcmp(prior.C, 'Wishart')
            %                 B = B + (Y(:,t)-G*Xs(:,t))*(Y(:,t)-G*Xs(:,t))' + G*Pst*G';
            %
            %                 %%% Output for debug
            %                 B1 = B1 + (Y(:,t)-G*Xs(:,t))*(Y(:,t)-G*Xs(:,t))';
            %                 B2 = B2 + G*Pst*G';
            %             end
            
            % Accumulation of logL
            %             logL = logL - log(det((G*Ppt*G'+C)))/2 - (Y(:,t)-G*Xp(:,t))'/(G*Ppt*G'+C)*(Y(:,t)-G*Xp(:,t))/2;
            logL = logL - sparse_find_log_det(G*Ppt*G'+C)/2 - (Y(:,t)-G*Xp(:,t))'/(G*Ppt*G'+C)*(Y(:,t)-G*Xp(:,t))/2;
            Pst = Pst_minus;
        end
        
    case {'SS', 'standard-SS'} % 'SS'
        %         disp('Using steady-state smoother')
        
        Pft = SS_pp.Pf;
        Ppt = SS_pp.Pp;
        %         disp('Computing Jt_minus')
        Jt_minus = Pft*F'/Ppt;
%         checker_Jtminus = any(isnan(Jt_minus(:)));
        Pst = Pft + Jt_minus*(Pft-Ppt)*Jt_minus'; % this expression is incorrect. It should be Pst-Ppt inside the paranthesis
        Plagt = Pst*Jt_minus';
        
        for t = T:-1:1
            %             disp(t)
            % Backward smoother
            if t == 1
                Xs0 = Xf0 + Jt_minus*(Xs(:,t)-Xp(:,t));
            else
                Xs(:,t-1) = Xf(:,t-1) + Jt_minus*(Xs(:,t)-Xp(:,t));
            end
            
            %             % Accumulation of A1, A2, and A3
            %             A1 = A1 + Pst + Xs(:,t)*Xs(:,t)';
            %
            %             if t == 1
            %                 A2 = A2 + Plagt + Xs(:,t)*Xs0';
            %                 A3 = A3 + Pst + Xs0*Xs0';
            %             else
            %                 A2 = A2 + Plagt + Xs(:,t)*Xs(:,t-1)';
            %                 A3 = A3 + Pst + Xs(:,t-1)*Xs(:,t-1)';
            %             end
            %
            %             if strcmp(prior.C, 'Wishart')
            %                 B = B + (Y(:,t)-G*Xs(:,t))*(Y(:,t)-G*Xs(:,t))' + G*Pst*G';
            %
            %                 %%% Output for debug
            %                 B1 = B1 + (Y(:,t)-G*Xs(:,t))*(Y(:,t)-G*Xs(:,t))';
            %                 B2 = B2 + G*Pst*G';
            %             end
            
            % Accumulation of logL
            % This is p(y|theta). See Lamus 2012 eqn 28.
            %             logL = logL - log(det((G*Ppt*G'+C)))/2 - (Y(:,t)-G*Xp(:,t))'/(G*Ppt*G'+C)*(Y(:,t)-G*Xp(:,t))/2;
            logL = logL - sparse_find_log_det(G*Ppt*G'+C)/2 - (Y(:,t)-G*Xp(:,t))'/(G*Ppt*G'+C)*(Y(:,t)-G*Xp(:,t))/2;
        end
end

A = A1 - A2*F' - F*A2' + F*A3*F';

end
