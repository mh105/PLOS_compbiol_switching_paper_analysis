classdef gen < ssm
    %GEN is a subclass for general single component structure in a SSM
    %class object. It is created to provide an interface with unified
    %notations and properties for future generalization of ssm objects with
    %different component structures that are accompanied by individual
    %method functions such as M-step and Z-transform analysis.
    %
    %This subclass structure could also be used as the starting point for a
    %more flexible object such as with additional properties that are not
    %contained in the ssm superclass or structure-specific methods that are
    %not shared across different components. One should call the default
    %constructor ssm() and ensure the new object class at least contains
    %all properties of a ssm object, and it should define all methods
    %defined here. The ctype property is checked by mustBeComponent() to
    %ensure these methods can be successfully called.
    %
    %There is no need to redefine methods in the ssm superclass, which
    %work at the model level instead of at the component level.
    %
    % Authors: Alex He; Last edit: 01/03/2022
    %
    %      y_t   ~ G x_t + N(0,R) - Observation Equation
    %      x_t+1 ~ F x_t + N(0,Q) - State Equation
    
    methods
        %% BASIC METHODS - OPERATORS ON GEN OBJECTS
        function obj = gen(varargin) % _init_ construct an instance of this subclass
            % call the superclass constructor: ssm(ctype,F,Q,mu0,Q0,G,R,y,Fs)
            obj = obj@ssm('general ssm',varargin);
            if exist('G','var') && ~isempty(G); obj.G = G; else; obj.G = default_G(obj); end
        end
        
        function G = default_G(obj) % output the default observation matrix G given component type
            assert(size(obj.F,1) > 0, 'To construct a GEN class ssm object, you must provide all state equation parameters.')
            G = ones(1, size(obj.F,1));
        end
        
        %% PARAMETER ESTIMATION METHODS - EM
        function [F_new, Q_new, mu0_new, Q0_new, G_new, R_new, R_ss] = ml_estimate(obj, x_t_n, P_t_n, P_t_tmin1_n, ht, A, B, C, T) % MLE of state equation parameters
            if nargin < 6
                % Definitions of A,B,C follow the notation used in equations (9,10,11) of S&S 1982
                A = sum(P_t_n(:,:,1:end-1),3) + x_t_n(:,1:end-1)*x_t_n(:,1:end-1)';
                B = sum(P_t_tmin1_n(:,:,2:end),3) + x_t_n(:,2:end)*x_t_n(:,1:end-1)';
                C = sum(P_t_n(:,:,2:end),3) + x_t_n(:,2:end)*x_t_n(:,2:end)';
                T = size(x_t_n,2)-1;
            end
            
            % ----------------------------------------------------
            % Component type specific update equations
            % ----------------------------------------------------
            % Update transition matrix --- F
            F_new = B/A;
            
            % Update state noise covariance matrix --- Q
            Q_new = 1/T*(C - B/A*B');
            % ----------------------------------------------------
            
            if nargout > 2
                % Update initial state mean --- mu0
                mu0_new = x_t_n(:,1);
                
                % Update initial state covariance --- Q0
                Q0_new = P_t_n(:,:,1) + x_t_n(:,1)*x_t_n(:,1)' - x_t_n(:,1)*mu0_new' - mu0_new*x_t_n(:,1)' + mu0_new*mu0_new';
                
                % Update observation matrix --- G
                y = obj.y;
                ht_3D(1,1,:) = ht;
                G_new = (ht .* y) * x_t_n(:,2:end)' / (sum(ht_3D .* P_t_n(:,:,2:end), 3) + (ht .* x_t_n(:,2:end)) * x_t_n(:,2:end)');
                
                % Update observation noise covariance --- R
                G = obj.G;
                R_ss = (y - G * x_t_n(:,2:end)) * (ht .* (y - G * x_t_n(:,2:end)))' + G * sum(ht_3D .* P_t_n(:,:,2:end), 3) * G'; % weighted version of the sum in equation (14) of S&S 1982
                R_new = R_ss / sum(ht);
            end
        end
        
        function [F_new, Q_new, mu0_new, Q0_new, G_new, R_new, R_ss, A, B, C] = map_estimate(obj, x_t_n, P_t_n, P_t_tmin1_n, ht, prior_sets, A, B, C, T) % MAP estimate of state equation parameters
            if nargin < 7
                % Definitions of A,B,C follow the notation used in equations (9,10,11) of S&S 1982
                A = sum(P_t_n(:,:,1:end-1),3) + x_t_n(:,1:end-1)*x_t_n(:,1:end-1)';
                B = sum(P_t_tmin1_n(:,:,2:end),3) + x_t_n(:,2:end)*x_t_n(:,1:end-1)';
                C = sum(P_t_n(:,:,2:end),3) + x_t_n(:,2:end)*x_t_n(:,2:end)';
                T = size(x_t_n,2)-1;
            end
            
            % ----------------------------------------------------
            % Component type specific update equations (no prior)
            % ----------------------------------------------------
            % Update transition matrix --- F
            F_new = B/A;
            
            % Update state noise covariance matrix --- Q
            Q_new = 1/T*(C - B/A*B');
            % ----------------------------------------------------
            
            if nargout > 2
                % Update initial state mean --- mu0 (no prior)
                mu0_new = x_t_n(:,1);
                
                % Update initial state covariance --- Q0 (no prior)
                Q0_new = P_t_n(:,:,1) + x_t_n(:,1)*x_t_n(:,1)' - x_t_n(:,1)*mu0_new' - mu0_new*x_t_n(:,1)' + mu0_new*mu0_new';
                
                % Update observation matrix --- G (no prior)
                y = obj.y;
                ht_3D(1,1,:) = ht;
                G_new = (ht .* y) * x_t_n(:,2:end)' / (sum(ht_3D .* P_t_n(:,:,2:end), 3) + (ht .* x_t_n(:,2:end)) * x_t_n(:,2:end)');
                
                % Update observation noise covariance --- R (inverse gamma prior)
                G = obj.G;
                R_ss = (y - G * x_t_n(:,2:end)) * (ht .* (y - G * x_t_n(:,2:end)))' + G * sum(ht_3D .* P_t_n(:,:,2:end), 3) * G'; % weighted version of the sum in equation (14) of S&S 1982
                R_init = prior_sets(1).R; % all components should have the same prior on R so calling from the first one
                alpha = sum(ht) * 0.1 / 2; % shape parameter is set to be peaky
                beta  = R_init * (alpha + 1); % setting the mode of inverse gamma prior to be R_init
                R_new = (beta + R_ss/2) / (alpha + sum(ht)/2 + 1); % using the mode of inverse gamma posterior as R_new
            end
        end
        
        function [prior_set] = initialize_priors(obj) % Initialize prior parameters for the component
            % Inverse gamma prior on R
            prior_set.R = obj.R;
        end
        
    end
end

%% HELPER FUNCTIONS
