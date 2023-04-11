classdef osc < ssm
    %OSC is a subclass for Matsuda oscillator components in a SSM class
    %object. It is created to provide an interface with unified notations
    %and properties for future generalization of ssm objects with different
    %component structures that are accompanied by individual method
    %functions such as M-step and Z-transform analysis.
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
        %% BASIC METHODS - OPERATORS ON OSC OBJECTS
        function obj = osc(varargin) % _init_ construct an instance of this subclass
            % call the superclass constructor: ssm(ctype,F,Q,mu0,Q0,G,R,y,Fs)
            obj = obj@ssm('osc',varargin);
            if exist('G','var') && ~isempty(G); obj.G = G; else; obj.G = default_G(obj); end
        end
        
        function G = default_G(obj) % output the default observation matrix G given component type
            G = repmat([1,0], max(1,size(obj.F,1)/2));
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
            % Update the rotation radian --- w
            w_new = atan2(rt(B), trace(B));
            
            % Update the rotation damping factor --- a
            a_new = (cos(w_new)*trace(B) + sin(w_new)*rt(B)) / trace(C);
            
            % Use the updated w and a to construct transition matrix --- F
            F_new = get_rot_F(a_new, w_new);
            
            % Update state noise covariance matrix --- Q
            sigma2_Q_new = 1/(2*T)*(trace(C) - 2*a_new*(cos(w_new)*trace(B) + sin(w_new)*rt(B)) + a_new^2*trace(A));
            Q_new = eye(2) * sigma2_Q_new;
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
            % Component type specific update equations
            % ----------------------------------------------------
            % Update the rotation radian --- w (adaptive Von Mises prior)
            kappa = prior_sets.vmp_param.kappa;
            w_prior = prior_sets.vmp_param.w_prior;
            w_new = atan2(rt(B) + kappa*sin(w_prior), trace(B) + kappa*cos(w_prior)); % kappa is adaptive as kappa * a_new/sigma2_Q_new
            
            % Update the rotation damping factor --- a (no prior)
            a_new = (cos(w_new)*trace(B) + sin(w_new)*rt(B)) / trace(C);
            
            % Use the updated w and a to construct transition matrix --- F
            F_new = get_rot_F(a_new, w_new);
            
            % Update state noise covariance matrix --- Q (inverse gamma prior)
            Q_ss = trace(C) - 2*a_new*(cos(w_new)*trace(B) + sin(w_new)*rt(B)) + a_new^2*trace(A);
            
            T = T * 2; % double the data length since we sum over real and imaginary noise processes
            
            Q_init = prior_sets.Q;
            alpha = T * 0.1 / 2; % shape parameter is set to be peaky
            beta  = Q_init * (alpha + 1); % setting the mode of inverse gamma prior to be Q_init
            sigma2_Q_new = (beta + Q_ss/2) / (alpha + T/2 + 1); % using the mode of inverse gamma posterior as sigma2_Q_new
            Q_new = eye(2) * sigma2_Q_new;
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
            % Von Mises Prior parameters
            [a_prior, w_prior] = get_rot_aw(obj.F);
            f_prior = w_prior/(2*pi)*obj.Fs;
            prior_set.vmp_param.kappa = 10000;
            prior_set.vmp_param.f_prior = f_prior;
            prior_set.vmp_param.w_prior = w_prior;
            
            % Inverse gamma prior on state noise covariance element of Q
            assert(diff(diag(obj.Q))==0, 'state noise differs between real and imaginary components.')
            prior_set.Q = obj.Q(1,1);
            
            % Inverse gamma prior on R
            prior_set.R = obj.R;
        end
        
    end
end

%% HELPER FUNCTIONS
function [F] = get_rot_F(a,w)
% F = a .* R(w) where R is a rotation matrix with rotation radian w
F = zeros(2,2);
F(1,1) = a * cos(w);
F(1,2) = a * -sin(w);
F(2,1) = a * sin(w);
F(2,2) = a * cos(w);
end

function [a, w] = get_rot_aw(F)
% F = a .* R(w) where R is a rotation matrix with rotation radian w
assert(all(size(F)==[2,2]))
a = sqrt(F(1,1)^2+F(2,1)^2);
w = atan2(F(2,1),F(1,1));
end

function rtB = rt(B)
% Compute the off-diagonal entry difference in a 2x2 matrix 
assert(all(size(B)==[2,2]), 'B is not a 2x2 matrix')
rtB = B(2,1) - B(1,2);
end