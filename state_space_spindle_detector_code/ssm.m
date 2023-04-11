classdef ssm
    %SSM is a MATLAB object class for state-space modeling in Purdon lab.
    %This SSM class supports general ssm structures including AR and other
    %forms of state-space equations, as well as kalman filtering,
    %smoothing, and switching state-space model methods.
    %
    %It should be noted that MATLAB econ toolbox also has a ssm.m function
    %for similar purposes. This function is independent of and should be
    %higher in search path to replace the default MATLAB ssm.m class.
    %
    % Authors: Alex He; Last edit: 07/29/2022
    %
    %      y_t   ~ G x_t + N(0,R) - Observation Equation
    %      x_t+1 ~ F x_t + N(0,Q) - State Equation
    
    properties
        ctype {mustBeComponent} % cell arrays of component objects [customizable]
        % State Equation parameters
        F double {mustBeNumeric, mustBeSquare} % state transition matrix
        Q double {mustBeNumeric, mustBeSquare} % state noise covariance
        mu0 double {mustBeNumeric} % initial state means at t=0
        Q0 double {mustBeNumeric, mustBeSquare} % initial state noise covariance at t=0
        % Observation Equation parameters
        G double {mustBeNumeric} % observation matrix
        R double {mustBeNumeric} % observation noise (co)variance
        y double {mustBeNumeric} % observed data time series (as row vectors)
        Fs double {mustBeNumeric} % sampling frequency for observed data
    end
    
    methods
        %% BASIC METHODS - OPERATORS ON SSM OBJECTS
        function [obj] = ssm(ctype,F,Q,mu0,Q0,G,R,y,Fs) % _init_ construct an instance of this class
            % instantiate state-space equation parameters
            if exist('F','var') && ~isempty(F); obj.F = F; end
            if exist('Q','var') && ~isempty(Q); obj.Q = Q; end
            if exist('mu0','var') && ~isempty(mu0); obj.mu0 = mu0; end
            if exist('Q0','var') && ~isempty(Q0); obj.Q0 = Q0; end
            % instantiate state-space model component types
            if ~exist('ctype','var') || isempty(ctype)
                num_components = length(obj.mu0(:,:,1))/2;
                ctype_store = cell(1,num_components);
                for ii = 1:num_components
                    ctype_store{ii} = osc; % default to all osc components
                end
                obj.ctype = ctype_store;
            else
                obj.ctype = ctype;
            end
            % instantiate observation matrix
            if ~exist('G','var') || isempty(G)
                if iscell(obj.ctype)
                    for ii = 1:length(obj.ctype) % loop through all components
                        obj.G = [obj.G, obj.ctype{ii}.default_G];
                    end
                end
            else
                G_store = [];
                if iscell(G)
                    for ii = 1:length(G)
                        G_store(:,:,ii) = G{ii}; %#ok<*AGROW> % third dimension index for different models
                    end
                else
                    G_store = G;
                end
                obj.G = G_store;
            end
            % instantiate observed data information
            if exist('R','var') && ~isempty(R); obj.R = R; end
            if exist('y','var') && ~isempty(y); obj.y = y; end
            if exist('Fs','var') && ~isempty(Fs); obj.Fs = Fs; end
            % check consistency in state dimensionality
            check_state_cardi(obj)
        end
        
        function [obj] = plus(o1,o2) % OVERLOAD: merge two models into one ssm object
            check_observed_data(o1,o2)
            obj = cat_property(o1,o2,fieldnames(o1));
            % make sure the numbers of alternative models are consistent
            % across the properties after concatenation
            alt_model_num = cellfun(@(x) size(obj.(x),3), fieldnames(obj));
            assert(length(unique(alt_model_num(alt_model_num~=1)))<=1, 'The numbers of alternative models are inconsistent across properties.')
        end
        
        function [obj] = sum(obj_array) % OVERLOAD: sum an array of the ssm objects using plus()
            obj = obj_array(1);
            if length(obj_array)>1
                for ii = 2:length(obj_array)
                    obj = obj + obj_array(ii);
                end
            end
        end
        
        function [obj] = mtimes(o1,o2) % OVERLOAD: multiply two models will invoke permutation of properties
            % concatenate properties just like in plus(), but instead of
            % checking for the same number of alternative models, call
            % expand() to permutate combinations of properties
            check_observed_data(o1,o2)
            obj = cat_property(o1,o2,fieldnames(o1));
            % permutation expansion of properties for alternative models
            obj = expand(obj);
        end
        
        function [obj] = times(o1,o2) % OVERLOAD: multiply two models will invoke permutation of properties
            obj = mtimes(o1,o2);
        end
        
        function [obj] = prod(obj_array) % OVERLOAD: permutation across an array of ssm objects
            % unlike sum(), the prod() function does not call mtimes() in
            % order to increase the efficiency of the expansion step
            obj = obj_array(1);
            if length(obj_array)>1
                for ii = 2:length(obj_array) % iterative concatenation
                    check_observed_data(obj,obj_array(ii))
                    obj = cat_property(obj,obj_array(ii),fieldnames(obj));
                end
                % permutation expansion of properties for alternative models
                obj = expand(obj);
            end
        end
        
        function [obj] = mpower(o1,n) % OVERLOAD: raising a model to power of n is equivalent to taking the product of n models
            obj = prod(repmat(o1,1,n));
        end
        
        function [obj] = power(obj_array,n) % OVERLOAD: raising a model to power of n is equivalent to taking the product of n models
            obj = arrayfun(@(x) x^n, obj_array);
        end
        
        function [obj] = cat_property(o1,o2,Props) % concatenate properties with duplicate checks
            % concatenate properties along the third dimension only if the
            % second model has new information compared to the first model
            obj = o1;
            if ~iscell(Props); Props = {Props}; end
            for ii = 1:length(Props)
                Prop = Props{ii};
                o1_prop = o1.(Prop);
                o2_prop = o2.(Prop);
                if strcmp(Prop,'ctype')
                    % to concatenate two ssm objects, ctype must be identical
                    if iscell(o1_prop)
                        o1_prop = o1_prop(:); o2_prop = o2_prop(:);
                        for jj = 1:length(o1_prop)
                            assert(all(o1_prop{jj}.ctype == o2_prop{jj}.ctype), 'ctype is different between the two models.')
                        end
                    else
                        assert(all(o1_prop == o2_prop), 'ctype is different between the two models.') % subclasses have ctype as strings
                    end
                else
                    new_prop_to_append = o2_prop(:,:,~squeeze(all(o2_prop == o1_prop,[1,2])));
                    obj.(Prop) = cat(3,o1_prop,new_prop_to_append);
                end
            end
        end
        
        function [obj] = expand(obj) % expand with permutation of model properties
            property_list = fieldnames(obj);
            alt_model_num = cellfun(@(x) size(obj.(x),3), property_list);
            update_idx = find(alt_model_num>1);
            total_alt_model_num = prod(alt_model_num(update_idx));
            last_property_multiple = 1;
            for j = 1:length(update_idx)
                % when expanding, we want to make sure there are no
                % repeated entries in a property, which will create many
                % duplicate copies after expansion
                current_property = obj.(property_list{update_idx(j)});
                for m = 1:size(current_property,3)
                    assert(sum(squeeze(all(current_property(:,:,m)==current_property,[1,2])))==1, 'Duplicates found in properties during expand().')
                end
                temp_property = repelem(current_property,1,1,total_alt_model_num/last_property_multiple/alt_model_num(update_idx(j)));
                obj.(property_list{update_idx(j)}) = repmat(temp_property,1,1,last_property_multiple);
                last_property_multiple = last_property_multiple * alt_model_num(update_idx(j));
            end
            check_state_cardi(obj);
        end
        
        function [obj] = add_prop(obj,new_prop,Prop) % add new values into a property of a ssm object
            % add_prop() will append the new property values into the third
            % dimension of the specified property, with duplicate checks
            if iscell(new_prop)
                new_prop_store = [];
                for ii = 1:length(new_prop)
                    new_prop_store(:,:,ii) = new_prop{ii}; % third dimension index for different models
                end
            else
                new_prop_store = new_prop;
            end
            temp_obj = obj;
            temp_obj.(Prop) = new_prop_store;
            obj = cat_property(obj,temp_obj,Prop);
        end
        
        function [] = check_observed_data(obj_array, o2) % make sure an array of models have the same observed data
            if nargin > 1 % if two objects, check all time points
                assert(all(obj_array.y==o2.y), 'observed data are different in the two models.')
                assert(all(obj_array.Fs==o2.Fs), 'sampling frequencies are different in the two models.')
            else % if an array, randomly select one time point to spot check
                rindex = randi(size(obj_array(1).y,2));
                for m = 2:length(obj_array)
                    assert(obj_array(1).y(rindex) == obj_array(m).y(rindex), 'Observed data are not identical across models.')
                end
            end
        end
        
        function [] = check_state_cardi(obj) % check state cardinality consistency
            % make sure the different models have consistent states when instantiating
            % a ssm class object. This check doesn't consider observation (R or G)
            state_model_nums = cellfun(@(x) size(x,3), {obj.F,obj.Q,obj.mu0,obj.Q0});
            assert(length(unique(state_model_nums)) <= 2, 'Too many different alternative models in state equation parameters.')
            assert(size(obj.ctype,3) == 1 || all(size(obj.ctype,3) == unique(state_model_nums)), 'Number of component type structures mismatches number of sets of state equation parameters.')
            if iscell(obj.ctype) && ~isempty(obj.ctype)
                assert(all(diff([size(obj.F,1), size(obj.Q,1), size(obj.mu0,1), size(obj.Q0,1), sum(cellfun(@(x) length(x.default_G), obj.ctype))]) == 0), 'State dimensions are inconsistent.')
            else
                assert(all(diff([size(obj.F,1), size(obj.Q,1), size(obj.mu0,1), size(obj.Q0,1)]) == 0), 'State dimensions are inconsistent.')
            end
        end
        
        function [obj] = fill_components(obj) % fill in parameters of ctypes
            state_cardis = cellfun(@(x) size(x.G,2), obj.ctype);
            for ii = 1:length(state_cardis) % iterate through components
                % fill the ctype objects with parameters for the components
                start_id = sum(state_cardis(1:ii-1)) + 1;
                end_id = sum(state_cardis(1:ii-1)) + state_cardis(ii);
                obj.ctype{ii}.F = obj.F(start_id:end_id, start_id:end_id, :);
                obj.ctype{ii}.Q = obj.Q(start_id:end_id, start_id:end_id, :);
                obj.ctype{ii}.mu0 = obj.mu0(start_id:end_id, 1, :);
                obj.ctype{ii}.Q0 = obj.Q0(start_id:end_id, start_id:end_id, :);
                obj.ctype{ii}.G = obj.G(1, start_id:end_id, :);
                obj.ctype{ii}.R = obj.R;
                %                 obj.ctype{ii}.y = obj.y;
                obj.ctype{ii}.Fs = obj.Fs;
            end
        end
        
        function [obj_array] = getarray(obj) % expand a model with multiple parameters into an array of ssm objects
            % make sure the number of alternative models are consistent
            % across all properties before forming the array
            property_list = fieldnames(obj);
            alt_model_num = cellfun(@(x) size(obj.(x),3), property_list);
            assert(length(unique(alt_model_num(alt_model_num~=1)))<=1, 'The numbers of alternative models are inconsistent across properties.')
            % form the array of ssm objects
            update_idx = find(alt_model_num>1);
            for ii = 1:max(alt_model_num)
                temp_obj = obj;
                for j = 1:length(update_idx)
                    temp_property = obj.(property_list{update_idx(j)});
                    temp_obj.(property_list{update_idx(j)}) = temp_property(:,:,ii);
                end
                obj_array(ii) = temp_obj;
            end
        end
        
        function [obj_array, K, T] = prepare_obj(obj_array) % Prepare an object array and output model dimensions
            if length(obj_array)==1
                obj_array = getarray(obj_array);
            end
            
            % verify that the observed data are identical across the models
            check_observed_data(obj_array)
            
            % Model dimensions
            K = length(obj_array);
            T = size(obj_array(1).y,2); % observed data y should be row vectors 
        end
        
        %% KALMAN FILTERING METHODS
        % Kalman filtering and smoothing are indexed from t=0 -> t=T.
        function [x_t_n_all,P_t_n_all,P_t_tmin1_n_all,logL_all,x_t_t_all,P_t_t_all,K_t_all,...
                x_t_tmin1_all,P_t_tmin1_all,fy_t_interp_all] = par_kalman(obj_array, varargin) % Parallel processing wrapper for Kalman filtering and smoothing
            % Multivariate observation data y is supported.
            p = inputParser;
            addRequired (p,'obj_array',               @(x)true)
            addParameter(p,'method',       'kalman',  @ischar)
            addParameter(p,'R_weights',     nan    ,  @isnumeric)
            parse(p,obj_array,varargin{:});
            input_arguments = struct2cell(p.Results);
            input_flags = fieldnames(p.Results);
            eval(['[', sprintf('%s ', input_flags{:}), '] = deal(input_arguments{:});']);
            
            % Prepare object array and get model dimensions
            [obj_array, K, T] = prepare_obj(obj_array);
            
            % set up trellis variables as cell arrays to accomodate
            % different dimensions of state variables
            % Forward recursion
            x_t_tmin1_all = cell(K,1);
            P_t_tmin1_all = cell(K,1);
            K_t_all = cell(K,1);
            x_t_t_all = cell(K,1);
            P_t_t_all = cell(K,1);
            % Backward recursion
            x_t_n_all = cell(K,1);
            P_t_n_all = cell(K,1);
            P_t_tmin1_n_all = cell(K,1);
            % log likelihood
            logL_all = zeros(K,T); % (index 1 corresponds to t=1)
            % interpolated conditional density (only available using De Jong version)
            fy_t_interp_all = zeros(K,T); % (index 1 corresponds to t=1)
            
            if isnan(R_weights)
                R_weights = nan * ones(K,1);
            else
                R_weights = R_weights; %#ok<ASGSL,*NODEF>
            end
            
            % skip calculating interpolated density when not needed
            if nargout > 9; skip_interp = false; else; skip_interp = true; end
            
            switch method
                case 'kalman'
                    kalman_method = @kalman_filt_smooth;
                case 'dejong'
                    kalman_method = @kalman_filt_smooth_dejong;
            end
            
            % kalman filtering and smoothing ssm with parallel processing
            parfor m = 1:K
                % kalman_method automatically calls different functions
                if skip_interp
                    [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1] = kalman_method(obj_array(m),'R_weights',R_weights(m,:)); fy_t_interp = nan; %#ok<PFBNS>
                else
                    [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1, fy_t_interp] = kalman_method(obj_array(m),'R_weights',R_weights(m,:));
                end
                % add to the trellis cell variables
                % Forward recursion
                x_t_tmin1_all{m} = x_t_tmin1;
                P_t_tmin1_all{m} = P_t_tmin1;
                K_t_all{m}  = K_t;
                x_t_t_all{m}  = x_t_t;
                P_t_t_all{m}  = P_t_t;
                % Backward recursion
                x_t_n_all{m}  = x_t_n;
                P_t_n_all{m}  = P_t_n;
                P_t_tmin1_n_all{m}  = P_t_tmin1_n;
                % log likelihood
                logL_all(m,:) = logL;
                % interpolated conditional density (only available using De Jong version)
                assert(isreal(fy_t_interp), 'Conditional density is complex. Something went wrong during filtering and smoothing.')
                fy_t_interp_all(m,:) = fy_t_interp;
            end
        end
        
        function [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t,...
                x_t_tmin1,P_t_tmin1, fy_t_interp] = kalman_filt_smooth(obj, varargin) % Kalman filtering and smoothing on a single set of parameters
            % This is the classical Kalman filter and fixed-interval
            % smoother. The input arguments can be a ssm object, or they
            % can be a set of parameters directly fed to the function. To
            % provide parameters directly, set the first argument to empty
            % and provide ([], F, Q, mu0, Q0, G, R, y) in that order.
            % Multivariate observation data y is supported. It calls the
            % classical kalman function in djkalman repository. 
            
            p = inputParser;
            addRequired(p,'obj',               @(x)true)
            addOptional(p,'F',          nan,   @isnumeric)
            addOptional(p,'Q',          nan,   @isnumeric)
            addOptional(p,'mu0',        nan,   @isnumeric)
            addOptional(p,'Q0',         nan,   @isnumeric)
            addOptional(p,'G',          nan,   @isnumeric)
            addOptional(p,'R',          nan,   @isnumeric)
            addOptional(p,'y',          nan,   @isnumeric)
            addParameter(p,'R_weights', nan,   @isnumeric)
            parse(p,obj,varargin{:});
            input_arguments = struct2cell(p.Results);
            input_flags = fieldnames(p.Results);
            eval(['[', sprintf('%s ', input_flags{:}), '] = deal(input_arguments{:});']);
            
            if ~isempty(obj) % use parameters contained in the ssm object
                F = obj.F;
                Q = obj.Q;
                mu0 = obj.mu0;
                Q0 = obj.Q0;
                G = obj.G;
                R = obj.R;
                y = obj.y;
            end
            
            [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1, fy_t_interp] = kalman(F, Q, mu0, Q0, G, R, y, R_weights);
        end
        
        function [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t,...
                x_t_tmin1,P_t_tmin1, fy_t_interp] = kalman_filt_smooth_dejong(obj, varargin) % (De Jong version) Kalman filtering and smoothing on a single set of parameters
            % This is the [De Jong 1989] Kalman filter and fixed-interval
            % smoother. The input arguments can be a ssm object, or they
            % can be a set of parameters directly fed to the function. To
            % provide parameters directly, set the first argument to empty
            % and provide ([], F, Q, mu0, Q0, G, R, y) in that order.
            % Multivariate observation data y is supported. It calls the
            % De Jong kalman function in djkalman repository.
            %
            % Since De Jong Kalman filtering and smoothing are not defined
            % at t=0, we repeat the estimates at t=1 to extend to t=0.
            
            p = inputParser;
            addRequired(p,'obj',               @(x)true)
            addOptional(p,'F',          nan,   @isnumeric)
            addOptional(p,'Q',          nan,   @isnumeric)
            addOptional(p,'mu0',        nan,   @isnumeric)
            addOptional(p,'Q0',         nan,   @isnumeric)
            addOptional(p,'G',          nan,   @isnumeric)
            addOptional(p,'R',          nan,   @isnumeric)
            addOptional(p,'y',          nan,   @isnumeric)
            addParameter(p,'R_weights', nan,   @isnumeric)
            parse(p,obj,varargin{:});
            input_arguments = struct2cell(p.Results);
            input_flags = fieldnames(p.Results);
            eval(['[', sprintf('%s ', input_flags{:}), '] = deal(input_arguments{:});']);
            
            if ~isempty(obj) % use parameters contained in the ssm object
                F = obj.F;
                Q = obj.Q;
                mu0 = obj.mu0;
                Q0 = obj.Q0;
                G = obj.G;
                R = obj.R;
                y = obj.y;
            end
            
            [x_t_n,P_t_n, P_t_tmin1_n, logL, x_t_t,P_t_t, K_t, x_t_tmin1,P_t_tmin1, fy_t_interp] = djkalman(F, Q, mu0, Q0, G, R, y, R_weights);
        end
        
        %% TRADITIONAL SWITCHING SSM METHODS
        function [Mprob, fy_t] = switching(obj_array, varargin) % Compute probabilities of switching state-space models
            % This is the main function to perform switching state-space
            % model inference using traditional algorithms. There are
            % multiple approaches, each associated with different input
            % parameter options.
            
            p = inputParser;
            addRequired (p,'obj_array',             @(x)true)
            addParameter(p,'method',      '1991',   @ischar)
            addParameter(p,'dwellp',      0.99,     @isnumeric)
            addParameter(p,'A',           nan,      @isnumeric)
            addParameter(p,'fixprior',    [0],      @iscolumn) %#ok<NBRAK>
            addParameter(p,'mimic1991',   false,    @islogical)
            addParameter(p,'HMMsmooth',   'none',   @ischar)
            addParameter(p,'futuresteps', 0,        @isnumeric)
            parse(p,obj_array,varargin{:});
            input_arguments = struct2cell(p.Results);
            input_flags = fieldnames(p.Results);
            eval(['[', sprintf('%s ', input_flags{:}), '] = deal(input_arguments{:});']);
            
            % Prepare object array and get model dimensions
            [obj_array, K, T] = prepare_obj(obj_array);
            
            % ============================================================
            % Choose among the various state-space switching methods:
            %   - 'static' is similar to 'a parp'
            %   - 'gpb1'   is similar to '1991'
            %   - 'gpb2'   is similar to 'IMM'
            % ============================================================
            switch method
                case {'static'}
                    % ----------------------------------------------------
                    % This is the naive static multiple model derived using
                    % a direct application of Bayes rule. It essentially
                    % uses predicted conditional density and runs an update
                    % on the prior for each time step. This method is
                    % obsolete now given the later development of dynamic
                    % methods, however with an ad-hoc modification of lower
                    % bounding model probability, it is still used in the
                    % literature.
                    %
                    % Reference:
                    % Fabri, S., & Kadirkamanathan, V. (2001). Functional
                    % adaptive control: an intelligent systems approach.
                    % Springer Science & Business Media.
                    % ----------------------------------------------------
                    
                    % run Kalman filtering on the static models in parallel
                    [~,~,~,~,~,~,~,x_t_tmin1_all,P_t_tmin1_all] = par_kalman(obj_array);
                    
                    % compute the conditional density (also called
                    % predicted likelihood function of y)
                    fy_t = compute_fy_t(obj_array, x_t_tmin1_all, P_t_tmin1_all);
                    
                    % initialize posterior model probability
                    Mprob = zeros(K,T+1); % (index 1 corresponds to t=0)
                    Mprob(:,1) = ones(1,K)/K;
                    
                    % iterate through the time steps
                    for ii = 2:T+1 % t=1 -> t=T
                        Mprob(:,ii) = fy_t(:,ii-1) .* Mprob(:,ii-1) ./ sum(fy_t(:,ii-1) .* Mprob(:,ii-1)); % direct application of Bayes rule
                        % ad-hoc solution of lower bounding model probability
                        lower_bound = 10^-2;
                        if any(Mprob(:,ii) < lower_bound)
                            Mprob(Mprob(:,ii) < lower_bound,ii) = lower_bound; % lower bound
                            Mprob(:,ii) = Mprob(:,ii) ./ sum(Mprob(:,ii)); % re-normalize
                        end
                    end
                    
                case {'gpb1'}
                    % ----------------------------------------------------
                    % This implements the generalized pseudo-Bayesian
                    % estimator of switching multiple models. The name
                    % comes from the fact that we combine at the end of
                    % each filtering + update step the K separate Gaussian
                    % distributions into a single approximating Gaussian.
                    %
                    % The merging of Gaussian is simply weighted averaging
                    % of the first two moments of the filtered state
                    % estimates using a pseudo-Bayesian update rule. It is
                    % not strictly Bayesian because the prior involves an
                    % averaging over transition matrix of discrete HMM.
                    %
                    % Note that this method is intimately related to the
                    % S&S 1991 approach. The only difference is that S&S
                    % 1991 takes the weighted average of innovations while
                    % gpb1 takes the weighted average of filtered state
                    % estimates. The former is exact solution derived under
                    % the assumption that only the observation matrices
                    % switch, while the latter is approximating a mixture
                    % of K Gaussians by merging their first two moments.
                    %
                    % If K models have the same model parameters (1991),
                    % at every iteration, the two methods produce the same
                    % first moment for the merged filtered estimate, but
                    % gpb1 has an additional covariance matrix coming from
                    % the deviation of mean of each of K Gaussians from the
                    % mean of the merged Gaussian. Similarly, one can use
                    % the 1991 method to implement gpb1 by using augmented
                    % hidden states and making F and Q block diagonal.
                    %
                    % Reference:
                    % Gordon, K., & Smith, A. F. M. (1990). Modeling and
                    % monitoring biomedical time series. Journal of the
                    % American Statistical Association, 85(410), 328-337.
                    % ----------------------------------------------------
                    
                    % gpb1 requires all the models to have the same
                    % dimensions of the hidden states in order to merge
                    % Gaussians - this is verified here
                    obj = obj_array(1);
                    for m = 2:K
                        assert(length(obj.mu0) == size(obj_array(m).G,2), 'Model hidden state dimensions are inconsistent.')
                    end
                    
                    p = length(obj.mu0);
                    q = 1; % y is one dimensional for now
                    
                    % transition matrix that is assumed to be known
                    if isnan(A)
                        A = abs(eye(K) - ones(K,K)*(1-dwellp)/(K-1));
                    end
                    
                    % Forward filtering trellis variables
                    x_t_tmin1 = zeros(p,K,T+1); % (index 1 corresponds to t=0, etc.)
                    P_t_tmin1 = zeros(p,p,K,T+1);
                    K_t = zeros(p,K,T+1);
                    x_t_t = zeros(p,T+1);
                    P_t_t = zeros(p,p,T+1);
                    Mprob = zeros(K,T+1);
                    fy_t = zeros(K,T); % conditional density of y given t=1...t-1 (index 1 corresponds to t=1)
                    
                    % initial model probability at time step 0
                    Mprob(:,1) = ones(1,K)/K;
                    % initialize x_0_0
                    x_t_t(:,1) = sum(cell2mat(arrayfun(@(x) obj_array(x).mu0 .* Mprob(x,1), 1:K, 'UniformOutput', false)), 2); % x_0_0
                    % initialize P_0_0
                    for m = 1:K
                        P_t_t(:,:,1) = P_t_t(:,:,1) + (obj_array(m).Q0 + (obj_array(m).mu0 - x_t_t(:,1))*(obj_array(m).mu0 - x_t_t(:,1))') .* Mprob(m,1);
                    end
                    % initialize log likelihood of filtering for each model
                    logL = zeros(K,1);
                    
                    for ii = 2:T+1 % forward iterations % t=1 -> t=T
                        % store for filtered estimates for each model to
                        % combine as one Gaussian at the end of the step
                        x_t_t_all = zeros(p,K);
                        P_t_t_all = zeros(p,p,K);
                        for m = 1:K % separate filter for each model
                            % current model parameters
                            F = obj_array(m).F;
                            Q = obj_array(m).Q;
                            G = obj_array(m).G;
                            R = obj_array(m).R;
                            
                            % one-step prediction
                            x_t_tmin1(:,m,ii) = F*x_t_t(:,ii-1);
                            P_t_tmin1(:,:,m,ii) = F*P_t_t(:,:,ii-1)*F' + Q;
                            
                            Sigma = G*P_t_tmin1(:,:,m,ii)*G' + R;
                            K_t(:,m,ii) = P_t_tmin1(:,:,m,ii)*G' / Sigma;
                            
                            % current time step univariate Gaussian pdf of y_t
                            fy_t(m,ii-1) = exp(-0.5 * (obj.y(ii-1) - G*x_t_tmin1(:,m,ii))^2 / Sigma) / sqrt(2*pi*Sigma);
                            
                            % innovation form of the log likelihood for the current model
                            logL(m) = logL(m) + log(fy_t(m,ii-1));
                            
                            % update equation to get filtered estimates for each model
                            x_t_t_all(:,m) = x_t_tmin1(:,m,ii) + K_t(:,m,ii)*(obj.y(ii-1) - G*x_t_tmin1(:,m,ii));
                            P_t_t_all(:,:,m) = P_t_tmin1(:,:,m,ii) - K_t(:,m,ii)*G*P_t_tmin1(:,:,m,ii);
                        end
                        
                        % update model probability from last step
                        Mprob_prior = A * Mprob(:,ii-1); % prior model probability
                        Mprob(:,ii) = fy_t(:,ii-1) .* Mprob_prior ./ sum(fy_t(:,ii-1) .* Mprob_prior); % filtered model probability
                        
                        % merge the K filtered Gaussians into one Gaussian
                        x_t_t(:,ii) = sum(x_t_t_all .* Mprob(:,ii)', 2);
                        for m = 1:K
                            P_t_t(:,:,ii) = P_t_t(:,:,ii) + (P_t_t_all(:,:,m) + (x_t_t_all(:,m) - x_t_t(:,ii))*(x_t_t_all(:,m) - x_t_t(:,ii))') .* Mprob(m,ii);
                        end
                    end
                    
                case {'gpb2'}
                    % ----------------------------------------------------
                    % This implements the second-order generalized pseudo
                    % Bayesian approach as described in Bar-Shalom & Li
                    % 1993. The method is an extension of gpb1 but differs
                    % by maintaining K different hidden state estimate
                    % distributions similar to the IMM method.
                    %
                    % The computations are similar to IMM but have K^2
                    % number of filters. Merging of K Gaussians into single
                    % approximating Gaussian is done for each of the model
                    % at the end of each time step. This is different from
                    % IMM, which mixes distributions at the beginning of
                    % every step which requires K filters instead of K^2.
                    %
                    % It can be shown that IMM of second order is identical
                    % to gpb2, therefore this method also serves as a
                    % second order extension of the IMM method.
                    %
                    % Reference:
                    % Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2004).
                    % Estimation with applications to tracking and
                    % navigation: theory algorithms and software. John
                    % Wiley & Sons.
                    % ----------------------------------------------------
                    
                    % gpb2 requires all the models to have the same
                    % dimensions of the hidden states in order to merge
                    % Gaussians - this is verified here
                    obj = obj_array(1);
                    for m = 2:K
                        assert(length(obj.mu0) == size(obj_array(m).G,2), 'Model hidden state dimensions are inconsistent.')
                    end
                    
                    p = length(obj.mu0);
                    q = 1; % y is one dimensional for now
                    
                    % transition matrix that is assumed to be known
                    if isnan(A)
                        A = abs(eye(K) - ones(K,K)*(1-dwellp)/(K-1));
                    end
                    
                    % Forward filtering trellis variables
                    x_t_tmin1 = zeros(p,K,K,T+1); % (index 1 corresponds to t=0, etc.)
                    P_t_tmin1 = zeros(p,p,K,K,T+1);
                    K_t = zeros(p,K,K,T+1);
                    x_t_t = zeros(p,K,T+1);
                    P_t_t = zeros(p,p,K,T+1);
                    Mprob = zeros(K,T+1);
                    fy_t = zeros(K,T); % conditional density of y given t=1...t-1 (index 1 corresponds to t=1)
                    fy_t_all = zeros(K,K,T); % used for model probability update equation
                    
                    % initial model probability at time step 0
                    Mprob(:,1) = ones(1,K)/K;
                    % initialize x_0_0 and P_0_0
                    for m = 1:K
                        x_t_t(:,m,1) = obj_array(m).mu0;
                        P_t_t(:,:,m,1) = obj_array(m).Q0;
                    end
                    % initialize log likelihood of filtering for each model
                    logL = zeros(K,1);
                    
                    for ii = 2:T+1 % forward iterations % t=1 -> t=T
                        for m = 1:K % separate filter for each model
                            % store for filtered estimates from each model to
                            % combine as one Gaussian at the end of the step
                            x_t_t_all = zeros(p,K);
                            P_t_t_all = zeros(p,p,K);
                            
                            % current filter parameters
                            F = obj_array(m).F; % in state m at current time step
                            Q = obj_array(m).Q;
                            G = obj_array(m).G;
                            R = obj_array(m).R;
                            
                            for j = 1:K % evaluate each model separately
                                % one-step prediction
                                x_t_tmin1(:,m,j,ii) = F*x_t_t(:,j,ii-1); % in state j at last time step
                                P_t_tmin1(:,:,m,j,ii) = F*P_t_t(:,:,j,ii-1)*F' + Q;
                                
                                Sigma = G*P_t_tmin1(:,:,m,j,ii)*G' + R;
                                K_t(:,m,j,ii) = P_t_tmin1(:,:,m,j,ii)*G' / Sigma;
                                
                                % current time step univariate Gaussian pdf of being in state j at t-1 step and switching to state m
                                fy_t_all(m,j,ii-1) = exp(-0.5 * (obj.y(ii-1) - G*x_t_tmin1(:,m,j,ii))^2 / Sigma) / sqrt(2*pi*Sigma);
                                
                                % update equation to get filtered estimates
                                x_t_t_all(:,j) = x_t_tmin1(:,m,j,ii) + K_t(:,m,j,ii)*(obj.y(ii-1) - G*x_t_tmin1(:,m,j,ii));
                                P_t_t_all(:,:,j) = P_t_tmin1(:,:,m,j,ii) - K_t(:,m,j,ii)*G*P_t_tmin1(:,:,m,j,ii);
                            end
                            
                            % compute the merging probability
                            Mprob_merge = fy_t_all(m,:,ii-1) .* A(m,:) .* Mprob(:,ii-1)' ./ sum(fy_t_all(m,:,ii-1) .* A(m,:) .* Mprob(:,ii-1)');
                            
                            % merge the K Gaussians into 1 Gaussian for each model
                            x_t_t(:,m,ii) = sum(x_t_t_all .* Mprob_merge, 2);
                            for j = 1:K
                                P_t_t(:,:,m,ii) = P_t_t(:,:,m,ii) + (P_t_t_all(:,:,j) + (x_t_t_all(:,j) - x_t_t(:,m,ii))*(x_t_t_all(:,j) - x_t_t(:,m,ii))') .* Mprob_merge(j);
                            end
                            
                            % weighted averaging of conditional density to get fy_t
                            fy_t(m,ii-1) = fy_t_all(m,:,ii-1) * Mprob_merge';
                            
                            % innovation form of the log likelihood for the current model
                            logL(m) = logL(m) + log(fy_t(m,ii-1));
                        end
                        
                        % update model probability from last step
                        Mprob(:,ii) = sum(fy_t_all(:,:,ii-1) .* A .* Mprob(:,ii-1)', 2) ./ sum(fy_t_all(:,:,ii-1) .* A .* Mprob(:,ii-1)', 'all');
                    end
                    
                case {'IMM'}
                    % ----------------------------------------------------
                    % This implements the interacting multiple model (IMM)
                    % approach as sketched out in Bar-Shalom & Li 1993. The
                    % method is distinguished by the key feature of mixing
                    % Gaussians at beginning of a step in Kalman filtering
                    % for each of the K models. Therefore this method
                    % maintains K different hidden state estimate
                    % distributions, which interact at the beginning of
                    % every step using mixing probabilities that are
                    % model probabilities at time step t-1 conditioned on
                    % being at a specific state at time step t. Note that
                    % this is different from the posterior model
                    % probabilities used for mixing filtered estimates
                    % (gpb1) or innovations (S&S 1991), which occur at the
                    % end of a time step in Kalman filtering.
                    %
                    % Reference:
                    % Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2004).
                    % Estimation with applications to tracking and
                    % navigation: theory algorithms and software. John
                    % Wiley & Sons.
                    % ----------------------------------------------------
                    
                    % IMM requires all the models to have the same
                    % dimensions of the hidden states in order to merge
                    % Gaussians - this is verified here
                    obj = obj_array(1);
                    for m = 2:K
                        assert(length(obj.mu0) == size(obj_array(m).G,2), 'Model hidden state dimensions are inconsistent.')
                    end
                    
                    p = length(obj.mu0);
                    q = 1; % y is one dimensional for now
                    
                    % transition matrix that is assumed to be known
                    if isnan(A)
                        A = abs(eye(K) - ones(K,K)*(1-dwellp)/(K-1));
                    end
                    
                    % Forward filtering trellis variables
                    x_t_tmin1 = zeros(p,K,T+1); % (index 1 corresponds to t=0, etc.)
                    P_t_tmin1 = zeros(p,p,K,T+1);
                    K_t = zeros(p,K,T+1);
                    x_t_t = zeros(p,K,T+1);
                    P_t_t = zeros(p,p,K,T+1);
                    Mprob = zeros(K,T+1);
                    fy_t = zeros(K,T); % conditional density of y given t=1...t-1 (index 1 corresponds to t=1)
                    
                    % initial model probability at time step 0
                    Mprob(:,1) = ones(1,K)/K;
                    % initialize x_0_0 and P_0_0
                    for m = 1:K
                        x_t_t(:,m,1) = obj_array(m).mu0;
                        P_t_t(:,:,m,1) = obj_array(m).Q0;
                    end
                    % initialize log likelihood of filtering for each model
                    logL = zeros(K,1);
                    
                    for ii = 2:T+1 % forward iterations % t=1 -> t=T
                        % compute the mixing probability
                        Mprob_mix = A .* Mprob(:,ii-1)' ./ sum(A .* Mprob(:,ii-1)', 2); % each row is the mixing probability of other models for a model m
                        
                        for m = 1:K % separate filter for each model
                            % current model parameters
                            F = obj_array(m).F;
                            Q = obj_array(m).Q;
                            G = obj_array(m).G;
                            R = obj_array(m).R;
                            
                            % mix the last step hidden states from K models
                            x0_t_t = sum(x_t_t(:,:,ii-1) .* Mprob_mix(m,:), 2);
                            P0_t_t = zeros(p,p);
                            for l = 1:K
                                P0_t_t = P0_t_t + (P_t_t(:,:,l,ii-1) + (x_t_t(:,l,ii-1) - x0_t_t)*(x_t_t(:,l,ii-1) - x0_t_t)') .* Mprob_mix(m,l);
                            end
                            
                            % one-step prediction using the merged Gaussian
                            x_t_tmin1(:,m,ii) = F*x0_t_t;
                            P_t_tmin1(:,:,m,ii) = F*P0_t_t*F' + Q;
                            
                            Sigma = G*P_t_tmin1(:,:,m,ii)*G' + R;
                            K_t(:,m,ii) = P_t_tmin1(:,:,m,ii)*G' / Sigma;
                            
                            % current time step univariate Gaussian pdf of y_t
                            fy_t(m,ii-1) = exp(-0.5 * (obj.y(ii-1) - G*x_t_tmin1(:,m,ii))^2 / Sigma) / sqrt(2*pi*Sigma);
                            
                            % innovation form of the log likelihood for the current model
                            logL(m) = logL(m) + log(fy_t(m,ii-1));
                            
                            % update equation to get filtered estimates for each model
                            x_t_t(:,m,ii) = x_t_tmin1(:,m,ii) + K_t(:,m,ii)*(obj.y(ii-1) - G*x_t_tmin1(:,m,ii));
                            P_t_t(:,:,m,ii) = P_t_tmin1(:,:,m,ii) - K_t(:,m,ii)*G*P_t_tmin1(:,:,m,ii);
                        end
                        
                        % update model probability from last step
                        Mprob(:,ii) = fy_t(:,ii-1) .* sum(A .* Mprob(:,ii-1)', 2) ./ sum(fy_t(:,ii-1) .* sum(A .* Mprob(:,ii-1)', 2));
                    end
                    
                case {'1991', '1991 pseu'}
                    % ----------------------------------------------------
                    % This implements the original Shumway & Stoffer 1991
                    % modified Kalman filtering approach for switching
                    % state space models. A distinguishing feature of this
                    % method from the gpb1 method is that only the
                    % observation matrices switch in S&S 1991, therefore
                    % the Kalman filtering equations are exact by weighting
                    % the innovations instead of merging Gaussians as
                    % approximation. Thus unless using pseudo-EM to learn
                    % the SSM parameters (in S&S 1991 Appendix), this
                    % method does not involve approximating Gaussians like
                    % all the other methods here.
                    %
                    % Reference:
                    % Shumway, R. H., & Stoffer, D. S. (1991). Dynamic
                    % linear models with switching. Journal of the American
                    % Statistical Association, 86(415), 763-769.
                    %
                    % The '1991 pseu' option stands for pseudo-smoothing,
                    % and it is a heuristic modification of the 1991 method
                    % by replacing the conditioanl density of y with a
                    % summation of the conditional density of y under each
                    % alternative model when filtering into the future for
                    % [futuresteps] steps. This is not strictly derivable
                    % under Bayes rule and therefore only serves as a
                    % heuristic solution.
                    % ----------------------------------------------------
                    
                    % if using the original S&S method, there is no look
                    % ahead for future steps
                    if strcmp(method,'1991'); futuresteps = 0; end
                    
                    % with the Shumway & Stoffer 1991 approach, all the
                    % models must have the same properties except the
                    % observation matrices - this is verified here
                    obj = obj_array(1);
                    for m = 2:K
                        assert(all(obj.F == obj_array(m).F,'all'))
                        assert(all(obj.Q == obj_array(m).Q,'all'))
                        assert(all(obj.mu0 == obj_array(m).mu0,'all'))
                        assert(all(obj.Q0 == obj_array(m).Q0,'all'))
                        assert(all(obj.R == obj_array(m).R,'all'))
                        assert(length(obj.mu0) == size(obj_array(m).G,2))
                    end
                    
                    p = length(obj.mu0);
                    q = 1; % y is one dimensional for now
                    
                    % initial model probability at time step 0
                    Mprob_init = ones(1,K)/K;
                    
                    % transition matrix to introduce smoothness in filtered model probability
                    if isnan(A)
                        A = abs(eye(K) - ones(K,K)*(1-dwellp)/(K-1));
                    end
                    
                    % Forward filtering trellis variables
                    x_t_tmin1 = zeros(p,T+1); % (index 1 corresponds to t=0, etc.)
                    P_t_tmin1 = zeros(p,p,T+1);
                    K_t = zeros(p,K,T+1);
                    x_t_t = zeros(p,T+1);
                    P_t_t = zeros(p,p,T+1);
                    Mprob = zeros(K,T+1);
                    fy_t = zeros(K,T); % conditional density of y given t=1...t-1+futuresteps (index 1 corresponds to t=1)
                    
                    % initialize hidden states at t=0
                    x_t_t(:,1) = obj.mu0; % x_0_0
                    P_t_t(:,:,1) = obj.Q0; % P_0_0
                    Mprob(:,1) = Mprob_init; % pi_0_0
                    logL = zeros(K,1);
                    
                    for ii = 2:T+1 % forward iterations % t=1 -> t=T
                        % one-step prediction
                        x_t_tmin1(:,ii) = obj.F*x_t_t(:,ii-1);
                        P_t_tmin1(:,:,ii) = obj.F*P_t_t(:,:,ii-1)*obj.F' + obj.Q;
                        
                        % calculate filtered model probabilities (Mprob_t_t)
                        fy_t_tmin1 = zeros(K, min(T+2-ii,futuresteps+1));
                        for m = 1:K % single joint filter by merging innovations from K models
                            G = obj_array(m).G; %#ok<*PROPLC> % current model observation matrix
                            Sigma = G*P_t_tmin1(:,:,ii)*G' + obj.R;
                            K_t(:,m,ii) = P_t_tmin1(:,:,ii)*G' / Sigma;
                            
                            % current time step univariate Gaussian pdf of y_t
                            fy_t_tmin1(m,1) = exp(-0.5 * (obj.y(ii-1) - G*x_t_tmin1(:,ii))^2 / Sigma) / sqrt(2*pi*Sigma);
                            
                            % innovation form of the log likelihood for the current model
                            logL(m) = logL(m) + log(fy_t_tmin1(m,1));
                            
                            % filter into the future if futuresteps > 0
                            temp_x_t_tmin1 = zeros(p,futuresteps+1); temp_x_t_tmin1(:,1) = x_t_tmin1(:,ii);
                            temp_P_t_tmin1 = zeros(p,p,futuresteps+1); temp_P_t_tmin1(:,:,1) = P_t_tmin1(:,:,ii);
                            temp_K_t = zeros(p,futuresteps+1); temp_K_t(:,1) = K_t(:,m,ii);
                            temp_x_t_t = zeros(p,futuresteps+1);
                            temp_P_t_t = zeros(p,p,futuresteps+1);
                            for k = 2:min(T+2-ii,futuresteps+1) % we will filter forward in the current model for [futuresteps] steps
                                % finish off the filtering for the last time step
                                temp_x_t_t(:,k-1) = temp_x_t_tmin1(:,k-1) + temp_K_t(:,k-1)*(obj.y(ii-1+k-2) - G*temp_x_t_tmin1(:,k-1));
                                temp_P_t_t(:,:,k-1) = temp_P_t_tmin1(:,:,k-1) - temp_K_t(:,k-1)*G*temp_P_t_tmin1(:,:,k-1);
                                % then we compute the prediction for the current step
                                temp_x_t_tmin1(:,k) = obj.F*temp_x_t_t(:,k-1);
                                temp_P_t_tmin1(:,:,k) = obj.F*temp_P_t_t(:,:,k-1)*obj.F' + obj.Q;
                                temp_Sigma = G*temp_P_t_tmin1(:,:,k)*G' + obj.R;
                                temp_K_t(:,k) = temp_P_t_tmin1(:,:,k)*G' / temp_Sigma;
                                % now calculate the conditional density of y_t for the current step
                                fy_t_tmin1(m,k) = exp(-0.5 * (obj.y(ii-1+k-1) - G*temp_x_t_tmin1(:,k))^2 / temp_Sigma) / sqrt(2*pi*temp_Sigma);
                            end
                        end
                        fy_t_tmin1 = sum(fy_t_tmin1,2); % sum conditional density for filtering [futuresteps] steps with all models
                        fy_t(:,ii-1) = fy_t_tmin1;
                        Mprob_prior = A * Mprob(:,ii-1); % prior model probability
                        Mprob(:,ii) = fy_t_tmin1 .* Mprob_prior ./ sum(fy_t_tmin1 .* Mprob_prior); % filtered model probability
                        
                        % filtering by merging innovations using filtered model probability
                        x_update = zeros(p,1);
                        P_update = zeros(p);
                        for m = 1:K
                            G = obj_array(m).G;
                            x_update = x_update + Mprob(m,ii)*K_t(:,m,ii)*(obj.y(ii-1) - G*x_t_tmin1(:,ii));
                            P_update = P_update + Mprob(m,ii)*(P_t_tmin1(:,:,ii) - K_t(:,m,ii)*G*P_t_tmin1(:,:,ii));
                        end
                        x_t_t(:,ii) = x_t_tmin1(:,ii) + x_update;
                        P_t_t(:,:,ii) = P_update;
                    end
                    
                case {'a parp', 'a pari', 'ab parp', 'ab pari', 'a pars', 'ab pars'}
                    % ----------------------------------------------------
                    % "Parallel model" approach treats the Gaussian process
                    % hidden states as known from the Kalman filtering and
                    % smoothing. The rest is then just the classical
                    % alpha-beta forward backward algorithm on a discrete
                    % state Markov Chain (hidden Markov model) that
                    % captures the switching state variable.
                    %
                    % The key assumption here is that after running
                    % Kalman filtering and smoothing of the parallel
                    % models, we use the predicted or interpolated density
                    % as approximations to the observation probability in
                    % the hidden Markov model and use forward backward
                    % algorithm to complete *approximate* inference.
                    %
                    % Two approximations are at play:
                    % 1) Approximate conditional independence across y
                    % 2) Approximate density with static history models
                    %
                    % Abbreviation:
                    % a parp  = filtered MC with predicted Gaussian states (conditional independence is not necessary)
                    % a pari  = filtered MC with interpolated Gaussian states (still non-causal)
                    % ab parp = smoothed MC with predicted Gaussian states
                    % ab pari = smoothed MC with interpolated Gaussian states (best option)
                    %
                    % a pars and ab pars are testing methods - do not use!
                    % ----------------------------------------------------
                    
                    % ------- SSM
                    % first run parallel kalman filtering and smoothing on
                    % an array of ssm objects, then compute the conditional
                    % density of y at each time step.
                    %
                    % Note that all the choices of hidden state estimates
                    % lead to approximations of fy_j(t|t-1) in the S&S 1991
                    % approach if alpha-filtering is used for HMM
                    if contains(method, 'parp') % predicted conditional density
                        [~,~,~,~,~,~,~,x_t_tmin1_all,P_t_tmin1_all] = par_kalman(obj_array);
                        fy_t = compute_fy_t(obj_array, x_t_tmin1_all, P_t_tmin1_all);
                    elseif contains(method, 'pars') % smoothed conditional density
                        [x_t_n_all,P_t_n_all] = par_kalman(obj_array);
                        fy_t = compute_fy_t(obj_array, x_t_n_all, P_t_n_all);
                    elseif contains(method, 'pari') % if using interpolated state estimates with De Jong filtering, conditional density is included in output
                        [~,~,~,~,~,~,~,~,~,fy_t] = par_kalman(obj_array, 'method', 'dejong');
                    end
                    
                    % ------- HHM
                    % force mimic1991 to be false if doing smoothing on HMM
                    if contains(method, 'ab') && mimic1991
                        mimic1991 = false; warning('Smoothing HMM cannot mimic S&S 1991.');
                    end
                    if mimic1991
                        HMMsmooth = 'none';
                        % If we try to use fy_t as an approximation of
                        % fy_j(t|t-1) in S&S 1991, we need a slightly
                        % different initialization of the alpha state at
                        % t=1, therefore we can't use the dedicated
                        % implementation of forward-backward algorithm. We
                        % implement the associated alpha filtering below.
                        
                        % transition matrix of the discrete state HMM for
                        % switching models
                        if all(fixprior==0)
                            if isnan(A)
                                A = abs(eye(K) - ones(K,K)*(1-dwellp)/(K-1)); % assume transition matrix is known
                            end
                        else % a fixed predictor/prior lacks temporal continuity, not MC anymore
                            predictor = fixprior; % this is provided as an option to mimic S&S 1991
                        end
                        
                        norm_a = zeros(K,T+1); % (index 1 corresponds to t=0, etc.)
                        norm_a(:,1) = ones(1,K)/K; % t=0, equivalent to Mprob_init in S&S 1991 approach
                        for ii = 2:T+1 % t=1 -> t=T
                            if all(fixprior==0)
                                predictor = A * norm_a(:,ii-1); % equivalent to Mprob_prior, prior model probability in S&S 1991
                            end
                            norm_a(:,ii) = fy_t(:,ii-1) .* predictor ./ sum(fy_t(:,ii-1) .* predictor);
                        end
                        
                        % this answer is equivalent to the original S&S
                        % 1991 approach using "optimal" hidden state
                        % estimates to approximate fy_j(t|t-1) and doing
                        % the same filtering estimates for discrete
                        % switching states
                        Mprob = norm_a(:,2:end); % Truncate the first time point to output vectors t=1 -> t=T
                    else
                        % Invoke proper forward-backward algorithm with
                        % more conventional initializations
                        method_str = strsplit(method);
                        HMMsmooth = method_str{1};
                    end
            end
            
            % Truncate the first time point to output vectors t=1 -> t=T
            if ~contains(method, 'par'); Mprob(:,1) = []; end
            
            % Apply forward-backward algorithm to fy_t if specified
            if isnan(A)
                A = abs(eye(K) - ones(K,K)*(1-dwellp)/(K-1));
            end
            p1 = ones(1,K)/K;
            switch HMMsmooth
                case 'a' % alpha filtering
                    Mprob = forward_backward(A, fy_t, p1);
                case 'ab' % alpha-beta smoothing
                    [~, Mprob] = forward_backward(A, fy_t, p1);
            end
        end
        
        function [fy_t] = compute_fy_t(obj_array, x_t_all, P_t_all) % Compute the univariate conditional density of Gaussian P(y_t|x_t)
            % Model dimensions
            K = length(obj_array);
            T = size(obj_array(1).y,2);
            fy_t = zeros(K,T); % (index 1 corresponds to t=1)
            
            parfor m = 1:K
                x_t = x_t_all{m}(:,2:end); % (index 1 corresponds to t=0, etc.)
                P_t = P_t_all{m}(:,:,2:end);
                for ii = 1:T % t=1 -> t=T
                    Sigma = obj_array(m).G*P_t(:,:,ii)*obj_array(m).G' + obj_array(m).R;
                    fy_t(m,ii) = exp(-0.5 * (obj_array(m).y(ii) - obj_array(m).G*x_t(:,ii))^2 / Sigma) / sqrt(2*pi*Sigma);
                end
            end
        end
        
        %% VARIATIONAL BAYESIAN SWITCHING SSM METHODS
        function [ht_m, qt_m, obj_array, A, VB_iter, logL_bound, em_params] = VBlearn_original(obj_array, varargin) % Original VB learning of segmental SSMs with Markov switching
            % ----------------------------------------------------
            % This is an implementation of Ghahramani & Hinton 2000
            % learning algorithm with variational bayesian approximation.
            % This algorithm has a generalized EM structure where the E
            % step will lower-bound the posterior distribution of the
            % hidden variables (both continuous hidden state and discrete
            % switching variables), and the M step will re-estimate the
            % state-space model (linear dynamical system) and hidden Markov
            % model parameters using Shumway & Stoffer 1982 EM on the SSM
            % and Baum-Welch algorithm on the HMM.
            %
            % This implementation follows exactly the algorithm described
            % in Ghahramani & Hinton 2000.
            %
            % Figure 5: Learning algorithm for switching state-space models
            % [E step] - Repeat until convergence of KL(Q||P):
            %       E.1 Compute qt_m from the prediction error of
            %       state-space model m on observation Y_t
            %       E.2 Compute ht_m using the forward-backward algorithm
            %       on the HMM, with observation probabilities qt_m
            %       E.3 For m = 1 to M (we use K instead of M)
            %           Run Kalm smoothing recursions, using the data
            %           weighted by ht_m
            %
            % [M step]
            %       M.1 Re-estimate parameters for each state-space model
            %       using the data weighted by ht_m
            %       M.2 Re-estimate parameters for the switching process
            %       using Baum-Welch update equations.
            %
            % Reference:
            % Ghahramani, Z., & Hinton, G. E. (2000). Variational learning
            % for switching state-space models. Neural computation, 12(4),
            % 831-864.
            % ----------------------------------------------------
            p = inputParser; run_onset = tic;
            addRequired (p,'obj_array',             @(x)true)
            addParameter(p,'dwellp',      0.99,     @isnumeric)
            addParameter(p,'A',           nan,      @isnumeric)
            addParameter(p,'maxE_iter',   100,      @isnumeric)
            addParameter(p,'maxVB_iter',  100,      @isnumeric)
            addParameter(p,'ht_thresh',   10^-6,    @isnumeric)
            addParameter(p,'plot_Estep',  false,    @islogical)
            addParameter(p,'verbose',     true,     @islogical)
            addParameter(p,'warm_start',  true,     @islogical)
            parse(p,obj_array,varargin{:});
            input_arguments = struct2cell(p.Results);
            input_flags = fieldnames(p.Results);
            eval(['[', sprintf('%s ', input_flags{:}), '] = deal(input_arguments{:});']);
            
            % prepare object array and get the number of alternative models
            [obj_array, K] = prepare_obj(obj_array);
            
            % ============================================================
            %                       INITIALIZATION
            % ============================================================
            % initial guess of transition probability matrix for HMM
            if isnan(A)
                A = abs(eye(K) - ones(K,K)*(1-dwellp)/(K-1)); % initial A (updated in next M step)
            end
            
            % initial HMM state prior at time step t=1
            p1 = ones(1,K)/K; % initial switching state prior (fixed to be uniform)
            
            % initialize VB iteration parameters
            logL_bound = zeros(1);
            delta_ht_m = inf;
            ht_m = ones(K, size(obj_array(1).y,2))*1/K; % initialize with equal responsibilities for the models
            VB_iter = 0;
                        
            % create a cell_array to store parameters during EM iterations
            em_params = {{obj_array, A}};
            
            while delta_ht_m > ht_thresh && VB_iter < maxVB_iter
                % VB iteration counting variables
                VB_iter = VB_iter + 1;
                last_ht_m = ht_m;
                
                % ============================================================
                %                           E STEP
                % ============================================================
                % initialize the fixed point iteration
                delta_qt_m = inf;
                qt_m = inf;
                E_iter = 0;
                T = 100; % deterministic annealing initial temperature parameter
                % initialize E step using converged h_t_m from the last EM iteration as a warm start
                if ~warm_start
                    ht_m = ones(K, size(obj_array(1).y,2))*1/K; % reset to equal responsibilities
                end
                [x_t_n_all, P_t_n_all] = par_kalman(obj_array, 'method', 'kalman', 'R_weights', 1./ht_m); % smoothed estimates
                
                while delta_qt_m > 10^-6 && E_iter < maxE_iter % minimize KL(Q||P)
                    % store qt_m to check for convergence of E step
                    E_iter = E_iter + 1;
                    last_qt_m = qt_m;
                    
                    % E.1 Compute qt_m using smoothed estimates
                    [qt_m, gt_m] = compute_qt_m(obj_array, x_t_n_all, P_t_n_all, T); % with deterministic annealing
                    
                    % E.2 Compute ht_m using forward-backward algorithm
                    [~, ht_m, ht_tmin1_m, logL_HMM] = forward_backward(A, qt_m, p1);
                    
                    % modify ht_m with deterministic annealing
                    ht_m = ht_m ./ T;
                    ht_tmin1_m = ht_tmin1_m ./ T;
                    T = T/2 + 1/2; % decreasing towards asymptote at T=1
                    
                    if plot_Estep
                        figure(5); plot(ht_m(1,:), 'LineWidth', 1); hold on; plot(ht_m(2,:), 'LineWidth', 1); hold off
                        title(['ht_m: iter ', num2str(E_iter)]); ylim([-0.1,1.1])
                        
                        figure(6); plot(qt_m(1,:)); hold on; plot(qt_m(2,:)); hold off
                        title(['qt_m: iter ', num2str(E_iter)])
                    end
                    
                    % E.3 Run Kalman smoothing recursions with weighted data
                    [x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_SSM] = par_kalman(obj_array, 'method', 'kalman', 'R_weights', 1./ht_m); % smoothed estimates
                    
                    % Check for convergence using the average change in elements of qt_m
                    delta_qt_m = mean(abs(last_qt_m - qt_m), 'all');
                end
                
                % Store the current bound on log likelihood
                logL_bound(VB_iter) = compute_logL_bound(obj_array, logL_HMM, logL_SSM, ht_m, gt_m);
                
                % ============================================================
                %                           M STEP
                % ============================================================
                % M.1 ML estimates of SSM parameters with weighted data
                obj_array = m1_step(obj_array, x_t_n_all, P_t_n_all, P_t_tmin1_n_all, ht_m, {}, true, {});
                
                % M.2 ML estimate of transition matrix of HMM
                A = m2_step(ht_m, ht_tmin1_m, obj_array(1)); % pass in a dummy object that isn't used
                
                % store the updated parameters
                em_params{length(em_params)+1} = {obj_array, A};
                
                % Check for convergence using the average change in elements of converged ht_m
                delta_ht_m = mean(abs(last_ht_m - ht_m), 'all');
                if verbose; disp(['delta_ht_m = ', num2str(delta_ht_m)]); end
            end
            
            if verbose; disp(['Time taken: ', num2str(toc(run_onset)), ' sec, VB iterations: ', num2str(VB_iter), '.']); end
        end
        
        function [ht_m, qt_m, obj_array, A, VB_iter, ht_m_soft, ht_m_hard, logL_bound, x_t_n_all, P_t_n_all, em_params] = VBlearn(obj_array, varargin) % Improved VB learning of segmental SSMs with Markov switching
            % ----------------------------------------------------
            % This is a new implementation of Ghahramani & Hinton 2000
            % learning algorithm with variational bayesian approximation.
            %
            % This implementation introduces various >>> improvements <<<
            % throughout the VB learning iterations over the original one:
            %   - smart initialization of E-step with interpolated density
            %   and avoid deterministic annealing
            %   - semi-hard (converged ht), soft (smoothed estimates of
            %   discrete HMM using interpolated density), and hard
            %   (Viterbi) segmentations are available
            %   - parallelized Kalman filtering and smoothing in De Jong
            %   version that avoids inverting conditional state noise
            %   covariance P_t_tmin1 and handles degenerate matrices
            %   automatically
            %   - parameters for shared components can be jointly estimated
            %   - MAP estimation instead of ML to incorporate priors
            %
            % Reference:
            % Ghahramani, Z., & Hinton, G. E. (2000). Variational learning
            % for switching state-space models. Neural computation, 12(4),
            % 831-864.
            % ----------------------------------------------------
            p = inputParser; run_onset = tic;
            addRequired (p,'obj_array',             @(x)true)
            addParameter(p,'dwellp',      0.99,     @isnumeric)
            addParameter(p,'A',           nan,      @isnumeric)
            addParameter(p,'maxE_iter',   100,      @isnumeric)
            addParameter(p,'maxVB_iter',  100,      @isnumeric)
            addParameter(p,'ht_thresh',   10^-6,    @isnumeric)
            addParameter(p,'plot_Estep',  false,    @islogical)
            addParameter(p,'verbose',     true,     @islogical)
            addParameter(p,'norm_qt_m',   false,    @islogical)
            addParameter(p,'shared_R',    true,     @(x)islogical(x)||isnan(x))
            addParameter(p,'shared_ctype',{},       @iscell)
            addParameter(p,'priors_all',  {},       @iscell)
            parse(p,obj_array,varargin{:});
            input_arguments = struct2cell(p.Results); %#ok<*NASGU>
            input_flags = fieldnames(p.Results);
            eval(['[', sprintf('%s ', input_flags{:}), '] = deal(input_arguments{:});']);
            
            % prepare object array and get the number of alternative models
            [obj_array, K] = prepare_obj(obj_array);
            
            % ============================================================
            %                       INITIALIZATION
            % ============================================================
            % initialize priors for the M1 step
            if isempty(priors_all)
                priors_all = arrayfun(@(x) initialize_priors((x.fill_components)), obj_array, 'UniformOutput', false);
            elseif isnan(priors_all{1})
                priors_all = {}; % use ML estimates in M.1 step
            end
            
            % initial guess of transition probability matrix for HMM
            if isnan(A)
                A = abs(eye(K) - ones(K,K)*(1-dwellp)/(K-1)); % initial A (updated in next M step)
            end
            
            % initial HMM state prior at time step t=1
            p1 = ones(1,K)/K; % initial switching state prior (fixed to be uniform)
            
            % initialize VB iteration parameters
            logL_bound = zeros(1);
            delta_ht_m = inf;
            ht_m = inf;
            VB_iter = 0;
            
            % create a cell_array to store parameters during EM iterations
            em_params = {{obj_array, A}};
            
            while delta_ht_m > ht_thresh && VB_iter < maxVB_iter
                % VB iteration counting variables
                VB_iter = VB_iter + 1;
                last_ht_m = ht_m;
                
                % ============================================================
                %                           E STEP
                % ============================================================
                % initialize the fixed point iteration
                delta_qt_m = inf;
                E_iter = 0;
                [~,P_t_n_all,~,~,~,~,~,~,~,qt_m] = par_kalman(obj_array, 'method', 'dejong'); % use the interpolated density to initialize qt_m in E.1
                
                % Normalize the scales of interpolated density across models
                if norm_qt_m
                    sum_density_weights = sum(qt_m, 2);
                    qt_m = qt_m .* (min(sum_density_weights) ./ sum_density_weights);
                    
                    P_t_n_weights = zeros(K, 1);
                    for m = 1:K
                        P_t_n_weights(m) = obj_array(m).G * mode(P_t_n_all{m}, 3) * obj_array(m).G';
                    end
                    qt_m = qt_m .* (min(P_t_n_weights) ./ P_t_n_weights);
                end
                
                while delta_qt_m > 10^-6 && E_iter < maxE_iter % minimize KL(Q||P)
                    % store qt_m to check for convergence of E step
                    E_iter = E_iter + 1;
                    last_qt_m = qt_m;
                    
                    % E.2 Compute ht_m using forward-backward algorithm
                    [~, ht_m] = forward_backward(A, qt_m, p1);
                    
                    if plot_Estep
                        figure(5); plot(ht_m(1,:), 'LineWidth', 1); hold on; plot(ht_m(2,:), 'LineWidth', 1); hold off
                        title(['ht_m: iter ', num2str(E_iter)]); ylim([-0.1,1.1])
                        
                        figure(6); plot(qt_m(1,:)); hold on; plot(qt_m(2,:)); hold off
                        title(['qt_m: iter ', num2str(E_iter)])
                    end
                    
                    % E.3 Run Kalman smoothing recursions with weighted data
                    [x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_SSM] = par_kalman(obj_array, 'method', 'dejong', 'R_weights', 1./ht_m); % smoothed estimates
                    
                    % E.1 Compute qt_m using smoothed estimates
                    [qt_m, gt_m] = compute_qt_m(obj_array, x_t_n_all, P_t_n_all);
                    
                    % Check for convergence using the average change in elements of qt_m
                    delta_qt_m = mean(abs(last_qt_m - qt_m), 'all');
                end
                
                % Run final forward-backward and weighted Kalman smoothing with converged qt_m
                [~, ht_m, ht_tmin1_m, logL_HMM] = forward_backward(A, qt_m, p1);
                [x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_SSM] = par_kalman(obj_array, 'method', 'dejong', 'R_weights', 1./ht_m);
                
                % Store the current bound on log likelihood
                logL_bound(VB_iter) = compute_logL_bound(obj_array, logL_HMM, logL_SSM, ht_m, gt_m);
                
                % ============================================================
                %                           M STEP
                % ============================================================
                % M.1 MAP estimates of SSM parameters with weighted data
                obj_array = m1_step(obj_array, x_t_n_all, P_t_n_all, P_t_tmin1_n_all, ht_m, priors_all, shared_R, shared_ctype);
                
                % M.2 ML estimate of transition matrix of HMM
                A = m2_step(ht_m, ht_tmin1_m, obj_array(1)); % pass in a dummy object that isn't used
                
                % store the updated parameters
                em_params{length(em_params)+1} = {obj_array, A};
                
                % Check for convergence using the average change in elements of converged ht_m
                delta_ht_m = mean(abs(last_ht_m - ht_m), 'all');
                if verbose; disp(['delta_ht_m = ', num2str(delta_ht_m)]); end
            end
            
            % support different segmentation methods (semi-hard, soft, hard)
            if nargout > 5
                [ht_m_soft, fy_t] = switching(obj_array, 'method', 'ab pari', 'A', A); % use parallel interpolated density for [soft] segmentation
                [~, ht_m_hard] = viterbi(A, fy_t, p1); % apply Viterbi on parallel interpolated density for [hard] segmentation
            end
            
            if verbose; disp(['Time taken: ', num2str(toc(run_onset)), ' sec, VB iterations: ', num2str(VB_iter), '.']); end
        end
        
        function [qt_m, gt_m] = compute_qt_m(obj_array, x_t_n_all, P_t_n_all, Temp) % Compute the unnormalized Gaussian density function of expected error for VB learning fixed point iteration
            % Reference:
            % Ghahramani, Z., & Hinton, G. E. (2000). Variational learning
            % for switching state-space models. Neural computation, 12(4),
            % 831-864.
            
            if nargin < 4
                Temp = 1; % temperature parameter for deterministic annealing
            end
            
            % Model dimensions
            K = length(obj_array);
            T = size(obj_array(1).y,2);
            gt_m = zeros(K,T);
            
            parfor m = 1:K
                x_t_n = x_t_n_all{m}(:,2:end); % (index 1 corresponds to t=0, etc.)
                P_t_n = P_t_n_all{m}(:,:,2:end);
                for ii = 1:T % t=1 -> t=T
                    gt_m(m,ii) = -0.5/Temp * (obj_array(m).y(ii)'/obj_array(m).R*obj_array(m).y(ii) -...
                        2*obj_array(m).y(ii)'/obj_array(m).R*obj_array(m).G * x_t_n(:,ii) +...
                        trace(obj_array(m).G'/obj_array(m).R*obj_array(m).G * (P_t_n(:,:,ii) + x_t_n(:,ii)*x_t_n(:,ii)')));
                end
            end
            
            qt_m = exp(gt_m); % equation 4.13
        end
        
        function [logL_bound] = compute_logL_bound(obj_array, logL_HMM, logL_SSM, ht_m, gt_m) % Compute the bound on the switching model log likelihood
            % This bound on the marginal log likelihood of observed data is
            % also known as Evidence Lower BOund (ELBO) or negative
            % variational free energy (F)
            
            % Precompute the log determinant of R
            R_logdet = arrayfun(@(x) sparse_find_log_det_mex(2*pi*x.R), obj_array);
            
            % First term: determinant of R
            F1 = -1/2 * R_logdet * sum(ht_m, 2);
            
            % Second term: marginal log likelihood from HMM
            F2 = sum(logL_HMM);
            
            % Third term: -ht*gt
            F3 = -sum(ht_m.*gt_m,'all');
            
            % Fourth term: marginal log likelihood from SSM combined with
            % log(ht) to deal with infinity values
            F4_tmp = logL_SSM + 1/2 * (R_logdet' - size(obj_array(1).y, 1)*log(ht_m));
            F4_tmp(isinf(F4_tmp)) = nan;
            F4 = nansum(F4_tmp,'all'); %#ok<NANSUM>
            
            logL_bound = F1 + F2 + F3 + F4;
        end
        
        function [obj_array] = m1_step(obj_array, x_t_n_all, P_t_n_all, P_t_tmin1_n_all, ht_m, priors_all, shared_R, shared_ctype) % MAP estimation of SSM parameters with weighted data
            % ML or MAP estimation of SSM parameters for the M.1 step in
            % improved Ghahramani & Hinton (2000) VB learning of switching
            % linear segments.
            
            % Number of alternative models
            K = length(obj_array);
            
            parfor m = 1:K % estimate model parameters individually
                if ~isempty(priors_all)
                    % Compute MAP estimation updates
                    [F_opt, Q_opt, mu0_opt, Q0_opt, G_opt, R_opt, R_ss_m, A_m, B_m, C_m] = map_estimate(obj_array(m), x_t_n_all{m}, P_t_n_all{m}, P_t_tmin1_n_all{m}, ht_m(m,:), priors_all{m}); %#ok<*ASGLU>
                else
                    % Compute ML estimation updates
                    [F_opt, Q_opt, mu0_opt, Q0_opt, G_opt, R_opt, R_ss_m, A_m, B_m, C_m] = ml_estimate(obj_array(m), x_t_n_all{m}, P_t_n_all{m}, P_t_tmin1_n_all{m}, ht_m(m,:));
                end
                
                % Update state equation parameters
                obj_array(m).F   = F_opt;
                obj_array(m).Q   = Q_opt;
                obj_array(m).mu0 = mu0_opt;
                obj_array(m).Q0  = Q0_opt;
                %obj_array(m).G   = G_opt;
                if ~isnan(shared_R)
                    obj_array(m).R   = R_opt;
                end
                
                % Store sum variables for joint estimations
                R_ss(:,:,m) = R_ss_m;
                A{m} = A_m;
                B{m} = B_m;
                C{m} = C_m;
            end
            
            if ~isnan(shared_R) && shared_R % Update the observation noise R shared by models
                R_ss = squeeze(sum(R_ss,3));
                if ~isempty(priors_all)
                    R_init = priors_all{1}(1).R; % all models and components should have the same prior on R if doing joint estimation
                    alpha = size(obj_array(1).y,2) * 0.1 / 2; % shape parameter is set to be peaky
                    beta  = R_init * (alpha + 1); % setting the mode of inverse gamma prior to be R_init
                    R_opt = (beta + R_ss/2) / (alpha + size(obj_array(1).y,2)/2 + 1); % using the mode of inverse gamma posterior as R_opt
                else
                    R_opt = R_ss / size(obj_array(1).y,2);
                end
                % distribute the optimized R to all models
                for m = 1:K
                    obj_array(m).R = R_opt;
                end
            end
            
            if ~isempty(shared_ctype) % Update F and Q for components shared across models
                assert(all(cellfun(@(x) length(x)==K, shared_ctype)==1), 'shared_ctype is not specified for all models. Use index = 0 if a model does not share the component.')
                
                for j = 1:length(shared_ctype)
                    % grab the indices for the shared component in all models
                    shared_ctype_index = shared_ctype{j};
                    
                    % create temporary trellis variables
                    update_index_store = {};
                    A_tmp = [];
                    B_tmp = [];
                    C_tmp = [];
                    
                    % T is multiplied by the number of models sharing the component
                    T = (size(x_t_n_all{1},2)-1) * sum(shared_ctype_index > 0);
                    
                    % accumulate the sums for joint estimation
                    for m = 1:K % loop through all alternative models
                        ctype_index = shared_ctype_index(m);
                        if ctype_index > 0
                            % take care of indexing the states
                            state_cardis = cellfun(@(x) size(x.G,2), obj_array(m).ctype);
                            start_id = sum(state_cardis(1:ctype_index-1)) + 1;
                            end_id = sum(state_cardis(1:ctype_index-1)) + state_cardis(ctype_index);
                            update_index_store{m} = [start_id, end_id];
                            
                            % accumulate the sums for the shared component
                            A_tmp(:,:,m) = A{m}(start_id:end_id, start_id:end_id);
                            B_tmp(:,:,m) = B{m}(start_id:end_id, start_id:end_id);
                            C_tmp(:,:,m) = C{m}(start_id:end_id, start_id:end_id);
                            
                            % grab an index for one model with the shared
                            % component for the next map_estimate step
                            m_idx = m;
                        end
                    end
                    
                    % call the ml/map_estimate method specific to the component
                    ctype = obj_array(m_idx).ctype{shared_ctype_index(m_idx)}; % obtain a dummy component object to call the method
                    if ~isempty(priors_all)
                        prior_set = priors_all{m_idx}(shared_ctype_index(m_idx)); % assume all models with the shared component use the same prior for that component
                        [F_ctype, Q_ctype] = map_estimate(ctype, [], [], [], [], prior_set, squeeze(sum(A_tmp,3)), squeeze(sum(B_tmp,3)), squeeze(sum(C_tmp,3)), T);
                    else
                        [F_ctype, Q_ctype] = ml_estimate(ctype, [], [], [], [], squeeze(sum(A_tmp,3)), squeeze(sum(B_tmp,3)), squeeze(sum(C_tmp,3)), T);
                    end
                    
                    % distribute to all models with the shared component
                    for m = 1:K
                        if shared_ctype_index(m) > 0
                            current_ids = update_index_store{m};
                            start_id = current_ids(1);
                            end_id = current_ids(2);
                            obj_array(m).F(start_id:end_id, start_id:end_id) = F_ctype;
                            obj_array(m).Q(start_id:end_id, start_id:end_id) = Q_ctype;
                        end
                    end
                end
            end
        end
        
        function [A, p1] = m2_step(p_ht_v1_T, edge_marginals, obj) %#ok<*INUSD> % ML estimation of HMM parameters
            % ML estimation of HMM parameters for the M.2 step in original
            % Ghahramani & Hinton (2000) VB learning of switching linear
            % segments.
            
            % M step - ML estimates of the parameters {A, p1}
            p1 = p_ht_v1_T(:,1); % updated prior = P(h1|v_1:T)
            edge_sum = sum(edge_marginals,3); % sum over time points
            A = edge_sum ./ sum(edge_sum,1); % updated transition matrix = \sum_t{P(ht-1,ht|v_1:T)} / \sum_t{P(ht-1|v_1:T)}
        end
        
        %% PARAMETER ESTIMATION METHODS - EM
        function [F_new, Q_new, mu0_new, Q0_new, G_new, R_new, R_ss, A, B, C] = ml_estimate(obj, x_t_n, P_t_n, P_t_tmin1_n, ht) % ML M-step for EM estimate of SSM parameters
            % Uses ML estimates to update SSM parameters.
            %
            % Reference:
            % Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series
            % smoothing and forecasting using the EM algorithm. Journal of time series
            % analysis, 3(4), 253-264.
            %
            % Ghahramani, Z., & Hinton, G. E. (1996). Parameter estimation for linear
            % dynamical systems.
            %
            % Ghahramani, Z., & Hinton, G. E. (2000). Variational learning for
            % switching state-space models. Neural computation, 12(4), 831-864.
            
            % Update the state equation parameters for each independent component --- F and Q
            % Definitions of A,B,C follow the notation used in equations (9,10,11) of S&S 1982
            % These terms correspond to maximizing <H>_Q in G&H 2000, and
            % we precompute sums for all components to increase efficiency
            A = sum(P_t_n(:,:,1:end-1),3) + x_t_n(:,1:end-1)*x_t_n(:,1:end-1)';
            B = sum(P_t_tmin1_n(:,:,2:end),3) + x_t_n(:,2:end)*x_t_n(:,1:end-1)';
            C = sum(P_t_n(:,:,2:end),3) + x_t_n(:,2:end)*x_t_n(:,2:end)';
            
            state_cardis = cellfun(@(x) size(x.G,2), obj.ctype);
            for ii = 1:length(state_cardis) % iterate through components
                % grab the sub-matrices for the current component
                start_id = sum(state_cardis(1:ii-1)) + 1;
                end_id = sum(state_cardis(1:ii-1)) + state_cardis(ii);
                A_tmp = A(start_id:end_id, start_id:end_id);
                B_tmp = B(start_id:end_id, start_id:end_id);
                C_tmp = C(start_id:end_id, start_id:end_id);
                
                % call the ml_estimate method specific to the component
                [F_new(start_id:end_id, start_id:end_id), Q_new(start_id:end_id, start_id:end_id)] = ml_estimate(obj.ctype{ii}, [], [], [], [], A_tmp, B_tmp, C_tmp, size(x_t_n,2)-1);
            end
            
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
        
        function [F_new, Q_new, mu0_new, Q0_new, G_new, R_new, R_ss, A, B, C] = map_estimate(obj, x_t_n, P_t_n, P_t_tmin1_n, ht, prior_sets) % MAP M-step for EM estimate of SSM parameters
            % Uses MAP estimates instead of ML estimates.
            %
            % Reference:
            % Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series
            % smoothing and forecasting using the EM algorithm. Journal of time series
            % analysis, 3(4), 253-264.
            %
            % Ghahramani, Z., & Hinton, G. E. (1996). Parameter estimation for linear
            % dynamical systems.
            %
            % Ghahramani, Z., & Hinton, G. E. (2000). Variational learning for
            % switching state-space models. Neural computation, 12(4), 831-864.
            
            % Update the state equation parameters for each independent component --- F and Q (with priors)
            % Definitions of A,B,C follow the notation used in equations (9,10,11) of S&S 1982
            % These terms correspond to maximizing <H>_Q in G&H 2000, and
            % we precompute sums for all components to increase efficiency
            A = sum(P_t_n(:,:,1:end-1),3) + x_t_n(:,1:end-1)*x_t_n(:,1:end-1)';
            B = sum(P_t_tmin1_n(:,:,2:end),3) + x_t_n(:,2:end)*x_t_n(:,1:end-1)';
            C = sum(P_t_n(:,:,2:end),3) + x_t_n(:,2:end)*x_t_n(:,2:end)';
            
            state_cardis = cellfun(@(x) size(x.G,2), obj.ctype);
            for ii = 1:length(state_cardis) % iterate through components
                % grab the sub-matrices for the current component
                start_id = sum(state_cardis(1:ii-1)) + 1;
                end_id = sum(state_cardis(1:ii-1)) + state_cardis(ii);
                A_tmp = A(start_id:end_id, start_id:end_id);
                B_tmp = B(start_id:end_id, start_id:end_id);
                C_tmp = C(start_id:end_id, start_id:end_id);
                
                % call the map_estimate method specific to the component
                [F_new(start_id:end_id, start_id:end_id), Q_new(start_id:end_id, start_id:end_id)] = map_estimate(obj.ctype{ii}, [], [], [], [], prior_sets(ii), A_tmp, B_tmp, C_tmp, size(x_t_n,2)-1);
            end
            
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
        
        function [prior_sets] = initialize_priors(obj) % Initialize prior parameters for all components
            % different component types could have different prior fields.
            % We need to look through all priors and fill in empty entries
            % in order to concatenate them
            fn_store = {};
            prior_sets = struct;
            for ii = 1:length(obj.ctype)
                prior_set = obj.ctype{ii}.initialize_priors;
                fn = fieldnames(prior_set);
                
                % missing fields in the store
                add_in_store = setdiff(fn, fn_store);
                for jj = 1:length(add_in_store)
                    prior_sets(1).(add_in_store{jj}) = [];
                end
                fn_store = [fn_store; add_in_store];
                
                % missing fields in the set
                add_in_set = setdiff(fn_store, fn);
                for jj = 1:length(add_in_set)
                    prior_set(1).(add_in_set{jj}) = [];
                end
                
                % concatenate the structures that now have the same fields
                prior_sets = [prior_sets, prior_set];
            end
            % remove the empty struct at the first index
            prior_sets(1) = [];
        end
        
        %% SIMULATION METHODS
        function [y, burst_y] = simulate(obj, varargin) % simulate time series
            % This is the function used to generate simulated data using
            % the specified model parameters and random Gaussian processes.
            
            % N.B.:
            % Currently this function only supports the case of one set of
            % underlying model parameters and two observation matrices. It
            % is assumed that the observation matrices are ranked in
            % decreasing complication order such that the first is the full
            % model and the second will have less oscillators. Right now
            % this is only used for spindle+slow / slow simulation, but
            % later we can expand this to more flexible simulation cases.
            
            p = inputParser;
            addRequired (p,'obj',                   @(x)true)
            addParameter(p,'ylength',     1001,     @isnumeric) % length of data
            addParameter(p,'transient',   0,        @isnumeric) % length of transient burst
            parse(p,obj,varargin{:});
            input_arguments = struct2cell(p.Results);
            input_flags = fieldnames(p.Results);
            eval(['[', sprintf('%s ', input_flags{:}), '] = deal(input_arguments{:});']);
            
            % temporary assert functions that can be lifted later
            assert(size(obj.F,3)==1)
            assert(size(obj.Q,3)==1)
            assert(size(obj.mu0,3)==1)
            assert(size(obj.Q0,3)==1)
            assert(size(obj.R,3)==1)
            
            p = length(obj.mu0);
            q = 1; % y is one dimensional for now
            
            % simulate the hidden state time series
            sim_x = zeros(p, ylength);
            sim_x(:,1) = obj.mu0; % x_0_0 % this probably should use Q0 as well instead of just the mean
            for ii = 2:ylength % <<<< this section needs a revision because x should have one extra time point than y
                sim_x(:,ii) = obj.F*sim_x(:,ii-1) + mvnrnd(zeros(1,p), obj.Q, 1)';
            end
            
            % generate the observed data time series
            y = zeros(q, ylength);
            for ii = 1:ylength
                y(ii) = obj.G(:,:,end)*sim_x(:,ii) + mvnrnd(zeros(1,q), obj.R, 1)';
            end
            
            % insert a transient burst of oscillation
            burst_start = floor(ylength/2);
            burst_end = burst_start + transient - 1;
            burst_y = zeros(size(y));
            for ii = burst_start:burst_end
                burst_y(ii) = obj.G(:,:,1)*sim_x(:,ii) - obj.G(:,:,end)*sim_x(:,ii);
                % y(ii) = obj.G(:,:,1)*sim_x(:,ii) + mvnrnd(zeros(1,q), obj.R, 1)';
                y(ii) =  y(ii) + burst_y(ii);
            end
            
        end
        
        %% VISUALIZATION METHODS
        % function - add theoretical spectra method
        
    end
end

%% ADDITIONAL FUNCTIONS
% These algorithm functions are not tied to the ssm class object.
function [p_ht_v1_t, p_ht_v1_T, p_ht_tmin1_v1_T, logL] = forward_backward(A, py_x, p1) % forward-backward algorithm
% Classical forward-backward algorithm for discrete state HMM with
% step-wise normalization. Note that this ensures the final conditional
% densities are correct, but the alpha and beta vectors are off by a
% normalization constant.
%
% Reference:
% Rabiner, L., & Juang, B. (1986). An introduction to hidden
% Markov models. ieee assp magazine, 3(1), 4-16.

% A - transition probability matrix
% py_x - observation probability
% p1 - initial prior of hidden state at t=1

% Model dimensions
K = size(A,1);
T = length(py_x);
logL = zeros(1,T);

if nargin < 3
    p1 = ones(1,K)/K; % default prior is uniform
end

% compute the alpha (forward pass)
norm_a = zeros(K,T); % (index 1 corresponds to t=1, etc.)
norm_a(:,1) = py_x(:,1) .* p1(:) ./ sum(py_x(:,1) .* p1(:)); % t=1
logL(1) = log(sum(py_x(:,1) .* p1(:)));
for ii = 2:T % t=2 -> t=T
    a = py_x(:,ii) .* (A * norm_a(:,ii-1));
    norm_a(:,ii) = a ./ sum(a);
    logL(ii) = log(sum(a)); % one-step predictive log likelihood
end
p_ht_v1_t = norm_a; % filtered hidden state posterior density

if nargout > 1
    % compute the beta (backward pass)
    norm_b = zeros(K,T);
    norm_b(:,end) = ones(1,K); % b(h_T) = 1 for all h_T at t=T
    for ii = T-1:-1:1 % t=T-1 -> t=1
        b = A' * (py_x(:,ii+1) .* norm_b(:,ii+1));
        norm_b(:,ii) = b ./ sum(b);
    end
    % compute the smoothed hidden state posterior density
    p_ht_v1_T = norm_a .* norm_b ./ sum(norm_a .* norm_b); % norm_a and norm_b are defined at t=1 -> t=T
end

if nargout > 2
    % compute pairwise edge marginals using a and b (note that they are normalized)
    p_ht_tmin1_v1_T = zeros(K,K,T);
    for ii = 2:T % t=2 -> t=T
        p_ht_tmin1_v1_T(:,:,ii) = norm_a(:,ii-1)' .* py_x(:,ii) .* A .* norm_b(:,ii) ./ sum(norm_a(:,ii-1)' .* py_x(:,ii) .* A .* norm_b(:,ii), 'all'); % P(ht-1,ht|v_1:T)
    end
end

end

function [A, p1] = baum_welch(A_init, py_x, p1_init, maxEM_iter) % Baum-Welch algorithm for EM estimate of HMM parameters
% In this function, the observation probability, i.e., the emission
% distribution, is fixed and therefore not estimated. A general Baum-Welch
% will be able to estimate py_x at the same time, regardless of discrete or
% Gaussian observation variable y.
%
% Reference:
% Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). A maximization
% technique occurring in the statistical analysis of probabilistic
% functions of Markov chains. The annals of mathematical statistics, 41(1),
% 164-171.

if nargin < 4
    maxEM_iter = 100;
end

% Initialize the EM loop
A = A_init; % A_ij = P(h_t = i | h_t-1 = j), i.e. rows >> t+1, columns >> t
p1 = p1_init;
A_delta = inf;
EM_iter = 0;

while A_delta > 10^-6 && EM_iter <= maxEM_iter
    % store A to check for convergence
    EM_iter = EM_iter + 1;
    last_A = A;
    
    % E step - taking expectation under the hidden state posterior using the
    % current parameters {A, p1}
    [~, ht_m, edge_marginals] = forward_backward(A, py_x, p1);
    
    % M step - ML estimates of the parameters {A, p1}
    p1 = ht_m(:,1); % updated prior = P(h1|v_1:T)
    edge_sum = sum(edge_marginals,3); % sum over time points
    A = edge_sum ./ sum(edge_sum,1); % updated transition matrix = \sum_t{P(ht-1,ht|v_1:T)} / \sum_t{P(ht-1|v_1:T)}
    
    % Check the matrix A for convergence
    A_delta = mean(abs(A - last_A), 'all');
end

end

function [viterbi_path, ht_m] = viterbi(A, py_x, p1) % Viterbi algorithm
% Classical Viterbi algorithm for discrete state HMM to solve argmax exact
% inference. Note that we use the log-probability version to cope with
% underflowing numerical precision, therefore this is an instance of the
% general Max-Sum algorithm specialized to the tree graFcal model of HMM.
%
% Reference:
% Viterbi, A. (1967). Error bounds for convolutional codes and an
% asymptotically optimum decoding algorithm. IEEE transactions on
% Information Theory, 13(2), 260-269.

% A - transition probability matrix
% py_x - observation probability
% p1 - initial prior of hidden state at t=1

% Model dimensions
K = size(A,1);
T = length(py_x);

if nargin < 3
    p1 = ones(1,K)/K; % default prior is uniform
end

% initialize the trellises
Tre_p = zeros(K,T); % store max probability
Tre_p(:,1) = log(p1') + log(py_x(:,1)); % initial state probability
Tre_h = zeros(K,T); % store argmax state

% forward pass to fill in the trellises
for ii = 2:T % t=2 -> t=T
    [Tre_p(:,ii), Tre_h(:,ii)] = max(Tre_p(:,ii-1)' + log(A) + log(py_x(:,ii)), [], 2);
end

% backward pass to identify the global argmax path
viterbi_path = zeros(1,T);
[~, viterbi_path(end)] = max(Tre_p(:,end));
for ii = T-1:-1:1 % t=T-1 -> t=1
    viterbi_path(ii) = Tre_h(viterbi_path(ii+1), ii+1);
end

if nargout > 1
    % output hard segmentation of the hidden states
    ht_m = zeros(K,T);
    for ii = 1:K
        ht_m(ii, viterbi_path==ii) = 1;
    end
end

end

%% HELPER FUNCTIONS
function [] = mustBeSquare(a) % check square matrix
% check if a matrix is square
assert(size(a,1)==size(a,2), 'matrix is not square')
end

function [] = mustBeComponent(a) % check if valid ctype
% check if the input is a valid component cell array or a component string
if isempty(a) % since no implicit default value, we accept any empty class
elseif iscell(a) % if cell, it should be an array of component objects
    cellfun(@(x) mustBeNumericOrLogical(x.default_G), a); % .default_G()
else
    mustBeText(a) % subclasses should declare component name here
end
end
