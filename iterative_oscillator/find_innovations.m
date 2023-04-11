function [y_pred,y_smooth,y_filt,y]=find_innovations(modelOsc,win)
    y = modelOsc.res{1,win}.y;
    Phi = modelOsc.res{1,win}.model_prams.Phi;
    Q = modelOsc.res{1,win}.model_prams.Q;
    R = modelOsc.res{1,win}.model_prams.R;
    mu = modelOsc.res{1,win}.model_prams.mu;
    Sigma = modelOsc.res{1,win}.model_prams.sigma;
    
    [x_t_n,P_t_n,P_t_tmin1_n,logL,x_t_t,P_t_t,K_t,x_t_tmin1,P_t_tmin1] = kalman_filter(y,Phi,Q,R,mu,Sigma);
    
    if size(x_t_n,1)>2
        y_pred=sum(x_t_tmin1(1:2:end,2:end));
        y_smooth=sum(x_t_n(1:2:end,2:end));
        y_filt=sum(x_t_t(1:2:end,2:end));
    else
        y_pred=x_t_tmin1(1,2:end);
        y_smooth=x_t_n(1,2:end);
        y_filt=x_t_t(1,2:end);
    end
end