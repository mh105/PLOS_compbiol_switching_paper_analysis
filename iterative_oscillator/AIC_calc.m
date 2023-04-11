function [AIC_out, param_num]=AIC_calc(modelOsc,win)
%Copyright 2019, All rights reserved. Amanda Beck and Patrick Purdon,
%Purdon Lab
%modelOsc is the outcome structure of osc_harm_decomp. Win is the window
%number in that structure (structure will contain many windows when
%parallelized)
    ll=modelOsc.res{1,win}.ll; %log likelihood from Shumway and Stoffer 1982, Gupta and Mehra 1974
    [osc_num, red_num]=size(modelOsc.res{1,win}.model_prams.Phi);
    %osc_num is 2*number of oscillations if AR(2). Each AR(2) has 3
    %parameters
    if ~mod(osc_num,2) %if dimension is even
        param_num=1.5*osc_num+1-4; %parameters are radius, frequency, and state variance %-4 for the 3 set frequencies and 1 harmonic
        %add 1 for observation noise
    else %if dimension is odd
        param_num=1.5*(osc_num-1)+3-4; %assuming there is only one AR(1). 2 parameters for AR(1) and 1 for observationnoise 
    end
    AIC_out = 2*(param_num)-2*ll; %Akaike Information Criterion 1981
end