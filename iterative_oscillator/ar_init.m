function [add_freq,add_rad, freqs, pole_r] = ar_init(y,Fs,fig_name,ar_order)
%Fitting a time series with a specified AR order, output frequency of pole
%with largest radius

% FFT
win_fft=(fft(y));
fftL=length(win_fft);
plot_fft=win_fft(1:floor(fftL/2));
fft_f=[1:floor(fftL/2)]*Fs/length(y);

%Multitaper Spectrum
specparams.Fs=Fs;
specparams.tapers =[3 5];
[s_yp, f_yp]=mtspectrumc(y,specparams);
Ks=abs(plot_fft(1))/abs(s_yp(1));

%Fitting with AR
a=aryule(y,ar_order);
H2=freqz(1,[a],fft_f,Fs);
K2=(abs(plot_fft))./(abs(H2'));
K2_mean=mean(K2);

%Identify the complex poles
r=roots(a);
freqs=atan2(imag(r),real(r))/(2*pi)*Fs;
pole_r=abs(r);
if freqs(1) == 0
    [maxR, r_ind]=max(r(2:end));
    r_ind = r_ind+1;
else
    [maxR, r_ind]=max(r);
end

%output frequency and radius for "most influential" pole
add_freq=abs(freqs(r_ind));
add_rad=pole_r(r_ind);

%% Plot Fit for later
f=figure;
plot(fft_f,20*log10(abs(plot_fft)),f_yp,20*log10(Ks*abs(s_yp)),'c',fft_f,20*log10(K2_mean*abs(H2)),'r','LineWidth',2);
legend('FFT of y','Multitaper Spectrum of y','AR fit');
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
title('Fitting the Innovations');
savefig(gcf,fig_name)
close(f)

end

