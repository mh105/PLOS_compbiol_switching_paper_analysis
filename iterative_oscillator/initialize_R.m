function R_out = initialize_R(data, Fs, freq_cutoff, one_point)
    
    fft_data=fftshift(fft(data));
    psd_data=abs(fft_data).^2;
    freq=linspace(-Fs/2,Fs/2,length(psd_data));
    [min_val min_i]=min(abs(freq-freq_cutoff));
    if one_point
        R_out=psd_data(min_i)/(2*pi*Fs);
    else
        R_out=mean(psd_data(min_i+1:end))/(2*pi*Fs); % should be divided by length(psd_data)
    end
    
%     figure
%     plot(freq, 20*log10(psd_data)); hold on
%     plot(freq(min_i+1:end),20*log10(psd_data(min_i+1:end)));
%     title('Check PSD and selected noise freq')
%     legend('Full PSD','Noise Freq')
    
end
