function graph_AIC(AIC_all,deltas,AIC_w)
    subplot(3,1,1)
    plot([1:length(AIC_w)],AIC_all,'-*'); title('AIC values');hold on
    subplot(3,1,2);
    plot([1:length(AIC_w)],deltas,'-*'); title('AIC deltas'); hold on
    subplot(3,1,3);
    plot([1:length(AIC_w)],AIC_w,'-*'); title('AIC weights');hold on
    xlabel('Number of Oscillations')
end
