%CLIMSCALE Rescale the color limits of an image to remove outliers with percentiles
%
%   clim=climscale(h, ptiles);
%
%   h: image handle (optional, otherwise h=gca)
%   ptiles: percentiles (optional, default [5 98])
function climscale(h, ptiles)
if nargin==0
    h=gca;
end

if nargin<2
    ptiles=[5 98];
end

children=get(h, 'children');

for i=1:length(children)
    if isprop(children(i),'cdata') && size(children(i).CData,1)>0
        data=get(children(i), 'cdata');
        clim=prctile(data(~isinf(data) & ~isnan(data)), ptiles);%prctile(reshape(data,1,numel(data)), ptiles);
        clim(2)=clim(2)+1e-10;
        set(h,'clim',clim);
    end
end