%% FUNCTION:plotRES
function plotRES(AXc,RES,XLIMS,AXs,AXr)

% Plot comments
axes(AXc)
if ~isempty(RES.cue),
    kk = find(sum(RES.cue')' > XLIMS(1) & RES.cue(:,1)<=XLIMS(2)) ;
    if ~isempty(kk),
        plot([RES.cue(kk,1) sum(RES.cue(kk,:)')']',0.2*ones(2,length(kk)),'k*-') ;
        for k=kk',
            text(max([XLIMS(1) RES.cue(k,1)+0.01]),0.6,RES.stype{k},'FontSize',10) ;
        end
    else
        plot(0,0,'k*-') ;
    end
else
    plot(0,0,'k*-') ;
end

set(AXc,'XLim',XLIMS,'YLim',[0 1]) ;
bc = get(gcf,'Color') ;
set(AXc,'Box','off','XTick',[],'YTick',[],'XColor',bc,'YColor',bc,'Color',bc) ;

% Plot strokes
axes(AXs)
yl = get(gca,'YLim') ; % Get current y axis limits before plotting
if ~isempty(RES.cue),
    kk = find(sum(RES.cue')' > XLIMS(1) & RES.cue(:,1)<=XLIMS(2)) ; % Find events that start and end within bounds of plot
    if ~isempty(kk),
        arrayfun(@(a)xline(a, 'k:'),RES.cue(kk,1)); % Plot lines
        ylim([yl]) ; % Reset axis limits
    else
        %plot(0,0,'k*-') ;
    end
else
    %plot(0,0,'k*-') ;
end

% Calculate stroke rate in strokes/minute
sr = 60./diff(RES.cue(:,1));
sr_s = RES.cue(:,1);

% If there is a gap in measurements greater than 10 seconds, show NA
for i = 2:length(sr)
    if sr_s(i) - sr_s(i-1) > 10
        sr_s(i-1) = NaN;
        sr(i-1) = NaN;
    end
end

% Plot stroke rate
axes(AXr)
xl = get(gca,'XLim') ; % Get current y axis limits before plotting
if ~isempty(RES.cue),
    %kk = find(sum(RES.cue')' > XLIMS(1) & RES.cue(:,1)<=XLIMS(2)) ; % Find events that start and end within bounds of plot
    %if ~isempty(kk),
        plot(sr_s(2:end), sr, 'k.-');
        xlim([xl]) ; % Reset axis limits
    %else
        %plot(0,0,'k*-') ;
    %end
else
    %plot(0,0,'k*-') ;
end
ylabel('Stroke Rate, strokes/min'), xlabel('Time, s')



return

end

