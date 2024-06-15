%% Script to detect and audti strokes of penguins from accelerometer data

% Ashley M. Blawas
% June 14, 2023
% This script largely relies on functions written by Mark Johnson orignally
% developed for acoustic auditing of DTAG data

%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 10, "Encoding", "UTF-8");

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Seconds", "ECG", "Accel", "Depth", "Var5", "Var6", "Var7", "Var8", "Var9", "Var10"];
opts.SelectedVariableNames = ["Seconds", "ECG", "Accel", "Depth"];
opts.VariableTypes = ["double", "double", "double", "double", "string", "string", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var5", "Var6", "Var7", "Var8", "Var9", "Var10"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var5", "Var6", "Var7", "Var8", "Var9", "Var10"], "EmptyFieldRule", "auto");

% Import the data
tbl = readtable("C:\Users\ashle\Downloads\07_Penguin-Phys_PP_01_AllData_EmperorPenguin35_3dives.csv", opts);

% Clear temporary variables
clear opts

%% Set tag vars

tag =      'EmperorPenguin35'; %Set tag name i.e. pm19_136a
recdir =   'C:\Users\ashle\Dropbox\Ashley\Post-Doc\Projects\Antarctic penguins\data\EmperorPenguin35'; 
prefix =   tag; %Set prefix, probably the same as the tag name

settagpath('audit', strcat(recdir, '\audit')); %Set path for audit files

%% Plot raw data

figure
f(1) = subplot(311);
plot(tbl.Seconds, tbl.Depth, 'k'); hold on
set(gca, 'YDir','reverse')
xlabel('Time (s)'); ylabel('Depth (m)');
%legend('Depth', 'Corrected Depth', 'Start of dive', 'End of dive');

f(2) = subplot(312);
plot(tbl.Seconds, tbl.ECG, 'r'); hold on
xlabel('Time (s)'); ylabel('ECG (V)');
%legend('X', 'Y', 'Z');

f(3) = subplot(313);
plot(tbl.Seconds, tbl.Accel, 'b'); hold on
xlabel('Time (s)'); ylabel('Acc-X (m s^-2)')


linkaxes(f, 'x');

%% Find the dives

fs = 1/(tbl.Seconds(2) - tbl.Seconds(1));

% Find dives of at least 10m, start/end at 2m, and don't include
% incomplete dives
T = finddives(tbl.Depth,fs,[10,2,0]);

%% Find beats
% 
% for int_num = 1:size(T, 1)*2+1
%     
%     % We know first interval is at surface
%     if int_num == 1
%         int_start_s = 0; % int start in seconds
%         int_end_s = T(int_num, 1)-1/fs; % int end in seconds
%         
%         int_start_i = 1;
%         int_end_i = find(min(abs(int_end_s - tbl.Seconds)) == abs(int_end_s - tbl.Seconds));
% 
%     elseif int_num == size(T, 1)*2+1
%         int_start_s = T((int_num-1)/2, 2); % int start in seconds
%         int_end_s = max(tbl.Seconds); % int end in seconds
%         
%         int_start_i = find(min(abs(int_start_s - tbl.Seconds)) == abs(int_start_s - tbl.Seconds));
%         int_end_i = length(tbl.Seconds);
%         
%     elseif mod(int_num, 2) == 0
%         int_start_s = T(int_num/2, 1); % int start in seconds
%         int_end_s = T(int_num/2, 2); % int end in seconds
%         
%         int_start_i = find(min(abs(int_start_s - tbl.Seconds)) == abs(int_start_s - tbl.Seconds));
%         int_end_i = find(min(abs(int_end_s - tbl.Seconds)) == abs(int_end_s - tbl.Seconds));
%         
%     elseif mod(int_num, 2) ~= 0
%         int_start_s = T((int_num-1)/2, 2); % int start in seconds
%         int_end_s = T((int_num+1)/2, 1); % int end in seconds
%         
%         int_start_i = find(min(abs(int_start_s - tbl.Seconds)) == abs(int_start_s - tbl.Seconds));
%         int_end_i = find(min(abs(int_end_s - tbl.Seconds)) == abs(int_end_s - tbl.Seconds));
%     end
%     
%     split_ECG.Seconds{int_num} = tbl.Seconds(int_start_i:int_end_i);
%     split_ECG.Depth{int_num} = tbl.Depth(int_start_i:int_end_i);
%     
%     % Process ECG
%     temp = filloutliers(tbl.ECG(int_start_i:int_end_i),"nearest","mean");
%     
%     [p,s,mu] = polyfit((1:numel(temp))', temp, 6);
%     f_y = polyval(p,(1:numel(temp))',[],mu);
%     
%     temp = temp - f_y;        % Detrend data
% 
%     split_ECG.ECG{int_num} = normalize(temp);
%     
% % Use wavelet feature detector
% % wt = modwt(split_ECG.ECG{int_num},5);
% % wtrec = zeros(size(wt));
% % wtrec(4:5,:) = wt(4:5,:);
% % y = imodwt(wtrec,'sym6');
% tm = split_ECG.Seconds{int_num};
% y = split_ECG.ECG{int_num};
% 
% % y = hilbert(y);
% % 
% % y = abs(y).^2;
% 
% [qrspeaks,locs] = findpeaks(y,tm,'MinPeakHeight', 1,...
%         'MinPeakDistance',0.25,'MinPeakProminence',0.5);
% 
% figure
% plot(tm,y)
% hold on
% plot(locs,qrspeaks,'ro')
% 
% xlabel('Seconds')
% title('R Peaks Localized by Wavelet Transform with Automatic Annotations')
% 
% % Calculate stroke rate in strokes/minute
% split_ECG.HR{int_num} = 60./diff(locs);
% split_ECG.HR_Seconds{int_num} = locs;
% 
% end

%% Find strokes

% We only care about strokes during dives
% Go through acc and split up into cells based on dives 

for dive_num = 1:size(T, 1)
    
    dive_start_s = T(dive_num, 1); % dive start in seconds
    dive_end_s = T(dive_num, 2); % dive end in seconds
    
    dive_start_i = find(min(abs(dive_start_s - tbl.Seconds)) == abs(dive_start_s - tbl.Seconds));
    dive_end_i = find(min(abs(dive_end_s - tbl.Seconds)) == abs(dive_end_s - tbl.Seconds));

    split_Accel.Seconds{dive_num} = tbl.Seconds(dive_start_i:dive_end_i);
    split_Accel.Depth{dive_num} = tbl.Depth(dive_start_i:dive_end_i);
    split_Accel.Accel{dive_num} = filloutliers(tbl.Accel(dive_start_i:dive_end_i),"nearest","mean");
    
% Use wavelet feature detector
wt = modwt(split_Accel.Accel{dive_num},5);
wtrec = zeros(size(wt));
wtrec(3:4,:) = wt(3:4,:);
y = imodwt(wtrec,'sym4');
tm = split_Accel.Seconds{dive_num};

y = hilbert(y);

y = abs(y).^2;

[qrspeaks,locs] = findpeaks(y,tm,'MinPeakHeight',0.002,...
    'MinPeakDistance',0.5, 'MinPeakProminence', 0.002);
figure
plot(tm,y)
hold on
plot(locs,qrspeaks,'ro')

xlabel('Seconds')
title('R Peaks Localized by Wavelet Transform with Automatic Annotations')

split_Accel.SR_Seconds{dive_num} = locs;

end

s_times = vertcat(split_Accel.SR_Seconds{:}); 

% Calculate stroke rate in strokes/minute
sr = 60./diff(s_times);
sr_s = s_times;

% If there is a gap in measurements greater than 10 seconds, show NA
for i = 2:length(sr)
    if sr_s(i) - sr_s(i-1) > 10
        sr_s(i-1) = NaN;
        sr(i-1) = NaN;
    end
end


temp = table(sr_s, zeros(length(sr_s), 1), repmat('s', 1, length(sr_s))');


% Write detections to text file
writetable(temp, strcat(recdir, '\audit\', tag, '_.txt'),'Delimiter','\t');

%% Audit strokes

R = loadaudit(tag); % Load an audit if one exists
tcue = 120; % Time cue in data to start analysis, should be 0 when you first begin
R = d3audit(recdir, prefix, tcue, R, tbl); %Run audit (for d3s)
saveaudit(prefix, R); % Save audit

locs = sort(R.cue(:, 1));
       
%% Plot stroke rate

figure
f(1) = subplot(411);
plot(tbl.Seconds, tbl.Depth, 'k'); hold on
set(gca, 'YDir','reverse')
xlabel('Time (s)'); ylabel('Depth (m)');
%legend('Depth', 'Corrected Depth', 'Start of dive', 'End of dive');

f(2) = subplot(412);
% for int_num = 1:1:size(T, 1)*2+1
%     line([split_ECG.HR_Seconds{int_num}(2:end), split_ECG.HR_Seconds{int_num}(2:end)], [min(tbl.ECG), max(tbl.ECG)], 'Color', 'k');
% hold on
% end
plot(tbl.Seconds, tbl.ECG, 'r'); hold on
xlabel('Time (s)'); ylabel('ECG (V)');
%legend('X', 'Y', 'Z');

f(3) = subplot(413);
%line([sr_s, sr_s], [min(tbl.Accel), max(tbl.Accel)], 'Color', 'k');
hold on
plot(tbl.Seconds, tbl.Accel, 'b'); hold on
xlabel('Time (s)'); ylabel('Acc-X (m s^-2)')

f(4) = subplot(414);
plot(sr_s(2:end), sr, 'k.-');
xlabel('Time (s)'); ylabel('Stroke Rate (strokes/min)')

linkaxes(f, 'x');