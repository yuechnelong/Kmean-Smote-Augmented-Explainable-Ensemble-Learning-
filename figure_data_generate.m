%% Plot data and classes
function []=figure_data_generate(origin_data,SyntheticData,origin_data_label,Synthetic_label)
rng('default')
figure('Position',[200,200,400,320])
% subplot(1,2,1)
histogram(origin_data, 'Normalization', 'probability', 'DisplayName', 'Original Data');
hold on;
histogram(SyntheticData, 'Normalization', 'probability', 'DisplayName', 'Synthetic Data');
legend('Original','Synthetic')
legend box off;
xlabel('Value');
ylabel('Probability');
set(gca, 'LineWidth',1,...                                 % Line width
'xGrid', 'on', 'YGrid', 'on', ...                 % Grid
'TickDir', 'in', 'TickLength', [.01 .01], ...     % Tick
'xMinorTick', 'on', 'YMinorTick', 'on', ...     % Minor tick
'xColor', [.1 .1 .1],  'YColor', [.2 .2 .2])      % Atest_yis color
% subplot(1,2,2)
% histogram(SyntheticData, 'Normalization', 'probability', 'DisplayName', 'Synthetic Data');
% hold on;
% real_data_mean=mean(origin_data(:));  
% real_data_std=std(origin_data(:));
% x_range = linspace(real_data_mean - 3 * real_data_std, real_data_mean + 3 * real_data_std, 100);
% real_data_distribution = normpdf(x_range, real_data_mean, real_data_std);
% plot(x_range, real_data_distribution, 'Color',[0.3,0.5,0.1], 'LineWidth', 1.3, 'DisplayName', 'Real Data Distribution');
% legend box off;
% xlabel('Value');
% ylabel('Probability');

set(gca, 'LineWidth',1,...                                 % Line width
'xGrid', 'on', 'YGrid', 'on', ...                 % Grid
'TickDir', 'in', 'TickLength', [.01 .01], ...     % Tick
'xMinorTick', 'on', 'YMinorTick', 'on', ...     % Minor tick
'xColor', [.1 .1 .1],  'YColor', [.2 .2 .2])      % Atest_yis color
%%
figure('Position',[200,200,600,300])
subplot(1,2,1)
boxchart(origin_data);title('Original Data');
ylabel('Value');
dimension=size(origin_data,2);
% xticks([1:dimension])
for i=1:size(origin_data,2)
    xtict_str{1,i}=['fearure',num2str(i)];
end
xticklabels(xtict_str)
set(gca, 'LineWidth',1,...                                 % Line width
'xGrid', 'on', 'YGrid', 'on', ...                 % Grid
'TickDir', 'in', 'TickLength', [.01 .01], ...     % Tick
'xMinorTick', 'on', 'YMinorTick', 'on', ...     % Minor tick
'xColor', [.1 .1 .1],  'YColor', [.2 .2 .2])      % Atest_yis color
subplot(1,2,2)
boxchart(SyntheticData);title('Synthetic Data');
ylabel('Value');
set(gca, 'LineWidth',1,...                                 % Line width
'xGrid', 'on', 'YGrid', 'on', ...                 % Grid
'TickDir', 'in', 'TickLength', [.01 .01], ...     % Tick
'xMinorTick', 'on', 'YMinorTick', 'on', ...     % Minor tick
'xColor', [.1 .1 .1],  'YColor', [.2 .2 .2])      % Atest_yis color
% xticks([1:size(origin_data,2)])
for i=1:size(origin_data,2)
    xtict_str{1,i}=['fearure',num2str(i)];
end
xticklabels(xtict_str)
%%
figure('Position',[200,200,600,300]) %图框大小 600,300分别是长和高
subplot(1,2,1)
origin_data_lowdim=tsne(origin_data); %高纬度数据降为低维,默认是2维度

numGroups = length(unique(origin_data_label)); %识别出数据类别是多少类
clr = hsv(numGroups);
gscatter(origin_data_lowdim(:,1),origin_data_lowdim(:,2),origin_data_label,clr,'*o^psd',3)
xlabel('tsne dimension1')
ylabel('tsne dimension2')

for i=1:numGroups
    legend_str{1,i}=['class',num2str(i)];
end
set(gca, 'LineWidth',1,...                                 % Line width
'xGrid', 'on', 'YGrid', 'on', ...                 % Grid
'TickDir', 'in', 'TickLength', [.01 .01], ...     % Tick
'xMinorTick', 'on', 'YMinorTick', 'on', ...     % Minor tick
'xColor', [.1 .1 .1],  'YColor', [.2 .2 .2])      % Atest_yis color
legend(legend_str)
legend box off
title('Original Data');

subplot(1,2,2)
boxchart(SyntheticData);
SyntheticData_lowdim=tsne(SyntheticData); %高纬度数据降为低维,默认是2维度

numGroups = length(unique(origin_data_label)); %识别出数据类别是多少类
clr = hsv(numGroups);
gscatter(SyntheticData_lowdim(:,1),SyntheticData_lowdim(:,2),Synthetic_label,clr,'*o^psd',3)
xlabel('tsne dimension1')
ylabel('tsne dimension2')
set(gca, 'LineWidth',1,...                                 % Line width
'xGrid', 'on', 'YGrid', 'on', ...                 % Grid
'TickDir', 'in', 'TickLength', [.01 .01], ...     % Tick
'xMinorTick', 'on', 'YMinorTick', 'on', ...     % Minor tick
'xColor', [.1 .1 .1],  'YColor', [.2 .2 .2])      % Atest_yis color
xlabel('tsne dimension1')
ylabel('tsne dimension2')
for i=1:numGroups
    legend_str{1,i}=['class',num2str(i)];
end
legend(legend_str)
legend box off
title('Synthetic Data');