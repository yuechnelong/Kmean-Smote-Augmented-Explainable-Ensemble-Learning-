%% =========================================================================
% plot_model_comparison.m
% 模型结果对比图 + 泰勒图
% =========================================================================
clc; clear; close all;

%% ========================================================================
%% 数据准备
%% ========================================================================
model_names = {
    'No Augmentation'
    'SMOTE'
    'GAN'
    'Diffusion-RF'
    'Diffusion-CBiLSTM'
    'Diffusion-RF-CBiLSTM'
    };

metrics = {'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity'};

% 数据矩阵 [模型数 x 指标数]
data = [
    0.80570,  0.74200,  0.78100,  0.76100,  0.95800;
    0.89740,  0.82400,  0.87200,  0.84732,  0.98100;
    0.90410,  0.83500,  0.88400,  0.85880,  0.98300;
    0.91000,  0.84500,  0.89500,  0.86935,  0.98500;
    0.89500,  0.83000,  0.88000,  0.85422,  0.98200;
    0.93478,  0.86855,  0.91895,  0.88052,  0.98836;
    ];

n_models = size(data, 1);
n_metrics = size(data, 2);

%% 配色 (Nature / Science 风格大地色系)
colors = [
    0.620  0.620  0.620;   % 灰 — 无增强
    0.839  0.376  0.302;   % 红褐 — SMOTE
    0.204  0.549  0.659;   % 青蓝 — GAN
    0.855  0.329  0.102;   % 橙色 — Diff-RF
    0.369  0.510  0.675;   % 钢蓝 — Diff-CBiLSTM
    0.133  0.545  0.133;   % 森林绿 — Diff-RF-CBiLSTM
    ];

%% ========================================================================
%% 图1: 分组柱状图 —— 各模型五指标对比
%% ========================================================================
figure('Position', [50, 150, 1280, 580], 'Color', 'w');

bar_width = 0.12;
group_gap = 0.18;
x_centers = 1:n_models;

hold on;
for m = 1:n_metrics
    x_pos = x_centers + (m - 1 - (n_metrics - 1) / 2) * bar_width;
    b = bar(x_pos, data(:, m), bar_width, ...
        'FaceColor', 'flat', 'EdgeColor', 'none', 'FaceAlpha', 0.92);
    for i = 1:n_models
        b.CData(i, :) = colors(i, :);
    end

    % 数据标签
    for i = 1:n_models
        text(x_pos(i), data(i, m) + 0.006, sprintf('%.3f', data(i, m)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7.2, ...
            'Rotation', 90, 'Color', [0.25 0.25 0.25]);
    end
end

% 添加最优模型竖线
x_best = x_centers(end);
xline(x_best, '--', 'LineWidth', 2.2, 'Color', [0.85 0.33 0.1], 'Alpha', 0.7);

set(gca, 'XTick', x_centers, 'XTickLabel', model_names, ...
    'FontSize', 11.5, 'LineWidth', 1.1, 'TickDir', 'out', ...
    'TickLength', [0.005 0.005]);
ylabel('Score', 'FontSize', 13, 'FontWeight', 'bold');
ylim([0.70, 1.01]);
yticks(0.70:0.05:1.00);
yticklabels(arrayfun(@(x) sprintf('%.2f', x), 0.70:0.05:1.00, 'UniformOutput', false));

% 图例
h_leg = zeros(1, n_metrics);
leg_str = metrics;
leg_colors = lines(n_metrics);
for m = 1:n_metrics
    h_leg(m) = bar(nan, nan, bar_width, 'FaceColor', leg_colors(m, :), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.92);
end
% 实际上我们想要模型图例，用 Patch 做
delete(h_leg);
hold off;

% 重做模型图例
ax = gca;
hold(ax, 'on');
h_model = zeros(1, n_models);
for i = 1:n_models
    h_model(i) = bar(nan, nan, 1, 'FaceColor', colors(i, :), ...
        'EdgeColor', 'none');
end
leg = legend(h_model, model_names, 'Location', 'northwest', ...
    'FontSize', 10, 'Box', 'off', 'NumColumns', 2);
leg.Title.String = 'Model';
leg.Title.FontSize = 11;
leg.Title.FontWeight = 'bold';

% 指标分组标签
ax_pos = get(gca, 'Position');
metric_x = x_centers;
for m = 1:n_metrics
    xp = x_centers(1) + (m - 1 - (n_metrics - 1) / 2) * bar_width;
    annotation('textbox', ...
        [ax_pos(1) + (xp - 0.5) / (n_models + 0.8) * ax_pos(3) - 0.015, ...
         ax_pos(2) - 0.06, 0.04, 0.03], ...
        'String', metrics{m}, 'FontSize', 7.5, 'Color', [0.35 0.35 0.35], ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
        'FontWeight', 'bold');
end

title('Model Performance Comparison across Five Metrics', ...
    'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.25]);
grid on; set(gca, 'GridAlpha', 0.15);

%% ========================================================================
%% 图2: 热力图 —— 模型 x 指标
%% ========================================================================
figure('Position', [50, 150, 750, 520], 'Color', 'w');

% 热力图
imagesc(data);
colormap(flipud(summer));
clim([0.74, 1.0]);
cb = colorbar;
cb.Label.String = 'Score';
cb.Label.FontSize = 12;
cb.Label.FontWeight = 'bold';

% 文字标注
for i = 1:n_models
    for j = 1:n_metrics
        if data(i, j) >= 0.90
            text(j, i, sprintf('%.4f', data(i, j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 14, ...
                'FontWeight', 'bold', 'Color', [0 0 0]);
        else
            text(j, i, sprintf('%.4f', data(i, j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 14, ...
                'FontWeight', 'bold', 'Color', [0.15 0.15 0.15]);
        end
    end
end

set(gca, 'XTick', 1:n_metrics, 'XTickLabel', metrics, ...
    'YTick', 1:n_models, 'YTickLabel', model_names, ...
    'FontSize', 12, 'LineWidth', 1, 'TickDir', 'out', ...
    'XAxisLocation', 'top');
title('Performance Heatmap: Models × Metrics', ...
    'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.25]);

% 高亮最佳单元格
[best_val, best_idx] = max(data(:));
[best_row, best_col] = ind2sub(size(data), best_idx);
hold on;
rectangle('Position', [best_col - 0.5, best_row - 0.5, 1, 1], ...
    'LineWidth', 3.5, 'EdgeColor', [0.85 0.33 0.1]);

%% ========================================================================
%% 图3: 雷达图/蜘蛛图
%% ========================================================================
figure('Position', [50, 150, 780, 680], 'Color', 'w');

theta = (0:n_metrics - 1) * 2 * pi / n_metrics;
theta_full = [theta, theta(1)];

% 只画最后4个模型（避免太拥挤）
plot_models = [3, 4, 5, 6];
n_plot = length(plot_models);
plot_colors = colors(plot_models, :);
plot_names = model_names(plot_models);

for i = 1:n_plot
    vals = [data(plot_models(i), :), data(plot_models(i), 1)];
    h = polarplot(theta_full, vals, '-', 'LineWidth', 2.4, ...
        'Color', plot_colors(i, :));
    hold on;
    polarplot(theta_full, vals, 'o', 'MarkerSize', 10, ...
        'MarkerFaceColor', plot_colors(i, :), ...
        'MarkerEdgeColor', 'none');
end

% 也画无增强作为基准
vals_base = [data(1, :), data(1, 1)];
polarplot(theta_full, vals_base, '--', 'LineWidth', 1.2, ...
    'Color', [0.6 0.6 0.6]);
polarplot(theta_full, vals_base, 's', 'MarkerSize', 7, ...
    'MarkerFaceColor', [0.6 0.6 0.6], 'MarkerEdgeColor', 'none');

set(gca, 'ThetaTick', rad2deg(theta), 'ThetaTickLabel', metrics, ...
    'FontSize', 12, 'LineWidth', 1.1, 'RLim', [0.70, 1.00]);
leg = legend([plot_names, 'No Aug (Baseline)'], ...
    'Location', 'northeastoutside', 'FontSize', 11, 'Box', 'off');
leg.Title.String = 'Model';
leg.Title.FontSize = 12;
leg.Title.FontWeight = 'bold';
title('Radar Chart: Multi-Metric Model Comparison', ...
    'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.25]);

%% ========================================================================
%% 图4: 泰勒图 (Taylor Diagram)
%% 此实现基于分类指标相对于完美预测(1,1,1,1,1)的偏差
%% ========================================================================
figure('Position', [50, 150, 720, 620], 'Color', 'w');

% 以 No Augmentation 为参考基准 (有非零方差, 可计算相关系数)
ref_model_idx = 1;
ref = data(ref_model_idx, :);
std_ref = std(ref);

% 计算每个模型相对于ref的相关系数、中心化RMSD和标准差
rms_diff  = zeros(n_models, 1);
corr_val  = zeros(n_models, 1);
std_val   = zeros(n_models, 1);

for i = 1:n_models
    rms_diff(i) = sqrt(mean(((data(i, :) - mean(data(i,:))) - (ref - mean(ref))).^2));
    corr_val(i) = corr(data(i, :)', ref');
    std_val(i)  = std(data(i, :));
end

std_ratio = std_val / std_ref;   % 归一化标准差
rms_ratio = rms_diff / std_ref;   % 归一化RMSD

% ---- 绘制泰勒图 ----
hold on;

% RMS差等值线 (归一化后)
rms_levels = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5];
for k = 1:length(rms_levels)
    rr = rms_levels(k);
    rho_vals = linspace(0, 1, 300);
    sigma_vals = zeros(size(rho_vals));
    for j = 1:length(rho_vals)
        rho = rho_vals(j);
        disc = (2*rho)^2 - 4*(1 - rr^2);
        if disc >= 0
            s1 = (2*rho + sqrt(disc)) / 2;
            s2 = (2*rho - sqrt(disc)) / 2;
            if s1 > 0, sigma_vals(j) = s1; else sigma_vals(j) = s2; end
        else
            sigma_vals(j) = NaN;
        end
    end
    valid = ~isnan(sigma_vals) & sigma_vals > 0 & sigma_vals < max(std_ratio) * 1.4;
    if sum(valid) > 2
        [tx, ty] = pol2cart(acos(rho_vals(valid)), sigma_vals(valid));
        plot(tx, ty, ':', 'LineWidth', 0.5, 'Color', [0.65 0.65 0.65]);
        idx_mid = round(sum(valid) / 2);
        if idx_mid > 0 && idx_mid <= length(tx)
            text(tx(idx_mid), ty(idx_mid) - 0.01, sprintf('%.1f', rr), ...
                'FontSize', 7, 'Color', [0.5 0.5 0.5]);
        end
    end
end

% 相关系数弧线
corr_ticks = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99];
rmax = max(std_ratio) * 1.2;
for c = corr_ticks
    ang = acos(c);
    [xl, yl] = pol2cart(ang, linspace(0, rmax, 80));
    plot(xl, yl, '-', 'LineWidth', 0.35, 'Color', [0.72 0.72 0.72]);
    text(rmax * cos(ang) + 0.02, rmax * sin(ang) + 0.01, ...
        sprintf('%.1f', c), 'FontSize', 8, 'Color', [0.4 0.4 0.4]);
end

% 标准差弧线
std_ticks = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5];
theta_full = linspace(0, pi/2, 100);
for s = std_ticks
    if s <= rmax
        [xa, ya] = pol2cart(theta_full, s * ones(size(theta_full)));
        plot(xa, ya, '-', 'LineWidth', 0.35, 'Color', [0.72 0.72 0.72]);
    end
end

% 参考点 (No Augmentation, 始终在 std_ratio=1, corr=1)
plot(0, 1, 'k*', 'MarkerSize', 16, 'LineWidth', 1.5);
text(0.02, 1.03, 'REF\n(No Aug)', 'FontSize', 9, ...
    'FontWeight', 'bold', 'Color', [0.15 0.15 0.15]);

% 绘制各个模型点
marker_styles = {'s', '^', 'd', 'o', 'v', 'p'};
marker_sizes  = [100, 100, 100, 140, 100, 200];

for i = 1:n_models
    ang = acos(max(min(corr_val(i), 1), 0));
    [px, py] = pol2cart(ang, std_ratio(i));
    h = scatter(px, py, marker_sizes(i), colors(i, :), ...
        marker_styles{i}, 'filled', 'MarkerEdgeColor', [0.15 0.15 0.15], ...
        'LineWidth', 1.2);
    off_x = 0.04 * (mod(i, 2) * 2 - 1);
    off_y = 0.04 * (1 - mod(i, 2));
    text(px + off_x, py + off_y, model_names{i}, ...
        'FontSize', 9, 'FontWeight', 'bold', 'Color', colors(i, :));
end

xlim([-0.02, rmax]);
ylim([-0.02, rmax]);
axis equal;
set(gca, 'FontSize', 11, 'LineWidth', 1, 'TickDir', 'out', ...
    'Color', 'none', 'XColor', [0.35 0.35 0.35], 'YColor', [0.35 0.35 0.35]);
xlabel('Standard Deviation (normalized to No Aug.)', ...
    'FontSize', 12, 'FontWeight', 'bold');
ylabel('Standard Deviation (normalized to No Aug.)', ...
    'FontSize', 12, 'FontWeight', 'bold');
title('Taylor Diagram: Model Metric Profiles Relative to Baseline', ...
    'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.25]);

text(rmax * 0.55, 0.03, '← Correlation →', ...
    'FontSize', 9, 'Color', [0.45 0.45 0.45]);
text(rmax * 0.82, 0.07, 'Normalized\n  RMSD', ...
    'FontSize', 9, 'Color', [0.45 0.45 0.45]);

box off;
hold off;

%% ========================================================================
%% 图5: 消融瀑布图 — 各增强方法相对于无增强的 Accuracy 提升
%% ========================================================================
figure('Position', [50, 150, 720, 500], 'Color', 'w');

acc_values = data(:, 1);
acc_base = acc_values(1);
acc_gain = acc_values - acc_base;

% 瀑布图
h_bar = bar(1:n_models, acc_gain, 0.6, 'FaceColor', 'flat', ...
    'EdgeColor', 'none');
for i = 1:n_models
    h_bar.CData(i, :) = colors(i, :);
end

% 标注
for i = 1:n_models
    if acc_gain(i) > 0
        text(i, acc_gain(i) + 0.003, sprintf('+%.4f', acc_gain(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 12, ...
            'FontWeight', 'bold', 'Color', [0.1 0.4 0.1]);
    else
        text(i, acc_gain(i) - 0.005, sprintf('%.4f', acc_gain(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 12, ...
            'FontWeight', 'bold', 'Color', [0.6 0.2 0.2]);
    end
end

% 基线
yline(0, '-', 'LineWidth', 2.5, 'Color', [0.5 0.5 0.5]);

set(gca, 'XTick', 1:n_models, 'XTickLabel', model_names, ...
    'FontSize', 11.5, 'LineWidth', 1, 'TickDir', 'out');
ylabel(sprintf('Accuracy Improvement over Baseline\n(Baseline = %.4f)', acc_base), ...
    'FontSize', 12, 'FontWeight', 'bold');
title('Ablation Study: Accuracy Gain Relative to No Augmentation', ...
    'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.25]);

% 箭头标注最优
[best_gain, best_i] = max(acc_gain);
annotation('textarrow', ...
    [0.78, 0.85], [0.82, 0.65], ...
    'String', sprintf('Best: +%.4f', best_gain), ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.85 0.33 0.1]);

grid on; set(gca, 'GridAlpha', 0.1);

%% ========================================================================
%% 保存
%% ========================================================================
% exportgraphics(figure(1), 'Fig_Model_BarChart.pdf', 'ContentType', 'vector', 'Resolution', 300);
% exportgraphics(figure(2), 'Fig_Model_Heatmap.pdf', 'ContentType', 'vector', 'Resolution', 300);
% exportgraphics(figure(3), 'Fig_Model_Radar.pdf',   'ContentType', 'vector', 'Resolution', 300);
% exportgraphics(figure(4), 'Fig_Model_Taylor.pdf',  'ContentType', 'vector', 'Resolution', 300);
% exportgraphics(figure(5), 'Fig_Model_Ablation.pdf','ContentType', 'vector', 'Resolution', 300);

fprintf('绘图完成。共生成 5 张图。\n');
fprintf('  Fig 1: 分组柱状图\n');
fprintf('  Fig 2: 热力图\n');
fprintf('  Fig 3: 雷达图\n');
fprintf('  Fig 4: 泰勒图\n');
fprintf('  Fig 5: 消融瀑布图\n');
