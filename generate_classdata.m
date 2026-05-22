function [SyntheticData1,Synthetic_label1]=generate_classdata(origin_data,origin_data_label,methodchoose,get_mutiple)
   %生成分类样本数据
   rng('default')
   if methodchoose==1
       %随机过采样样本生成
       [SyntheticData,Synthetic_label]=generate_classdata_SMOTE(origin_data,origin_data_label,get_mutiple);
   elseif methodchoose==2
       %GAN数据生成
       [SyntheticData,Synthetic_label]=generate_classdata_GAN(origin_data,origin_data_label,get_mutiple);
   elseif methodchoose==3
       %GMM高斯混合模型数据生成
       [SyntheticData,Synthetic_label]=generate_classdata_GMM(origin_data,origin_data_label,get_mutiple);
     elseif methodchoose==4
       %GMM高斯混合模型数据生成
       [SyntheticData,Synthetic_label]=generate_classdata_LSTM(origin_data,origin_data_label,get_mutiple);     
     elseif methodchoose==5
       %GMM高斯混合模型数据生成
       [SyntheticData,Synthetic_label]=generate_classdata_diffusion(origin_data,origin_data_label,get_mutiple);     
      
   end
  
   %% 统计生成数据前后数据分布图
   unique_class=unique(origin_data_label);
   label_get=[];unique_str_label=[];
   for i=1:length(unique_class)
       label_get(i)=length(find(origin_data_label==unique_class(i)));
       unique_str_label{1,i}=['class',num2str(i)];
   end
   figure('Position',[300,300,800,300])
   subplot(1,2,1)

           
   bar_plot_f=bar(1:length(label_get),label_get,0.75);   %  重要性衡量
   bar_plot_f.FaceColor = 'flat';
   for i=1:length(unique_class)
       bar_plot_f.CData(i,:)=[0.6314    0.6627    0.8157];
       %                     bar_plot_f(i).FaceColor=color_get(1+i*(floor(length(color_get)/length(imp))-1),:);
   end
   xtips1 = bar_plot_f.XEndPoints;
   ytips1 = bar_plot_f.YEndPoints;
   labels1 = string(bar_plot_f.YData);
   text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
       'VerticalAlignment','bottom')
   % index_name_plot=data_biao1(1:end-1);
   xticks(1:length(label_get))
   xticklabels(unique_str_label)
   ylim([0,1.1*max(label_get)])
   title('原样本数');
   ylabel('num');
   set(gca,"FontSize",11,"LineWidth",1)
   box off

   % bar(1:length(label_get),label_get,0.75)
   % set(gca,'FontSize',12,'LineWidth',1.2)
   % ylabel('原样本数')
   % xticks(1:length(label_get))
   % xticklabels(unique_str_label)

   
   % 为了保持样本数平衡 原来数据多的样本少取，少的样本多取
   label_syn_get=[]; %生成数据统计
   for i=1:length(unique_class)
       label_syn_get{i}=(find(Synthetic_label==unique_class(i)));       
   end
   label_get_rio=label_get/sum(label_get);
   label_get_rio1=1./label_get_rio;
   label_get_rio2=label_get_rio1/max(label_get_rio1);

   SyntheticData1=[];Synthetic_label1=[];
   for i=1:length(unique_class)
       data_syno_get=label_syn_get{i};   
       data_syno_get1=data_syno_get(1:round(label_get_rio2(i)*size(data_syno_get,1)));
       SyntheticData1=[SyntheticData1;SyntheticData(data_syno_get1,:)];
       Synthetic_label1=[Synthetic_label1;Synthetic_label(data_syno_get1,:)];
   end
   data_label_new=[origin_data_label;Synthetic_label1];

   subplot(1,2,2)
   for i=1:length(unique_class)
       label_get1(i)=length(find(data_label_new==unique_class(i)));
   end
      bar_plot_f1=bar(1:length(label_get1),label_get1,0.75);   %  重要性衡量
   bar_plot_f1.FaceColor = 'flat';
   for i=1:length(unique_class)
       bar_plot_f1.CData(i,:)=[0.5882    0.8000    0.7961];
       %                     bar_plot_f(i).FaceColor=color_get(1+i*(floor(length(color_get)/length(imp))-1),:);
   end
   xtips1 = bar_plot_f1.XEndPoints;
   ytips1 = bar_plot_f1.YEndPoints;
   labels1 = string(bar_plot_f1.YData);
   text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
       'VerticalAlignment','bottom')
   % index_name_plot=data_biao1(1:end-1);
   xticks(1:length(label_get1))
   xticklabels(unique_str_label)
   title('增样后总样本数');
   ylabel('num');
   set(gca,"FontSize",11,"LineWidth",1)
   ylim([0,1.1*max(label_get1)])
   box off
   % bar(1:length(label_get1),label_get1,0.75)
   % set(gca,'FontSize',12,'LineWidth',1.2)
   % ylabel('增样后总样本数')
   % xticks(1:length(label_get1))
   % xticklabels(unique_str_label)
end

%% SMOTE随机过采样数据生成
function [SyntheticData,Synthetic_label]=generate_classdata_SMOTE(origin_data,origin_data_label,get_mutiple)
% get_mutiple  %生成样本数是原数据的多少倍
%随机过采样部分代码
unique_class=unique(origin_data_label);
unique_class1=unique_class;
% [~,max_index]=max(label_get);
% unique_class1(max_index)=[];
% x_aug=origin_data;
% y_aug=origin_data_label;
% y_aug1=origin_data_label;
x_aug=[];
y_aug=[];
y_aug1=[];

for i=1:length(unique_class1)
    flabel_get=unique_class1(i);
    % 假设数据矩阵 X 中的最后一列为类别标签，0 表示多数类，1 表示少数类

    % 调用 SMOTE 函数
    numMinority = sum(origin_data_label == flabel_get); % 少数类样本数量
    % get_mutiple=ceil(sum(train_y_feature_label == max_index)/numMinority)-1;
    numNeighbors = 5; % 设置邻居数量
    newData = SMOTE(origin_data(origin_data_label == flabel_get, :), numMinority, numNeighbors,get_mutiple);

    % 将生成的合成样本与原始数据合并
    x_aug = [x_aug; newData];
    y_aug = [y_aug; flabel_get*ones(size(newData, 1), 1)]; % 将合成样本标记为少数类（1）
    y_aug1=[y_aug1; flabel_get*ones(size(newData, 1), 1)];
end
% train_x_feature_label=x_aug;
% train_y_feature_label=y_aug;
SyntheticData=x_aug;
Synthetic_label=y_aug;
end

%%
function [SyntheticData,Synthetic_label]=generate_classdata_diffusion(origin_data,origin_data_label,get_mutiple)

origin_data_label_unique=unique(origin_data_label);
origin_data_label_unique_index_class=[];
origin_data_label_unique_index=[];
for i =1:length(origin_data_label_unique)
    origin_data_label_unique_index(i)=length(find(origin_data_label==origin_data_label_unique(i)));
    origin_data_label_unique_index_class{1,i}=find(origin_data_label==origin_data_label_unique(i));
end
origin_data_label_unique_index_class1=[];
data_min=min(origin_data_label_unique_index);
data_get_index=[];
for i =1:length(origin_data_label_unique)
    index_get=origin_data_label_unique_index_class{1,i};

    origin_data_label_unique_index_class1{1,i}=index_get(randperm(length(index_get),data_min));
    data_get_index=[data_get_index,origin_data_label_unique_index_class1{1,i}];
end
origin_data_label=origin_data_label(data_get_index,:);
origin_data=origin_data(data_get_index,:);

T=100;
% Y_onehot = full(ind2vec(origin_data_label'));  % 300×3 矩阵
Y_onehot = full(ind2vec(origin_data_label'))';  % 300×3 矩阵
Z_norm=[origin_data,Y_onehot];
D = size(Z_norm,2);
beta = linspace(1e-4, 0.02, T);
alpha = 1 - beta;
alpha_bar = cumprod(alpha);
Xtrain = [];
Ytrain = [];
for i = 1:length(origin_data_label)
    for t = 1:T
        z_i = Z_norm(i,:);
        noise = randn(1, D);

        z_noisy = sqrt(alpha_bar(t)) * z_i + sqrt(1 - alpha_bar(t)) .* noise;

        Xtrain(end+1,:) = z_noisy;     % 当前带噪样本
        Ytrain(end+1,:) = noise;       % 噪声作为监督目标
    end
end

% Step 4: 网络定义
layers = [
    featureInputLayer(D)
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(D)       % 多维输出
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 60, ...
    'MiniBatchSize',2^(ceil(log(length(origin_data_label)*T/10))), ...
    'Shuffle','every-epoch', ...
    'Verbose', false, ...
    'Plots','training-progress');
% options = trainingOptions('adam', ...
%     'MaxEpochs', 60, ...
%     'Shuffle','every-epoch', ...
%     'Verbose', false, ...
%     'Plots','training-progress');
% 训练网络
net = trainNetwork(Xtrain, Ytrain, layers, options);

% Step 5: 生成数据（从纯噪声出发）
n_gen = round(get_mutiple*length(origin_data_label));
z_gen = randn(n_gen, D);   % 初始为纯噪声

for t = T:-1:1
    a = alpha(t);
    a_bar = alpha_bar(t);

    z_pred = predict(net, z_gen);  % 预测噪声

    z_gen = (z_gen - sqrt(1 - a_bar) .* z_pred) ./ sqrt(a);
end

SyntheticData=z_gen(:,1:size(origin_data,2));
[~,max_idl]=max(z_gen(:,size(origin_data,2)+1:end)');

% [~,max_idl]=max(z_gen(:,1:size(origin_data,2))');
Synthetic_label=max_idl';

end

%% GAN数据生成
function [SyntheticData,Synthetic_label]=generate_classdata_GAN(origin_data,origin_data_label,get_mutiple)
  %GAN生成分类数据
original_data=reshape(origin_data,1,[]); % Preprocessing - convert matrix to vector

% Define the generator and discriminator networks
generator = @(z) original_data; % Identity mapping for simplicity
discriminator = @(x) (x - original_data); % Z-score normalization

% Training parameters
num_samples = 500;
num_epochs = 4;
batch_size = 160;
learning_rate = 0.01;
% Each run generates samples equal with number of samples in the origianl
% data. So, 3 runs means original data * 3.
Runs= get_mutiple;  %生成样本数是原数据的多少倍

for i=1:Runs
    % Training loop
    for epoch = 1:num_epochs
        for batch = 1:num_samples/batch_size
            % Generate noise samples for the generator
            noise = randn(batch_size, 1);
            % Generate synthetic data using the generator
            synthetic_data = generator(noise);
            % Train the discriminator to distinguish real from synthetic data
            discriminator_loss = mean((discriminator(synthetic_data) - noise).^2);
            % Update the generator to fool the discriminator
            generator_loss = mean((discriminator(generator(noise)) - noise).^2);
            % Update the generator and discriminator parameters
            generator = @(z) generator(z) - learning_rate * generator_loss;
            discriminator = @(x) discriminator(x) - learning_rate * discriminator_loss;
        end
        Run = [' Epoch "',num2str(epoch)];
        disp(Run);
    end
    %
    % Generate synthetic data using the trained generator
    noise_samples = randn(num_samples/4, 1);
    synthetic_data1= generator(noise_samples);
    Syn(i,:)=synthetic_data1;
    % Run2 = [' Run "',num2str(Runs)];
    % disp(Run2);
end

% Converting cell to matrix
S = size(Syn(Runs)); SO = size (origin_data); SF = SO (1,2); SO = SO (1,1);
for i=1:Runs
    Syn2{i}=reshape(Syn(i,:),[SO,SF]);
    Syn2{i}(:,end+1)=origin_data_label;
end
Synthetic3 = cell2mat(Syn2');
SyntheticData=Synthetic3(:,1:end-1);
Synthetic_label=Synthetic3(:,end);

end
%%
function [SyntheticData,Synthetic_label]=generate_classdata_GMM(origin_data,origin_data_label,get_mutiple)
%采用高斯混合模型进行样本生成
NoofSynthetic=get_mutiple*length(origin_data_label);

% 高斯混合模型(GMM)拟合原始数据
GMModel1 = fitgmdist(origin_data,length(unique(origin_data_label)));

% 生成数据 (SDG)
SyntheticData = random(GMModel1,NoofSynthetic);

% 用K-means聚类方法获取合成生成数据的标签
Synthetic_label= kmeans(SyntheticData,length(unique(origin_data_label)));
end

%%
function [SyntheticData,Synthetic_label]=generate_classdata_LSTM(origin_data,origin_data_label,get_mutiple)
% 采用LSTM进行数据生成  
Runs= get_mutiple;  %生成样本数是原数据的多少倍
data=reshape(origin_data,1,[]); 
for Num_N=1:Runs
    
    mu = mean(data);
    sig = std(data);
    dataTrainStandardized = (data - mu) / sig;
    XTrain = dataTrainStandardized;
    YTrain = dataTrainStandardized;
    % Define LSTM Network Architecture
    numFeatures = 1;
    numResponses = 1;
    numHiddenUnits = 128;
    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(numResponses)
        regressionLayer];
    options = trainingOptions('adam', ...
        'MaxEpochs',30, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.009, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',256, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0);
    net = trainNetwork(XTrain,YTrain,layers,options);

    % Forecast Future Time Steps
    dataTestStandardized = (data - mu) / sig;
    XTest = dataTestStandardized;
    net = predictAndUpdateState(net,XTrain);
    [net,YPred] = predictAndUpdateState(net,YTrain(end));
    numTimeStepsTest = numel(XTest);
    for i = 2:numTimeStepsTest
        [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
    end

    % Update Network State with Observed Values
    net = resetState(net);
    net = predictAndUpdateState(net,XTrain);
    YPred = [];
    numTimeStepsTest = numel(XTest);
    for i = 1:numTimeStepsTest
        [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
    end
    YPred = sig*YPred + mu;
    Synthetic{Num_N}=YPred;
    rmse = sqrt(mean((YPred-data).^2))*0.00001;
    RMSE(Num_N)=rmse;
end
Synthetic=Synthetic';% Converting cell to matrix (the last time)

Synthetic2 = cell2mat(Synthetic);% Converting matrix to cell
Synthetic2=Synthetic2';

SO = size(origin_data); SF =SO(1,2); SO = SO (1,1); 
for i = 1 : Runs
    Generated1{i}=reshape(Synthetic2(:,i),[SO,SF]);
    Generated1{i}(:,end+1)=origin_data_label;
end
Synthetic3 = cell2mat(Generated1');
SyntheticData=Synthetic3(:,1:end-1);
Synthetic_label=Synthetic3(:,end);
end
%%
function syntheticData = SMOTE(X, numMinority, numNeighbors,get_muti)
    % X: 少数类样本特征矩阵
    % numMinority: 少数类样本数量
    % numNeighbors: 邻居数量
    
    % 计算 k 近邻
    [idx] = knnsearch(X, X, 'K', numNeighbors);
    
    % 生成合成样本
%     syntheticData =[];
    for i = 1:round(numMinority * get_muti/numNeighbors)
        for j = 1:numNeighbors
            if i<size(X,1)
            
            neighbor = X(idx(i, j), :);
            gap = neighbor - X(i, :);
            alpha = rand(); % 随机选择一个权重
            syntheticData((i - 1) * numNeighbors + j, :) = X(i, :) + alpha * gap;
            end
        end
    end
end