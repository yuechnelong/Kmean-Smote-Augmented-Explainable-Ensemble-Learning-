clc;clear;close all;	
load('R_18_May_2026_17_19_57.mat')	
random_seed=G_out_data.random_seed ;  %界面设置的种子数 	
rng(random_seed)  %固定随机数种子 	
	
data_str=G_out_data.data_path_str ;  %读取数据的路径 	
dataO=readtable(data_str,'VariableNamingRule','preserve'); %读取数据 	
data1=dataO(:,2:end);test_data=table2cell(dataO(1,2:end));	
for i=1:length(test_data)	
      if ischar(test_data{1,i})==1	
          index_la(i)=1;     %char类型	
      elseif isnumeric(test_data{1,i})==1	
          index_la(i)=2;     %double类型	
      else	
        index_la(i)=0;     %其他类型	
     end 	
end	
index_char=find(index_la==1);index_double=find(index_la==2);	
 %% 数值类型数据处理	
if length(index_double)>=1	
    data_numshuju=table2array(data1(:,index_double));	
    index_double1=index_double;	
	
    index_double1_index=1:size(data_numshuju,2);	
    data_NAN=(isnan(data_numshuju));    %找列的缺失值	
    num_NAN_ROW=sum(data_NAN);	
    index_NAN=num_NAN_ROW>round(0.2*size(data1,1));	
    index_double1(index_NAN==1)=[]; index_double1_index(index_NAN==1)=[];	
    data_numshuju1=data_numshuju(:,index_double1_index);	
    data_NAN1=(isnan(data_numshuju1));  %找行的缺失值	
    num_NAN__COL=sum(data_NAN1');	
    index_NAN1=num_NAN__COL>0;	
    index_double2_index=1:size(data_numshuju,1);	
    index_double2_index(index_NAN1==1)=[];	
    data_numshuju2=data_numshuju1(index_double2_index,:);	
    index_need_last=index_double1;	
 else	
    index_need_last=[];	
    data_numshuju2=[];	
end	
%% 文本类型数据处理	
	
data_shuju=[];	
 if length(index_char)>=1	
  for j=1:length(index_char)	
    data_get=table2array(data1(index_double2_index,index_char(j)));	
    data_label=unique(data_get);	
    if j==length(index_char)	
       data_label_str=data_label ;	
    end    	
	
     for NN=1:length(data_label)	
            idx = find(ismember(data_get,data_label{NN,1}));  	
            data_shuju(idx,j)=NN; 	
     end	
  end	
 end	
label_all_last=[index_char,index_need_last];	
[~,label_max]=max(label_all_last);	
 if(label_max==length(label_all_last))	
     str_label=0; %标记输出是否字符类型	
     data_all_last=[data_shuju,data_numshuju2];	
     label_all_last=[index_char,index_need_last];	
 else	
    str_label=1;	
    data_all_last=[data_numshuju2,data_shuju];	
    label_all_last=[index_need_last,index_char];     	
 end	
 data=data_all_last;	
 data_biao_all=data1.Properties.VariableNames;	
 for j=1:length(label_all_last)	
    data_biao{1,j}=data_biao_all{1,label_all_last(j)};	
 end	
	
% 异常值检测	
	
 unique_index_ab=G_out_data.unique_index_ab; 	
 data(:,unique_index_ab)=[];	
 label_all_last(unique_index_ab)=[];	
 data_biao1=data_biao; data_biao1(unique_index_ab)=[]; 	
	
data=data;	
	
%%  特征处理 特征选择或者降维	
	
 A_data1=data;	
	
 select_feature_num=G_out_data.select_feature_num1;   %特征选择的个数	
	
data_select=A_data1;	
feature_need_last=1:size(A_data1,2)-1;	
	
	
	
%% 数据划分	
x_feature_label=data_select(:,1:end-1);    %x特征	
y_feature_label=data_select(:,end);          %y标签	
index_label1=randperm(size(x_feature_label,1));	
index_label=G_out_data.spilt_label_data;  % 数据索引	
if isempty(index_label)	
     index_label=index_label1;	
end	
spilt_ri=G_out_data.spilt_rio;  %划分比例 训练集:验证集:测试集	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          %训练集个数	
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数	
 %训练集，验证集，测试集	
train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
%Zscore 标准化	
%训练集	
x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化  	
%验证集	
vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    %验证数据标准化	
vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  %验证数据标准化	
%测试集	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 测试数据标准化	
test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 测试数据标准化  	
	
%% 参数设置	
num_pop=5;   %种群数量	
num_iter=10;   %种群迭代数	
method_mti=G_out_data.method_mti1;   %优化方法	
BO_iter=G_out_data.BO_iter;   %贝叶斯迭代次数	
min_batchsize=G_out_data.min_batchsize;   %batchsize	
max_epoch=G_out_data.max_epoch1;   %maxepoch	
hidden_size=G_out_data.hidden_size1;   %hidden_size	
attention_label=G_out_data.attention_label;   %注意力机制标签	
attention_head=G_out_data.attention_head;   %注意力机制设置	
	
%% 数据增强部分	
get_mutiple=G_out_data.get_mutiple;  %数据增加倍数	
methodchoose=5; 	
origin_data=[train_x_feature_label_norm;vaild_x_feature_label_norm]; 	
origin_data_label=[train_y_feature_label;vaild_y_feature_label]; 	
[SyntheticData,Synthetic_label]=generate_classdata(origin_data,origin_data_label,methodchoose,get_mutiple); 	
% 绘制生成后数据样本图	
figure_data_generate(origin_data,SyntheticData,origin_data_label,Synthetic_label)	
X_new_DATA=[origin_data;SyntheticData];             %生成的X特征数据	
Y_new_DATA=[origin_data_label;Synthetic_label];  %生成的Y标签数据	
	
syn_spilt=round(spilt_ri(1)/(spilt_ri(1)+spilt_ri(2))*length(Y_new_DATA));	
syn_index=randperm(length(Y_new_DATA));	
%以下将生成的数据随机分配到训练集和验证集中	
train_x_feature_label_norm=X_new_DATA(syn_index(1:syn_spilt),:);	
vaild_x_feature_label_norm=X_new_DATA(syn_index(syn_spilt+1:end),:);	
train_y_feature_label=Y_new_DATA(syn_index(1:syn_spilt),:);	
vaild_y_feature_label=Y_new_DATA(syn_index(syn_spilt+1:end),:);	
train_x_feature_label=train_x_feature_label_norm.*x_sig+x_mu;	
vaild_x_feature_label=vaild_x_feature_label_norm.*x_sig+x_mu;	
%数据生成输出数据	
train_x_feature_label_aug=(train_x_feature_label_norm.*x_sig)+x_mu;	
vaild_x_feature_label_aug=(vaild_x_feature_label_norm.*x_sig)+x_mu;	
%总体生成数据+原数据保存在以下的 augdata_all 数据里面	
augdata_all=[train_x_feature_label_aug,train_y_feature_label;vaild_x_feature_label_aug,vaild_y_feature_label;test_x_feature_label,test_y_feature_label];	
	
	
%% 算法处理块	
	
	
	
  t1=clock;	
disp('优化CBiLSTM-RF分类')	
  	
 p_train1=reshape(train_x_feature_label_norm',size(train_x_feature_label_norm,2),1,1,size(train_x_feature_label,1));  	
  	
  	
 p_vaild1=reshape(vaild_x_feature_label_norm',size(vaild_x_feature_label_norm,2),1,1,size(vaild_x_feature_label,1));  	
 	
  	
 p_test1=reshape(test_x_feature_label_norm',size(test_x_feature_label_norm,2),1,1,size(test_x_feature_label,1)); 	
  	
	
method_mti='SSA麻雀搜索算法';		
 [Model_CBiLSTM,~,fitness,Loss,pop] = optimize_fitCCNN_BiLSTM_att(p_train1, categorical(train_y_feature_label),p_vaild1,(vaild_y_feature_label),num_pop,num_iter,method_mti,max_epoch,min_batchsize,attention_label,attention_head)  ;	
[Model_RF,fitness] = optimize_fitctreebag(train_x_feature_label_norm,train_y_feature_label,vaild_x_feature_label_norm,vaild_y_feature_label,num_pop,num_iter,method_mti);     	 	
	
 y_train_predict_RF= RF_process((predict(Model_RF,train_x_feature_label_norm))); 	
 y_train_predict_CBiLSTM= double(classify(Model_CBiLSTM,p_train1));	
 y_vaild_predict_RF= RF_process((predict(Model_RF,vaild_x_feature_label_norm)));	
 y_vaild_predict_CBiLSTM= double(classify(Model_CBiLSTM,p_vaild1)); 	
 y_test_predict_RF= RF_process((predict(Model_RF,test_x_feature_label_norm)));	
 y_test_predict_CBiLSTM= double(classify(Model_CBiLSTM,p_test1)); 	
 	
AUC_CBiLSTM=sum((y_vaild_predict_CBiLSTM == vaild_y_feature_label)) /length(vaild_y_feature_label) ; 	
AUC_RF=sum((y_vaild_predict_RF == vaild_y_feature_label)) /length(vaild_y_feature_label) ; 	
  	
 if abs((AUC_RF-AUC_CBiLSTM))<0.05 	
      quan1=AUC_RF/(AUC_CBiLSTM+AUC_RF);quan2=AUC_CBiLSTM/(AUC_RF+AUC_CBiLSTM); 	
else 	
     if(AUC_CBiLSTM>AUC_RF)	
       quan1=0;quan2=1;     	
     else  	
     quan1=1;quan2=0; 	
     end 	
  end	
  	
 Mdl{1,1}=Model_RF; Mdl{1,2}=Model_CBiLSTM; Mdl{1,3}=[quan1,quan2]; 	
 y_test_predict_CBiLSTM_RF=round(quan1*y_test_predict_RF+quan2*y_test_predict_CBiLSTM);y_test_predict=y_test_predict_CBiLSTM_RF;	
y_train_predict_CBiLSTM_RF=round(quan1*y_train_predict_RF+quan2*y_train_predict_CBiLSTM);y_train_predict=y_train_predict_CBiLSTM_RF; 	
y_vaild_predict_CBiLSTM_RF=round(quan1*y_vaild_predict_RF+quan2*y_vaild_predict_CBiLSTM);y_vaild_predict=y_vaild_predict_CBiLSTM_RF; 	
	
	
	
 disp(['RF验证集正确率：',num2str(AUC_RF)])       	
 disp(['RF验证集正确率：',num2str(AUC_RF)])     	
  disp(['CBiLSTM-RF结合权重：',num2str([quan1,quan2])]) 	
 CBiLSTM_RF_AUC=sum((y_test_predict_CBiLSTM_RF == test_y_feature_label)) /length(test_y_feature_label) ;  	
 disp(['CBiLSTM-RF测试集正确率：',num2str(CBiLSTM_RF_AUC)])	
 RF_AUC=sum((y_test_predict_RF == test_y_feature_label))/length(test_y_feature_label) ; 	
 disp(['RF测试集正确率：',num2str(RF_AUC)])    	
 CBiLSTM_AUC=sum((y_test_predict_CBiLSTM == test_y_feature_label))/length(test_y_feature_label) ;	
 disp(['CBiLSTM测试集正确率：',num2str(CBiLSTM_AUC)])   	
 	
 t2=clock; 	
 Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));  	
   	
 analyzeNetwork(Model_CBiLSTM)  	
 figure  	
subplot(2, 1, 1)	
plot(1 : length(Loss.TrainingAccuracy), Loss.TrainingAccuracy, '-', 'LineWidth', 1)	
xlabel('迭代次数'); ylabel('准确率');legend('训练集准确率');title ('训练集准确率迭代曲线');grid;set(gcf,'color','w')	
	
subplot(2, 1, 2)	
plot(1 : length(Loss.TrainingLoss), Loss.TrainingLoss, '-', 'LineWidth', 1)	
xlabel('迭代次数');ylabel('损失函数');legend('训练集损失值');title ('训练集损失函数曲线');grid;set(gcf,'color','w')	
	
	
disp(['运行时长: ',num2str(Time)])	
confMat_train = confusionmat(train_y_feature_label,y_train_predict);	
TP_train = diag(confMat_train);      TP_train=TP_train'; % 被正确分类的正样本 True Positives	
FP_train = sum(confMat_train, 1)  - TP_train;  %被错误分类的正样本 False Positives	
FN_train = sum(confMat_train, 2)' - TP_train;  % 被错误分类的负样本 False Negatives	
TN_train = sum(confMat_train(:))  - (TP_train + FP_train + FN_train);  % 被正确分类的负样本 True Negatives	
	
disp('训练集*******************************************************************************')	
accuracy_train = sum(TP_train) / sum(confMat_train(:)); accuracy_train(isnan(accuracy_train))=0; disp(['训练集accuracy：',num2str(mean(accuracy_train))])% Accuracy 	
precision_train = TP_train ./ (TP_train + FP_train); precision_train(isnan(precision_train))=0; disp(['训练集precision_train：',num2str(mean(precision_train))]) % Precision	
recall_train = TP_train ./ (TP_train + FN_train);recall_train(isnan(recall_train))=0; disp(['训练集recall_train：',num2str(mean(recall_train))])  % Recall / Sensitivity	
F1_score_train = 2 * (precision_train .* recall_train) ./ (precision_train + recall_train); F1_score_train(isnan(F1_score_train))=0;  disp(['训练集F1_score_train：',num2str(mean(F1_score_train))])   % F1 Score	
specificity_train = TN_train ./ (TN_train + FP_train); specificity_train(isnan(specificity_train))=0; disp(['训练集specificity_train：',num2str(mean(specificity_train))])  % Specificity	
	
disp('验证集********************************************************************************')	
confMat_vaild = confusionmat(vaild_y_feature_label,y_vaild_predict);	
TP_vaild = diag(confMat_vaild);      TP_vaild=TP_vaild'; % 被正确分类的正样本 True Positives	
FP_vaild = sum(confMat_vaild, 1)  - TP_vaild;  %被错误分类的正样本 False Positives	
FN_vaild = sum(confMat_vaild, 2)' - TP_vaild;  % 被错误分类的负样本 False Negatives	
TN_vaild = sum(confMat_vaild(:))  - (TP_vaild + FP_vaild + FN_vaild);  % 被正确分类的负样本 True Negatives	
accuracy_vaild = sum(TP_vaild) / sum(confMat_vaild(:)); accuracy_vaild(isnan(accuracy_vaild))=0; disp(['验证集accuracy：',num2str(accuracy_vaild)])% Accuracy 	
precision_vaild = TP_vaild ./ (TP_vaild + FP_vaild); precision_vaild(isnan(precision_vaild))=0; disp(['验证集precision_vaild：',num2str(mean(precision_vaild))]) % Precision	
recall_vaild = TP_vaild ./ (TP_vaild + FN_vaild); recall_vaild(isnan(recall_vaild))=0;  disp(['验证集recall_vaild：',num2str(mean(recall_vaild))])  % Recall / Sensitivity	
F1_score_vaild = 2 * (precision_vaild .* recall_vaild) ./ (precision_vaild + recall_vaild);  F1_score_vaild(isnan(F1_score_vaild))=0;  disp(['验证集F1_score_vaild：',num2str(mean(F1_score_vaild))])   % F1 Score	
specificity_vaild = TN_vaild ./ (TN_vaild + FP_vaild); specificity_vaild(isnan(specificity_vaild))=0; disp(['验证集specificity_vaild：',num2str(mean(specificity_vaild))])  % Specificity	
disp('测试集********************************************************************************') 	
confMat_test = confusionmat(test_y_feature_label,y_test_predict);	
TP_test = diag(confMat_test);      TP_test=TP_test'; % 被正确分类的正样本 True Positives	
FP_test = sum(confMat_test, 1)  - TP_test;  %被错误分类的正样本 False Positives	
FN_test = sum(confMat_test, 2)' - TP_test;  % 被错误分类的负样本 False Negatives	
TN_test = sum(confMat_test(:))  - (TP_test + FP_test + FN_test);  % 被正确分类的负样本 True Negatives	
	
accuracy_test = sum(TP_test) / sum(confMat_test(:)); accuracy_test(isnan(accuracy_test))=0; disp(['测试集accuracy：',num2str(accuracy_test)])% Accuracy	
precision_test = TP_test ./ (TP_test + FP_test);  precision_test(isnan(precision_test))=0; disp(['测试集precision_test：',num2str(mean(precision_test))]) % Precision	
recall_test = TP_test ./ (TP_test + FN_test); recall_test(isnan(recall_test))=0; disp(['测试集recall_test：',num2str(mean(recall_test))])  % Recall / Sensitivity	
F1_score_test = 2 * (precision_test .* recall_test) ./ (precision_test + recall_test); F1_score_test(isnan(F1_score_test))=0; disp(['测试集F1_score_test：',num2str(mean(F1_score_test))])   % F1 Score	
specificity_test = TN_test ./ (TN_test + FP_test); specificity_test(isnan(specificity_test))=0; disp(['测试集specificity_test：',num2str(mean(specificity_test))])  % Specificity	
	
disp('验证集+测试集 （没有用到优化可以直接当作整体的测试集）********************************************************************************') 	
test_y1=[vaild_y_feature_label;test_y_feature_label];y_test_predict1=[y_vaild_predict;y_test_predict];	
confMat_test1 = confusionmat(test_y1,y_test_predict1);	
TP_test1 = diag(confMat_test1);      TP_test1=TP_test1'; % 被正确分类的正样本 True Positives	
FP_test1 = sum(confMat_test1, 1)  - TP_test1;  %被错误分类的正样本 False Positives	
FN_test1 = sum(confMat_test1, 2)' - TP_test1;  % 被错误分类的负样本 False Negatives	
TN_test1 = sum(confMat_test1(:))  - (TP_test1 + FP_test1 + FN_test1);  % 被正确分类的负样本 True Negatives	
accuracy_test1 = sum(TP_test1) / sum(confMat_test1(:)); accuracy_test1(isnan(accuracy_test1))=0;  disp(['验证集+测试集accuracy：',num2str(accuracy_test1)])% Accuracy	
precision_test1 = TP_test1 ./ (TP_test1 + FP_test1);  precision_test1(isnan(precision_test1))=0;  disp(['验证集+测试集precision_test：',num2str(mean(precision_test1))]) % Precision	
recall_test1 = TP_test1 ./ (TP_test1 + FN_test1); recall_test1(isnan(recall_test1))=0;  disp(['验证集+测试集recall_test：',num2str(mean(recall_test1))])  % Recall / Sensitivity	
F1_score_test1 = 2 * (precision_test1 .* recall_test1) ./ (precision_test1 + recall_test1); F1_score_test1(isnan(F1_score_test1))=0; disp(['验证集+测试集F1_score_test：',num2str(mean(F1_score_test1))])   % F1 Score	
specificity_test1 = TN_test1 ./ (TN_test1 + FP_test1); specificity_test1(isnan(specificity_test1))=0; disp(['验证集+测试集specificity_test：',num2str(mean(specificity_test1))])  % Specificity	
	
%% K折验证	
x_feature_label_norm_all=(x_feature_label-x_mu)./x_sig;    %x特征	
y_feature_label_norm_all=y_feature_label;	
Kfold_num=G_out_data.Kfold_num;	
cv = cvpartition(size(x_feature_label_norm_all, 1), 'KFold', Kfold_num); % Split into K folds	
for k = 1:Kfold_num	
    trainingIdx = training(cv, k);	
    validationIdx = test(cv, k);	
     x_feature_label_norm_all_traink=x_feature_label_norm_all(trainingIdx,:);	
   y_feature_label_norm_all_traink=y_feature_label_norm_all(trainingIdx,:);	
	
   x_feature_label_norm_all_testk=x_feature_label_norm_all(validationIdx,:);	
   y_feature_label_norm_all_testk=y_feature_label_norm_all(validationIdx,:);	
	
   p_traink1=[];p_testk1=[];	
	
  	
    p_traink1=reshape(x_feature_label_norm_all_traink',size(x_feature_label_norm_all_traink,2),1,1,size(x_feature_label_norm_all_traink,1)); 	
   	
    p_testk1=reshape(x_feature_label_norm_all_testk',size(x_feature_label_norm_all_testk,2),1,1,size(x_feature_label_norm_all_testk,1)); 	
   	
  	
	
	
   optionsk = trainingOptions('adam', ... 	
        'Shuffle','every-epoch',...	
        'MaxEpochs',max_epoch, ..., 	
        'MiniBatchSize',min_batchsize,... 	
        'InitialLearnRate',0.001,... 	
        'ValidationFrequency',20);	
  	
   layers1 = [  imageInputLayer([  size(x_feature_label_norm_all_traink,2) 1 1])%%2D-CNN	
         convolution2dLayer([2,1],round(pop(2)))  	
         batchNormalizationLayer   	
         reluLayer	
         maxPooling2dLayer([2 1],'Stride',round(pop(5)))	
         convolution2dLayer([2,1],round(pop(3)))	
         batchNormalizationLayer 	
         reluLayer	
         maxPooling2dLayer([2 1],'Stride',round(pop(5)))	
         flattenLayer	
         bilstmLayer(round(pop(4)), 'OutputMode', 'last')      % LSTM层 	
         reluLayer	
         fullyConnectedLayer(length(unique(train_y_feature_label)))	
         softmaxLayer	
         classificationLayer];	
	
     Mdlkf1=TreeBagger(Model_RF.NumTrees ,x_feature_label_norm_all_traink,y_feature_label_norm_all_traink,'Method','classification','MinLeafSize',Model_RF.MinLeafSize);	
     Mdlkf2=trainNetwork(p_traink1,categorical(y_feature_label_norm_all_traink),layers1, optionsk);	
     Mdl_kfold{1,k}=Mdlkf1;Mdl_kfold{2,k}=Mdlkf2;	
   y_test_predict_norm_all_testk1=predict(Mdlkf1,x_feature_label_norm_all_testk);  %测试集预测结果	
   y_test_predict_all_testk1=RF_process(y_test_predict_norm_all_testk1);	
   y_test_predict_norm_all_testk2=double(classify(Mdlkf2, p_testk1));  %测试集预测结果	
   y_test_predict_all_testk2=y_test_predict_norm_all_testk2;	
	
   	
   y_test_predict_all_testk=round(y_test_predict_all_testk1*quan1+y_test_predict_all_testk2*quan2);	
	
	
   test_kfold=sum((y_test_predict_all_testk==y_feature_label_norm_all_testk))/length(y_feature_label_norm_all_testk);	
   AUC_kfold(k)=test_kfold;	
	
	
	
end	
	
	
% k折验证结果绘图	
figure('color',[1 1 1]);	
	
color_set=[0.4353    0.5137    0.7490];	
plot(1:length(AUC_kfold),AUC_kfold,'--p','color',color_set,'Linewidth',1.3,'MarkerSize',6,'MarkerFaceColor',color_set,'MarkerFaceColor',[0.3,0.4,0.5]);	
grid on;	
box off;	
grid off;	
ylim([0.92*min(AUC_kfold),1.2*max(AUC_kfold)])	
xlabel('kfoldnum')	
ylabel('accuracy')	
xticks(1:length(AUC_kfold))	
set(gca,'Xgrid','off');	
set(gca,'Linewidth',1);	
set(gca,'TickDir', 'out', 'TickLength', [.005 .005], 'XMinorTick', 'off', 'YMinorTick', 'off');	
yline(mean(AUC_kfold),'--')	
%小窗口柱状图的绘制	
axes('Position',[0.6,0.65,0.25,0.25],'box','on'); % 生成子图	
GO = bar(1:length(AUC_kfold),AUC_kfold,1,'EdgeColor','k');	
GO(1).FaceColor = color_set;	
xticks(1:length(AUC_kfold))	
xlabel('kfoldnum')	
ylabel('accuracy')	
disp('****************************************************************************************') 	
disp([num2str(Kfold_num),'折验证预测准确率accuracy结果：'])	
disp(AUC_kfold) 	
disp([num2str(Kfold_num),'折验证  ','accuracy均值为： ' ,num2str(mean(AUC_kfold)),'    accuracy标准差为： ' ,num2str(std(AUC_kfold))]) 	
	
	
	
	
	
	
	
	
	
	
%% LIME可解释分析	
num_set=500;    %这个值越大运行时间越长	
if size(test_x_feature_label_norm,1)>num_set	
    num_sample_get=num_set;	
    listshap_sample=round(1:size(test_x_feature_label_norm,1)/num_sample_get:size(test_x_feature_label_norm,1));	
else	
    listshap_sample=1:size(test_x_feature_label_norm,1);	
end	
	
disp('************************************');	
disp('正在进行LIME分析  请耐心等待');	
disp('************************************');	
	
index_name_plot=G_out_data.index_name_plot;	
color_get=G_out_data.color_get;	
for n=1:size(test_x_feature_label_norm,2)	
    train_x_shap=test_x_feature_label_norm(listshap_sample,n); train_x_shap1=train_x_feature_label_norm(:,n);                	
    c_index=train_x_shap-min(train_x_shap1); c_index1=ceil(c_index/max(c_index)*length(color_get))+1;	
    if isnan(c_index1)	
        c_index1=ones(length(c_index1),1);	
    end	
    color_shap1(:,n)=c_index1;	
end	
LimeValues=[];	
	
LimePredict_symble=G_out_data.LimePredict_symble;	
others=G_out_data.others;	
	
myPredict = @(x) LimePredict(Mdl,x,LimePredict_symble,others);	
	
explainer_lime = lime(myPredict,train_x_feature_label_norm,'Type','classification','DataLocality', 'local', 'KernelWidth', 0.5);	
	
for i=1:length(listshap_sample)	
     queryPoint=test_x_feature_label_norm(listshap_sample(i),:);	
     results = fit(explainer_lime,queryPoint,size(test_x_feature_label_norm,2));	
 	   LimeValues=[LimeValues;results.SimpleModel.Beta'];	
end	
	
LimeValues_imptance=mean(abs(LimeValues));	
[LimeValues_imptance1,LimeValues_sort]=sort(LimeValues_imptance, 'descend');	
	
X_get=1:size(LimeValues,2);	
X_get1=repmat(X_get,size(LimeValues,1),1);	
	
figure('Position',[200,200,800,400])	
	
 yline(0,'-','LineWidth',1.1,'Color',[0.6,0.6,0.6])	
 hold on	
	
for i=1:size(LimeValues,2)	
    s(i)=swarmchart(X_get1(:,(i)),LimeValues(:,LimeValues_sort(i)),15,color_shap1(:,LimeValues_sort(i)),'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);	
    hold on	
    s(i).XJitterWidth = 0.7;	
 end	
	
 colormap(color_get)	
cbtick= linspace(1,256,2);	
colorbar_index=colorbar('Ticks',cbtick,'TickLabels',{'Low','High'});	
colorbar_index.Label.String = 'Feature value';	
colorbar_index.Label.FontSize = 12;	
xticks([1:length(LimeValues_imptance1)])	
xticklabels(index_name_plot(LimeValues_sort))	
set(gca,'LineWidth',1.2)	
ylabel('Lime value (impact on model output) ')	
	
figure('Position',[200,200,600,350]) ;	
bar_plot_f=bar(LimeValues_imptance1);   %  重要性衡量	
bar_plot_f.FaceColor = 'flat';	
for i=1:length(LimeValues_imptance1)	
     bar_plot_f.CData(i,:)=[color_get(1+i*(floor(length(color_get)/length(LimeValues_imptance1))-1),:)];	
end	
	
xticks([1:length(LimeValues_imptance1)])	
xticklabels(index_name_plot(LimeValues_sort))	
title('Lime analysis')	
ylabel('Predictor importance estimates');	
xlabel('Predictors');	
	
	
	
	
index_name_plot1=G_out_data.index_name_plot1;	
color_get=G_out_data.color_get;	
LimePredict_symble=G_out_data.LimePredict_symble;	
others=G_out_data.others;	
	
PDPxlabel=G_out_data.PDPxlabel;  	
	
[PDP_one_dimension_plot,PDP_two_dimension_plot] = PDP_Predict(Mdl,test_x_feature_label_norm,test_y_feature_label_norm,x_mu,x_sig,y_mu,y_sig,LimePredict_symble,others,index_name_plot1,PDPxlabel,color_get);   	
%% 绘图块	
color_list=G_out_data.color_list;   %颜色数据库	
rand_list1=G_out_data.rand_list1;   %颜色数据库	
Line_Width=G_out_data.Line_Width;   %线粗细	
makesize=G_out_data.makesize;   %标记大小	
yang_str2=G_out_data.yang_str2;   %符号库	
yang_str3=G_out_data.yang_str3;   %符号库	
kuang_width=G_out_data.kuang_width;   %画图展示数据	
show_num=G_out_data.show_num;   %测试集画图展示数据	
show_num1=G_out_data.show_num1;   %验证集画图展示数据	
show_num2=G_out_data.show_num2;   %训练集画图展示数据	
FontName=G_out_data.FontName;  %绘制字体	
FontSize=G_out_data.FontSize;   %字体设置	
xlabel1=G_out_data.xlabel1;   %	
ylabel1=G_out_data.ylabel1;   %	
title1=G_out_data.title1;   %	
legend1=G_out_data.legend1;   %图例	
box1=G_out_data.box1;   %框	
le_kuang=G_out_data.le_kuang;   %图例框	
grid1=G_out_data.grid1;   %网格	
yang_fu3_ku=G_out_data.yang_fu3_ku;  %总体符号库	
color_index=G_out_data.color_index;	
yangsi_idnex=G_out_data.yangsi_idnex;  %总体样式库 	
figure	
uni_index=unique(train_y_feature_label);num_index=floor(show_num2/length(uni_index)); index_la_all=[];	
index_la_all=[];   %目的是从每个类分别找一些出来展示	
for j=1:length(uni_index)   	
    index_la=find(train_y_feature_label==uni_index(j));	
      if(length(index_la)<num_index)	
         index_la_all=[index_la_all;index_la];	
      else	
         index_la_all=[index_la_all;index_la(1:num_index)];	
      end	
end	
index_show=index_la_all;	
stairs(train_y_feature_label(index_show),yang_str2{1,3},'Color',color_list(rand_list1(1),:),'LineWidth',Line_Width(1));	
hold on	
stairs(y_train_predict(index_show),yang_str3{1,1},'Color',color_list(rand_list1(2),:),'LineWidth',Line_Width(2),'MarkerSize',makesize);	
hold on	
set(gca,'FontSize',FontSize,'LineWidth',kuang_width,'FontName',FontName)	
xlabel(gca,xlabel1)	
ylabel(gca,ylabel1)	
title(gca,'训练集结果')	
legend(gca,legend1) 	
box(gca,box1)	
legend(gca,le_kuang) %图例框消失	
grid(gca,grid1)	
	
figure	
cm = confusionchart(train_y_feature_label, y_train_predict);	
cm.Title = 'Confusion Matrix for train Data';	
cm.ColumnSummary = 'column-normalized';	
cm.RowSummary = 'row-normalized';	
	
	
figure	
uni_index=unique(vaild_y_feature_label);num_index=floor(show_num1/length(uni_index)); index_la_all=[];	
index_la_all=[];   %目的是从每个类分别找一些出来展示	
for j=1:length(uni_index)   	
    index_la=find(vaild_y_feature_label==uni_index(j));	
      if(length(index_la)<num_index)	
         index_la_all=[index_la_all;index_la];	
      else	
         index_la_all=[index_la_all;index_la(1:num_index)];	
      end	
end	
	
index_show=index_la_all;	
	
stairs(vaild_y_feature_label(index_show),yang_str2{1,3},'Color',color_list(rand_list1(1),:),'LineWidth',Line_Width(1));	
hold on	
stairs(y_vaild_predict(index_show),yang_str3{1,1},'Color',color_list(rand_list1(2),:),'LineWidth',Line_Width(2),'MarkerSize',makesize);	
hold on	
	
set(gca,'FontSize',FontSize,'LineWidth',kuang_width,'FontName',FontName)	
xlabel(gca,xlabel1)	
ylabel(gca,ylabel1)	
title(gca,'验证集结果')	
legend(gca,legend1) 	
box(gca,box1)	
legend(gca,le_kuang) %图例框消失	
grid(gca,grid1)	
	
figure	
cm = confusionchart(vaild_y_feature_label, y_vaild_predict);	
cm.Title = 'Confusion Matrix for vaild Data';	
cm.ColumnSummary = 'column-normalized';	
cm.RowSummary = 'row-normalized';	
	
	
figure	
uni_index=unique(test_y_feature_label);num_index=floor(show_num/length(uni_index)); index_la_all=[];	
index_la_all=[];   %目的是从每个类分别找一些出来展示	
for j=1:length(uni_index)   	
    index_la=find(test_y_feature_label==uni_index(j));	
      if(length(index_la)<num_index)	
         index_la_all=[index_la_all;index_la];	
      else	
         index_la_all=[index_la_all;index_la(1:num_index)];	
      end	
end	
	
index_show=index_la_all;	
stairs(test_y_feature_label(index_show),yang_str2{1,3},'Color',color_list(rand_list1(1),:),'LineWidth',Line_Width(1));	
hold on	
stairs(y_test_predict(index_show),yang_str3{1,1},'Color',color_list(rand_list1(2),:),'LineWidth',Line_Width(2),'MarkerSize',makesize);	
hold on	
	
set(gca,'FontSize',FontSize,'LineWidth',kuang_width,'FontName',FontName)	
xlabel(gca,xlabel1)	
ylabel(gca,ylabel1)	
title(gca,'测试集结果')	
legend(gca,legend1) 	
box(gca,box1)	
legend(gca,le_kuang) %图例框消失	
grid(gca,grid1)	
	
figure	
cm = confusionchart(test_y_feature_label, y_test_predict);	
cm.Title = 'Confusion Matrix for test Data';	
cm.ColumnSummary = 'column-normalized';	
cm.RowSummary = 'row-normalized';	
