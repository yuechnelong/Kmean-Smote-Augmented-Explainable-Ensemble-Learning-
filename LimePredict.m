function yPred = LimePredict(Mdl,X,symbol,others)
if symbol==1
    yPred= predict(Mdl,X);
elseif symbol==2
    % LSTM之类的
    p_X=[];
    for i = 1 : size(X,1)
        p_X{i, 1}  = (X(i,:))';
    end
    yPred = predict(Mdl, p_X);
elseif symbol==3
    % CNN之类的
    p_X=reshape(X',size(X,2),1,1,size(X,1));

    yPred = predict(Mdl, p_X);
elseif symbol==4
    % XGBoost 回归
    yPred = predict_xgb(Mdl, X);
elseif symbol==5
    % LightGBM回归
    % others=best_iter;
    yPred=predict_LGB1('LightGBM_model.txt',X,others);
elseif symbol==6
    %MLP-RF机器学习组合回归
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=predict(Model1,X);
    y_test_predict_norm2=predict(Model2,X);
    yPred=quan1*y_test_predict_norm1+quan2*y_test_predict_norm2;
elseif symbol==7
    % MLP-XGB
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=predict_xgb(Model1,X);
    y_test_predict_norm2=predict(Model2,X);
    yPred=quan1*y_test_predict_norm1+quan2*y_test_predict_norm2;

elseif symbol==8
    %优化LSTM-XGB
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=predict_xgb(Model1,X);
    for i = 1 : size(X,1)
        p_test1{i, 1}  = (X(i,:))';
    end
    y_test_predict_norm2=predict(Model2,p_test1);
    yPred=quan1*y_test_predict_norm1+quan2*y_test_predict_norm2;
elseif symbol==9
    %优化CNNGRU-XGB
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=predict_xgb(Model1,X);
    p_test1=reshape(X',size(X,2),1,1,size(X,1));
    y_test_predict_norm2=predict(Model2,p_test1);
    yPred=quan1*y_test_predict_norm1+quan2*y_test_predict_norm2;
elseif symbol==10
    %优化BiLSTM-RF回归/分类
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=predict(Model1,X);
    for i = 1 : size(X,1)
        p_test1{i, 1}  = (X(i,:))';
    end
    y_test_predict_norm2=predict(Model2,p_test1);
    yPred=quan1*y_test_predict_norm1+quan2*y_test_predict_norm2;
elseif symbol==11
    %优化CNNBiLSTM-RF回归/分类
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=predict(Model1,X);
    p_test1=reshape(X',size(X,2),1,1,size(X,1));

    y_test_predict_norm2=predict(Model2,p_test1);
    yPred=quan1*y_test_predict_norm1+quan2*y_test_predict_norm2;

elseif symbol==12
    y_test_predict_norm = sim(Mdl,X'); yPred=y_test_predict_norm';

elseif symbol==13
    Mdl1=Mdl(1,1:end);
    yPred=DELMPredict(X,Mdl1,Mdl{3,1});

elseif symbol==14
    cof1=Mdl{1,1};
    cof0=Mdl{1,2};
    yPred=cof0+X*cof1;  %测试集预测结果

elseif symbol==15
    for i = 1 : size(X,1)
        p_test1{i, 1}  = (X(i,:))';
    end
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; Model3=Mdl{1,3};quan_all=Mdl{1,4};
    P_test_y_RF=predict(Model1,X);
    P_test_y_MLP=predict(Model2,X);
    P_test_y_LSTM=predict(Model3,p_test1);
    yPred=quan_all(1)*P_test_y_RF+quan_all(2)*P_test_y_MLP+quan_all(3)*P_test_y_LSTM;

elseif symbol==16
    yPred=predict(Mdl,X);

elseif symbol==17
    %随机森林分类
    yPred=RF_process(predict(Mdl,X));
elseif symbol==18
    %SVM分类
    [~,score_test] = predict(Mdl,X);
    [~,y_test_predict] = max(score_test');
    yPred=y_test_predict';
elseif symbol==19

    score_test = Mdl(X');
    [~,y_test_predict] = max(score_test);
    yPred=y_test_predict';
elseif symbol==20
    y_test_predict_norm=predict_xgb(Mdl,X);
    yPred=round(y_test_predict_norm);
elseif symbol==21
    y_test_predict_norm=predict_LGB1('LightGBM_model_class.txt',X,others);
    [~, yPred] = max(y_test_predict_norm, [], 2);

elseif symbol==22
    for i = 1 : size(X,1)
        p_test1{i, 1}  = (X(i,:))';
    end
    yPred=double(classify(Mdl, p_test1));
elseif symbol==23
    p_test1=reshape(X',size(X,2),1,1,size(X,1));
    yPred=double(classify(Mdl, p_test1));

elseif symbol==24
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=RF_process(predict(Model1,X));
    y_test_predict_norm2=predict(Model2,X);
    yPred=round(quan1*y_test_predict_norm1+quan2*y_test_predict_norm2);

elseif symbol==25
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=round(predict_xgb(Model1,X));
    y_test_predict_norm2=predict(Model2,X);
    yPred=round(quan1*y_test_predict_norm1+quan2*y_test_predict_norm2);
elseif symbol==26
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=round(predict_xgb(Model1,X));
    for i = 1 : size(X,1)
        p_test1{i, 1}  = (X(i,:))';
    end
    y_test_predict_norm2=double(classify(Model2, p_test1));
    yPred=round(quan1*y_test_predict_norm1+quan2*y_test_predict_norm2);
elseif symbol==27
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=round(predict_xgb(Model1,X));
    p_test1=reshape(X',size(X,2),1,1,size(X,1));
    y_test_predict_norm2=double(classify(Model2, p_test1));
    yPred=round(quan1*y_test_predict_norm1+quan2*y_test_predict_norm2);
elseif symbol==28
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=RF_process(predict(Model1,X));
    for i = 1 : size(X,1)
        p_test1{i, 1}  = (X(i,:))';
    end
    y_test_predict_norm2=double(classify(Model2, p_test1));
    yPred=round(quan1*y_test_predict_norm1+quan2*y_test_predict_norm2);
elseif symbol==29
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=RF_process(predict(Model1,X));
    p_test1=reshape(X',size(X,2),1,1,size(X,1));
    y_test_predict_norm2=double(classify(Model2, p_test1));
    yPred=round(quan1*y_test_predict_norm1+quan2*y_test_predict_norm2);
elseif symbol==30
    for i = 1 : size(X,1)
        p_test1{i, 1}  = (X(i,:))';
    end
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; Model3=Mdl{1,3};quan_all=Mdl{1,4};
    P_test_y_RF=RF_process(predict(Model1,X));
    P_test_y_MLP=predict(Model2,X);
    P_test_y_LSTM=double(classify(Model3, p_test1));
    yPred=round(quan_all(1)*P_test_y_RF+quan_all(2)*P_test_y_MLP+quan_all(3)*P_test_y_LSTM);
elseif symbol==31
    Model1= Mdl{1,1}; Model2=Mdl{1,2}; quan_all=Mdl{1,3}; quan1=quan_all(1);quan2=quan_all(2);
    y_test_predict_norm1=(predict(Model1,X));
    y_test_predict_norm2=predict(Model2,X);
    yPred=round(quan1*y_test_predict_norm1+quan2*y_test_predict_norm2);

elseif symbol==32
    for i = 1 : size(X,1)
        p_test1{i, 1}  = (X(i,:))';
    end
    cof1=Mdl{2,1};  %=b(2:end)
    cof0=Mdl{2,2};%=b(1);
    % b=[cof0;cof1];
    masklist=Mdl{2,3};% =masklist;
    index_sort_cor1=Mdl{2,4};%=index_sort_cor(1:max_fun);

    Model1= Mdl{1,1};  Model3=Mdl{3,1};quan_all=Mdl{4,1};
    P_test_y_RF=predict(Model1,X);

    % P_test_y_MLP=predict(Model2,X);
    n = size(X,2);  % 想要的数量
    data_biao2 = arrayfun(@(k) sprintf('F%d', k), 1:n, 'UniformOutput', false);
    % data_biao2=[];
    [test_x_feature_label_norm_SR_pre] = generate_SR_Pre(X, data_biao2,masklist);
    test_x_feature_label_norm_SR_pre1=test_x_feature_label_norm_SR_pre(:,index_sort_cor1);
    P_test_y_SR=cof0+test_x_feature_label_norm_SR_pre1*cof1;  %测试集预测结果
    P_test_y_LSTM=predict(Model3,p_test1);
    yPred=quan_all(1)*P_test_y_RF+quan_all(2)*P_test_y_SR+quan_all(3)*P_test_y_LSTM;

end

end
