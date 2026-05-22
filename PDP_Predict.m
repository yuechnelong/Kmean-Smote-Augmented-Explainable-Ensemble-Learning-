function [PDP_one_dimension_plot,PDP_two_dimension_plot] = PDP_Predict(Mdl,X,y,x_mu,x_sig,y_mu,y_sig,symbol,others,data_biao,PDPxlabel,colorget)
% data_biao 特征的标识
% PDPxlabel PDP 可解释的标签
%预测模型PDP可解释检验

if length(PDPxlabel)>size(X,2) || max(PDPxlabel)>size(X,2)
    PDPxlabel=1:size(X,2) ;
end

N_max=5; %最大设置展示的特征列
if length(PDPxlabel)>5
   disp('PDP分析中******默认只对前5个特征分析**修改可以在 PDP_Predict.m 函数中修改')
   PDPxlabel=PDPxlabel(1:N_max);
end

Dim_set=length(PDPxlabel);

PDP_onedimension=[];
PDP_two_dimension=[]; %二维度
X_origin=X;
numPoints = 50;  %绘制点数
for i=1:(Dim_set)
    X1=X_origin(:,PDPxlabel(i));
    X1_range = linspace(min(X1), max(X1), numPoints);
    X=X_origin;
    for j=1:numPoints

        X(:,PDPxlabel(i))=X1_range(j);
        yPred=LimePredict(Mdl,X,symbol,others);
         PDP_onedimension(j,i) = mean(yPred);
    end

   
    PDP_one_X(:,i)=X1_range;
    biao_str_len=length(data_biao);
    figure
    scatter(X1.*x_sig(PDPxlabel(i))+x_mu(PDPxlabel(i)),y.*y_sig+y_mu,20)
    hold on
    plot(X1_range.*x_sig(PDPxlabel(i))+x_mu(PDPxlabel(i)),PDP_onedimension(:,i).*y_sig+y_mu,'LineWidth',2,'Color',[0.85 0.33 0.1]);

    % plot(X1_range,PDP_onedimension(:,i),'LineWidth',2,'Color',[0.85 0.33 0.1]);
    xstr=[data_biao{1,PDPxlabel(i)}];
    xlabel(xstr);
    ystr=['meanpredict-',data_biao{1,end}];
    ylabel(ystr);
    title('单特征部分依赖图 (PDP)');
    grid on;
    % disp(PDP_onedimension)

end
PDP_one_dimension_plot{1,1}=PDP_one_X.*x_sig(PDPxlabel(i))+x_mu(PDPxlabel(i));
PDP_one_dimension_plot{1,2}=PDP_onedimension.*y_sig+y_mu;


%二维交互PDP
numPoints=30;
if Dim_set>=2
     X_temp = X_origin;
    for i=1:numPoints
        for j=1:numPoints
            X1=X_origin(:,PDPxlabel(1));
            X2=X_origin(:,PDPxlabel(2));
            [X1_grid,X2_grid] = meshgrid(linspace(min(X1),max(X1),numPoints), ...
                linspace(min(X2),max(X2),numPoints));

            X_temp(:,1) = X1_grid(i,j);
            X_temp(:,2) = X2_grid(i,j);
            X=X_temp;
            yPred=LimePredict(Mdl,X,symbol,others);
            PDP_two_dimension(i,j) = mean(yPred);
        end
    end

    PDP_two_dimension_plot{1,1}=X1_grid.*x_sig(PDPxlabel(1))+x_mu(PDPxlabel(1));
    PDP_two_dimension_plot{1,2}=X2_grid.*x_sig(PDPxlabel(2))+x_mu(PDPxlabel(2));
    PDP_two_dimension_plot{1,3}=PDP_two_dimension.*y_sig+y_mu;
    % disp(PDP_two_dimension)



figure;
surf(X1_grid.*x_sig(PDPxlabel(1))+x_mu(PDPxlabel(1)),X2_grid.*x_sig(PDPxlabel(2))+x_mu(PDPxlabel(2)),PDP_two_dimension.*y_sig+y_mu,'EdgeColor','none');
xlabel([data_biao{1,PDPxlabel(1)}]);
ylabel([data_biao{1,PDPxlabel(2)}]);
zlabel(['meanpredict ',data_biao{1,end}]);
title('双特征部分依赖图 (PDP)');
colorbar;
colormap(colorget);
view(135,30);
grid on;


figure;
% surf(X1_grid,X2_grid,PDP_two_dimension,'EdgeColor','none');

sc=surfc(X1_grid.*x_sig(PDPxlabel(1))+x_mu(PDPxlabel(1)),X2_grid.*x_sig(PDPxlabel(2))+x_mu(PDPxlabel(2)),PDP_two_dimension.*y_sig+y_mu,'FaceAlpha',0.9,'EdgeColor','none');
colorbar
% zlim([min 11])
sc(2).FaceColor='auto';
sc(2).EdgeColor='none';
% sc(2).ZLocation='zmax';
sc(2).FaceAlpha=0.9;
% xlabel('x')
% ylabel('y')
% zlabel('z')

xlabel([data_biao{1,PDPxlabel(1)}]);
ylabel([data_biao{1,PDPxlabel(2)}]);
zlabel(['meanpredict-',data_biao{1,end}]);
title('双特征部分依赖图 (PDP)');
colorbar;
colormap(colorget);
view(135,30);
grid on;
% disp(data_biao)
end

end