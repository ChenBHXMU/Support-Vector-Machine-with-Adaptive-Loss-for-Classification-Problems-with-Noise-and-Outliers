function  main()
Data =  cell2mat(struct2cell(load('air.mat')));
[m,n] = size(Data);
X = Data(:,1:n-1);
Y = Data(:,n);
X = SetGuaNoise1(X,0.5);%比例 噪声大小
X = mapminmax(X',0,1)';
data = [X,Y];
indices = crossvalind('Kfold', m, 5);%5折交叉 Cross validation

for i = 1 : 5
    % 获取第i份测试数据的索引逻辑值
    test = (indices == i);
    % 取反，获取第i份训练数据的索引逻辑值
    train = ~test;
    %1份测试，5份训练
    test_data = data(test, 1 : n - 1);
    test_label = data(test, n);
    train_data = data(train, 1 : n - 1);
    train_label = data(train, n);
    %训练数据加标签噪声
    [train_label] = setLabelNoise(train_label,0.2);
    %参数
    paraList = [0.01,0.1,1,10,100];%C
    for iPara = 1:length(paraList)
        C = paraList(iPara);
        kertype = 'linear'; % linear 线性  rbf 高斯
        item = 20;
        [result,accuracy,Svsnum,TrainTime,TestTime,QBest,pBest,margin] = testmySVM(train_data,train_label,test_data,test_label,kertype,C,item);
        accuracy
    end
end

end

