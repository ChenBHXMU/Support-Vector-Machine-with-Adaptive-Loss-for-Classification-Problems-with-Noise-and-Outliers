function  main()
Data =  cell2mat(struct2cell(load('air.mat')));
[m,n] = size(Data);
X = Data(:,1:n-1);
Y = Data(:,n);
X = SetGuaNoise1(X,0.5);%���� ������С
X = mapminmax(X',0,1)';
data = [X,Y];
indices = crossvalind('Kfold', m, 5);%5�۽��� Cross validation

for i = 1 : 5
    % ��ȡ��i�ݲ������ݵ������߼�ֵ
    test = (indices == i);
    % ȡ������ȡ��i��ѵ�����ݵ������߼�ֵ
    train = ~test;
    %1�ݲ��ԣ�5��ѵ��
    test_data = data(test, 1 : n - 1);
    test_label = data(test, n);
    train_data = data(train, 1 : n - 1);
    train_label = data(train, n);
    %ѵ�����ݼӱ�ǩ����
    [train_label] = setLabelNoise(train_label,0.2);
    %����
    paraList = [0.01,0.1,1,10,100];%C
    for iPara = 1:length(paraList)
        C = paraList(iPara);
        kertype = 'linear'; % linear ����  rbf ��˹
        item = 20;
        [result,accuracy,Svsnum,TrainTime,TestTime,QBest,pBest,margin] = testmySVM(train_data,train_label,test_data,test_label,kertype,C,item);
        accuracy
    end
end

end

