%function [accuracy,RRList,QQList,ppList] = testmySVM(trainData,trainLabel,testData,testLabel,kertype,C)
function [result,accuracy,Svsnum,TrainTime,TestTime,QBest,pBest,margin,F1] = testmySVM(trainData,trainLabel,testData,testLabel,kertype,C,item)
%���ǩԤ�� trainData=������*ά�� ��trainLabel = ������*ά�� Svsnum���������

%%%%%��������%%%%%%
YY = unique(trainLabel);
[MY,NY] = size(YY);
numY = MY;

% %%%%%%ѵ������С%%%
% [Mtrain,~] = size(trainData);
% trainYList = zeros(Mtrain,numY);
% %%%%���Լ���С%%%
[Mtest,~] = size(testData);
testYList = zeros(Mtest,numY);



pList = [0.3,0.5,0.7,0.9];
QList = [1.2,1.4,1.6,1.8];
RList = [1];
result = zeros(length(pList),length(QList),length(RList));
trainTime =zeros(length(pList),length(QList),length(RList));
testTime = zeros(length(pList),length(QList),length(RList));
svsNum = zeros(length(pList),length(QList),length(RList));
% ProList = zeros(numY,Mtest,length(pList),length(QList),length(RList));
marginList = zeros(length(pList),length(QList),Mtest);
for R_i = 1:length(RList)
    R = RList(R_i);
    for p_i = 1:length(pList)
        p1 = pList(p_i);
        for Q_i = 1:length(QList)
            [Acc,SVs,preY,TrainTime,TestTime,~,margin,~,~,~] = adaptivesvmTrain_multiclss( trainData',trainLabel',testData',testLabel',kertype,C,item,QList(Q_i),p1);
            trainTime(p_i,Q_i,R_i) = TrainTime;
            testTime(p_i,Q_i,R_i) = TestTime;
            result(p_i,Q_i,R_i) = Acc;
            svsNum(p_i,Q_i,R_i) = SVs;
            marginList(p_i,Q_i,:) = margin;
        end
    end
end
[accuracy] = max(max(max(result)));
s=size(result);%������άά����Ĵ�С
Lax=find(result>=accuracy);%�������ֵλ�õĵ��±�
[i,k,j]=ind2sub(s,Lax);%����Сֵ���±�תΪ��ά���±�
%ʱ��
TrainTime = trainTime(i(1),k(1),j(1));
TestTime = testTime(i(1),k(1),j(1));
Svsnum =svsNum(i(1),k(1),j(1));
QBest = QList(k(1));
pBest = pList(i(1));
margin = reshape(marginList(i(1),k(1),:),1,Mtest);
end

