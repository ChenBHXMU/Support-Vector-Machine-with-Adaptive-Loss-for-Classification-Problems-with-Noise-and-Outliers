function [X] = SetGuaNoise1(X,P)
%�����˹����
% X n*d
% P�Ӱٷֱ����� noise level
[nX,mX] = size(X);

N = 1:nX;
for i = 1:mX
    %ÿһ������
    T = X(:,i);
    minT = min(T);
    maxT = max(T);
    %��������
    t = normrnd(0,1,length(N),1); % ����N*1��0-1����̬�ֲ������ N(0,1)
    tt = minT + (maxT-minT)*t; %����������һ��
    X(N,i) = X(N,i) + P*tt;
end
end

