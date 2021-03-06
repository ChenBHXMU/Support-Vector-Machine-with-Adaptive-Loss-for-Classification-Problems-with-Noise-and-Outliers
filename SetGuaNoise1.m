function [X] = SetGuaNoise1(X,P)
%制造高斯噪声
% X n*d
% P加百分比噪声 noise level
[nX,mX] = size(X);

N = 1:nX;
for i = 1:mX
    %每一列特征
    T = X(:,i);
    minT = min(T);
    maxT = max(T);
    %加入噪声
    t = normrnd(0,1,length(N),1); % 产生N*1，0-1的正态分布随机数 N(0,1)
    tt = minT + (maxT-minT)*t; %跟样本量纲一致
    X(N,i) = X(N,i) + P*tt;
end
end

