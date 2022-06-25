%function [softDis,d1] = SoftDistance1(X,Y,W,b,n,t,C,q,p,c)
function [d1,gg] = SoftDistanceNewS(M,t,C,q,p,n)
% X \in R^{d*n}, Y \in R^{n*c}
%CS||X||2q + (1-S)||X||2p  q>1,0<p<=1
%   X'*W+ones(n,1)*b'-Y
%L = Y.*M;
d1 = ones(n,1);
gg = zeros(n,1);
[nL,mL] = find(M~=0);
mi = length(mL);
for i = 1:mi
    indj = mL(i);
    dis1 = M(indj);
    [d1(indj,1),gg(indj,1)] = getv(dis1,q,p,t);
end

end

