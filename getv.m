function [v,gg] = getv(dis1,q,p,t)
%UNTITLED3 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
dis2 = dis1.^2;
s = exp(-dis2/t);
v = s*(-2*dis1^(q+1))+q*s*dis1^(q-1)+p*dis1^(p-1)+2*dis1^(p+1)*s-p*dis1^(p-1)*s;
gg = s*dis1^(q)+(1-s)*dis1^p;
end

