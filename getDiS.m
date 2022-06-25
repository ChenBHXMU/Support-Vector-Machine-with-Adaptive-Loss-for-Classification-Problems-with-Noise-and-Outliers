function [di,D,gg] = getDiS(svm, Xt, Yt, kertype,r,q,p)
%获得di
%   此处显示详细说明
n=length(Yt);
%temp = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,svm.Xsv,kertype);  
%b = mean(svm.Ysv-temp);  %b取均值  
w = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,Xt,kertype); 

result.score = w + svm.b;  
D = max(0,1-Yt.*result.score);
[di,gg] = SoftDistanceNewS(D,1,r,q,p,n);


end

