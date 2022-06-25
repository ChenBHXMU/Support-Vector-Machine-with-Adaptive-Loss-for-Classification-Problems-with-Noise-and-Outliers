function [obj] = getObjectiveFun(w,xi,q,p)
%Obtain objective function

xi2 = xi.^2;
s = exp(-xi2);
loss = s.*xi.^q+(1-s).*xi.^p;

obj = 0.5*norm(w,2)^2 + sum(loss);
end

