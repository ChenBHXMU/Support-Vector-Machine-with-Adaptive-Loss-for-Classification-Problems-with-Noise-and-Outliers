function df = dfda(a,x,y,sv_label,type)
%对偶问题求导得梯度 The gradient of dual problem 1-sum(ajyiyjxi^Txj) 
[n,m] = size(a);
df = zeros(n,m);

for item = 1:length(a)
%for i = 1:length(sv_label)
    %item = sv_label(i);
    xi = x(:,item);
    dd = 0;
    for j = 1:length(a)
        aj = a(j);
        XX = kernel(xi,x(:,j),type);
        dd = dd + y(item)*y(j)*aj*XX;
    end
    df(item) = 1 - dd;
end

end

