function [Acc,SVs,preY,trainTime,testTime,svm,maxLabel,objFuc,AccTrain,AccTest] = adaptivesvmTrain_multiclss( trainData,trainLabel,testData,testLabel,kertype,C,item,q,p)
%trainData dim*n  trainLabel 1*n
%分类y=max(f(x))
%使标签为[-1,1]
SVs = 0;%支持向量个数
class = unique(trainLabel);
nuclass = length(class);
trainTime = 0;
testTime = 0;
r = 1;
objFuc = zeros(item,1);
objFuc1 = zeros(item,1);
AccTrain = zeros(item,1);
AccTest = zeros(item,1);
[mTrain,nTrain] = size(trainData);
[mTest,nTest] = size(testData);
testYList = zeros(nuclass,nTest);
proList = zeros(nuclass,nTest);
stepsize = 1;
aa = [];
epsilon=1e-5;

if(nuclass == 2)
    preY = zeros(1,nTest);
    tl = mapminmax([trainLabel,testLabel],-1,1);
    trainLabel = tl(1:nTrain);
    testLabel = tl(nTrain+1:(nTrain+nTest));
    index1 = find(trainLabel == 1); index2 = find(trainLabel == -1);
    trainData = [trainData(:,index1),trainData(:,index2)];trainLabel = [trainLabel(:,index1),trainLabel(:,index2)];
    options=optimset;
    options.LargerScale='off';
    options.Display='off';
    n=length(trainLabel);
    H=(trainLabel'*trainLabel).*kernel(trainData,trainData,kertype);
    f=-ones(n,1);
    A=[];
    b=[];
    Aeq=trainLabel;
    beq=0;
    lb=zeros(n,1);
    ub=C*ones(n,1);
    a0=zeros(n,1);
    tic;
    [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
    [svm,~] = calculate_rho(a,trainData',trainLabel',C,kertype);
    
    i = 0;
    while(i<=item)
        i = i + 1;
        %Cnew=di*C
        [di,ei,gg] = getDiS(svm, trainData, trainLabel, kertype,r,q,p);
        ub=di.*(C*ones(n,1));
        %ub=(C*ones(n,1))./di;
        a0=zeros(n,1);
        [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
        [svm,sv_label] = calculate_rho(a,trainData',trainLabel',C,kertype);

        objFuc(i) = getObjectiveFun(svm.w,ei,q,p);
        R1 = svmTest(svm, trainData, kertype);
        AccTrain(i) = size(find(trainLabel==R1.Y))/size(trainLabel);
        %R2 = svmTest(svm, testData, kertype);
        %maxLabel =testLabel.*R2.score;
        
        AccTest(i) = size(find(testLabel==preY))/size(testLabel);
        
        if(i>1 && abs(objFuc(i)-objFuc(i-1))<=epsilon)
            %i=i
            objFuc((i+1):item) = objFuc(i);
            AccTrain((i+1):item) = AccTrain(i);
            AccTest((i+1):item) = AccTest(i);
            break;
        elseif(i>1 && objFuc(i)>objFuc(i-1)) %Non convergence
            a = aa;
            %Armijo rule
            df = dfda(a,trainData,trainLabel,sv_label,kertype);
            a = a + stepsize*df;%对偶问题求max Finding the maximum of dual problem
            objFuc(i) = 0;
            i = i - 1;
            %get svm again to update di
            [svm,sv_label] = calculate_rho(a,trainData',trainLabel',C,kertype);
            stepsize = stepsize*0.1;
            if(stepsize <= 1e-6) %避免陷入震荡 Set the minimum step size to avoid non convergence
                objFuc((i+1):item) = objFuc(i);
                svm = svmsvm;
                AccTrain((i+1):item) = AccTrain(i);
                AccTest((i+1):item) = AccTest(i);
                break;
            end
        else
            %记录使得目标函数值下降的alpha
            aa = a;
            svmsvm = svm;
            stepsize = 1;
        end
    end
    SVs = SVs + svm.svnum;
    trainTime = toc;
    tic;
    result=svmTest_multiclass(svm,testData,kertype);
    testTime = toc;
    maxLabel =testLabel.*result.score;
    preY = sign(result.score);
    
    %得到精度 obtain Acc
    Acc = size(find(preY==testLabel))/size(testLabel);
    result = svmTest(svm, testData, kertype);
    
    %真实标签 Get label
    indPreYmin = find(result.score<0);
    preY(1,indPreYmin) = min(class);
    indPreYmax = find(result.score>0);
    preY(1,indPreYmax) = max(class);
end
if(nuclass > 2) % multi-class classification
    nn = 0;
    [mTrain,nTrain] = size(trainData);
    svsList = zeros(nTrain,1); %用来记录支持向量的个数，非0即支持向量 the number of SVs
    for ii = 0:nuclass-1
        nn = nn + 1;
        iclass = class(ii+1);
        iindex = find(trainLabel==iclass);
        jindex = find(trainLabel~=iclass);
        itrainLabel = trainLabel(iindex) - trainLabel(iindex) + 1;%one为1
        jtrainLabel = trainLabel(jindex) - trainLabel(jindex) - 1;%All为-1
        itrainData = trainData(:,iindex);
        jtrainData = trainData(:,jindex);
        ijtrainLabel = [itrainLabel,jtrainLabel];
        ijtrainData = [itrainData,jtrainData];
        options=optimset;
        options.LargerScale='off';
        options.Display='off';
        
        n=length(ijtrainLabel);
        H=(ijtrainLabel'*ijtrainLabel).*kernel(ijtrainData,ijtrainData,kertype);
        f=-ones(n,1);
        A=[];
        b=[];
        Aeq=ijtrainLabel;
        beq=0;
        lb=zeros(n,1);
        ub=C*ones(n,1);
        a0=zeros(n,1);
        tic;
        [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
        %求b
        [svm,~] = calculate_rho(a,ijtrainData',ijtrainLabel',C,kertype);
        i = 0;
        while(i<=item)
            i = i + 1;
            [di,ei] = getDiS(svm, ijtrainData, ijtrainLabel, kertype,r,q,p);
            ub=di.*(C*ones(n,1));
            a0=zeros(n,1);
            [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
            [svm,sv_label] = calculate_rho(a,ijtrainData',ijtrainLabel',C,kertype);
            %converage
            objFuc(i) = getObjectiveFun(svm.w,ei,q,p);
            
            if(i>1 && abs(objFuc(i)-objFuc(i-1))<=epsilon)
                objFuc((i+1):item) = objFuc(i);
                break;
            elseif(i>1 && objFuc(i)>objFuc(i-1))
                a = aa;
                %Armijo rule
                df = dfda(a,ijtrainData,ijtrainLabel,sv_label,kertype);
                a = a + stepsize*df;
                objFuc(i) = 0;
                i = i - 1;
                %update svm to get di
                [svm,sv_label] = calculate_rho(a,ijtrainData',ijtrainLabel',C,kertype);
                stepsize = stepsize*0.1;
                if(stepsize <= 1e-6)
                    objFuc((i+1):item) = objFuc(i);
                    svm = svmsvm;
                    sv_label = sv_labels;
                    break;
                end
            else
                aa = a;
                stepsize = 1;
                svmsvm = svm;
                sv_labels = sv_label;
            end
        end
        svsList(sv_label(:,:))=1;%单次支持向量
        trainTime = trainTime + toc;
        tic;
        result=svmTest_multiclass(svm,testData,kertype);
        testTime = testTime + toc;
        testYList(nn,:) = result.score;
    end
    [maxLabel,maxIndex] = max(testYList);
    testY = class(maxIndex);
    %得到精度
    preY = testY;
    Acc = size(find(testLabel==preY))/size(testLabel);
    svm.svnum = length(find(svsList==1));
    SVs = SVs + svm.svnum;
end

end

