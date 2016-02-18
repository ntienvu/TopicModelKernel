function [ Output ] = TopicModelKernel( labelTrain, featureTrain, labelTest, featureTest, varargin )
% input ===================================================================
% labelTrain: [size of Training Data x 1]
% featureTrain: [size of Training Data x size of Feature]
% labelTest: [size of Testing of Data x 1]
% featureTest: [size of Testing of Data x size of Feature]
% 'IsDefaultParameter'= 1 (default value) : using default parameter C for SVM and sigma for TMK)
% 'IsDefaultParameter'= 0 : cross-validation on training set for searching the best parameter C and sigma)
% 'IsPlot'= 1 : plot the accuracy w.r.t the parameter space of the cross-validation
% 'IsPlot'= 0 (default value): donot plot anything.
% output ==================================================================
% Output.Accuracy = classification accuracy 
% Output.PredictedLabel = predicted label : [size of Testing of Data x 1]
% Output.NumberOfClass = number of class in the data for classification
% Output.TotalNumberOfTrainData = total number of training data used
% Output.TotalNumberOfTestData = total number of testing data used
% Output.FeatureSize = the dimension of feature vector for each instance
% Output.DefaultSigma = the default value of sigma is used
% Output.DefaultC = the default value of C is used
% =========================================================================
% contact: ntienvu@gmail.com
% cite: T.V. Nguyen, D. Phung, S. Venkatesh, Topic Model Kernel: An Empirical Study Towards Probabilistically
% Reduced Features For Classification, ICONIP 2013

warning off

disp('Topic Model Kernel with LibSVM for Classification...');

tic



%% scale data based on LibSVM
epsilon=0.0000001;
% minimums = min(featureTrain, [], 1);
% ranges = max(featureTrain, [], 1) - minimums;
% idx=find(ranges==0);
% ranges(idx)=epsilon;
% featureTrain = (featureTrain - repmat(minimums, size(featureTrain, 1), 1)) ./ repmat(ranges, size(featureTrain, 1), 1);
% featureTest = (featureTest - repmat(minimums, size(featureTest, 1), 1)) ./ repmat(ranges, size(featureTest, 1), 1);


% check the integrity of result
if length(unique(labelTrain)) ~= length(unique(labelTest))
    error('The label in Training set is not consistent with the Testing set');
end;

%% scale
dimensionTrain=size(featureTrain,2);
nTrain=size(featureTrain,1);
for i=1:nTrain
    [val, idx]=find(featureTrain(i,:)>0);
    m=min(featureTrain(i,idx));
    %featureTrain(i,:)=(featureTrain(i,:)+smooth)/(sum(featureTrain(i,:))+smooth);
    idxzero=find(featureTrain(i,:)==0);
    %featureTrain(i,idxzero)=m/(dimensionTrain*dimensionTrain);
    featureTrain(i,idxzero)=epsilon;
end

dimensionTest=size(featureTest,2);

nTest=size(featureTest,1);

if dimensionTest<dimensionTrain
    disp('dimensionTest<dimensionTrain');
    for dd=dimensionTest+1:dimensionTrain
        featureTest(:,dd)=zeros(nTest,1);
    end
end

fprintf('Feature Size = %d\n',dimensionTrain);
fprintf('Number of Classes = %d\n',length(unique(labelTest)));
fprintf('Total number of Training Instances = %d\n',nTrain);
fprintf('Total number of Testing Instances = %d\n',nTest);

for i=1:nTest
    [val, idx]=find(featureTest(i,:)>0);
    m=min(featureTest(i,idx));
    %featureTest(i,:)=(featureTest(i,:)+smooth)/(sum(featureTest(i,:))+smooth);
    idxzero=find(featureTest(i,:)==0);
    %featureTest(i,idxzero)=m/(dimensionTest*dimensionTest);
    featureTest(i,idxzero)=epsilon;
end

%% =========================

[IsDefault, IsPlot] = process_options(varargin,'IsDefaultParameter',1,'IsPlot',0);

%compute Training Similarity Matrix
NTrain=size(labelTrain,1);

%compute Testing Similarity Matrix
NTest=size(labelTest,1);
dimension=size(featureTrain,2);

bestc=1;%at default
bestsigma=sqrt(dimension);
AccCVJS=zeros(10,24);
bestcv = 0;

if IsDefault==0 %sigma is different from default value (sigma=1)
    disp('Searching optimal parameter...');
    flag=0;
    for log2c = -12:2:10
        
        cv=zeros(1,10);
        for temp=0:2:9 %0:0.5:5
            
            logsigma=temp/2;
            
            str=sprintf('log2c=%2.1f \t logsigma=%2.1f',log2c,logsigma);
            disp(str);
            
            cv(temp+1)=SearchingBestSigma(labelTrain,featureTrain,2^logsigma,log2c);
            
            fprintf(repmat('\b',1,length(str)+1));
        end
        [val idx]=max( cv );
        if (val >= bestcv),
            bestcv = val; bestc = 2^log2c;
            bestsigma = 2^( (idx-1) /2);
        end
        AccCVJS(:,log2c+13)=cv;
    end
    
    AccCVJS(:,2:2:23)=[];
    AccCVJS(2:2:10,:)=[];
end

cmd = sprintf('-t 4 -q -c %f',bestc);
%SimMatTrain2=SimMatSymmetricBuilding(featureTrain,bestsigma);

disp('Computing gram matrix for training set...');

distance=pdist(featureTrain,@(Xi,Xj)JSD(Xi,Xj));
distance=squareform(distance);
%distance=JSD(featureTrain(1,:),featureTrain);
SimMatTrain=exp(-sqrt(distance)/(bestsigma*bestsigma));
%SimMatTrain=squareform(SimMatTrain);
model=ovrtrain(labelTrain,[(1:NTrain)' SimMatTrain],cmd);



%SimMatTest2=SimMatBuilding(featureTest,featureTrain,bestsigma);

disp('Computing gram matrix for training vs testing set...');

distanceTest=pdist2(featureTest,featureTrain,@(Xi,Xj)JSD(Xi,Xj));
%distanceTest=squareform(distanceTest);
SimMatTest=exp(-sqrt(distanceTest)/(bestsigma*bestsigma));
[predict_label, accuracy, dec_values] = ovrpredict(labelTest, [(1:NTest)' SimMatTest], model);


%fprintf(repmat('\b',1,42*10));
  
ellapse=toc;

fprintf('Classification Accuracy by SVM with Topic Model Kernel=%.3f%%, duration=%.2f secs\n',accuracy*100,ellapse);

Output.Accuracy=accuracy;
Output.PredictedLabel=predict_label;
Output.NumberOfClass=length(unique(labelTrain));
Output.TotalNumberOfTrainData=nTrain;
Output.TotalNumberOfTestData=nTest;
Output.FeatureSize=dimension;

if IsDefault==1
    Output.DefaultSigma=bestsigma;
    Output.DefaultC=bestc;
else
    Output.AccuracyCrossValidation=AccCVJS;
    Output.BestSigma=bestsigma;
    Output.BestC=bestc;
end


if IsPlot && IsDefault==0
    %AccCVJS2=repmat(AccCVJS,10,1);
    [XXcor YYcor]=meshgrid( [-12:2:12],[-10:2:-1]);
    figure(5);mesh(XXcor,YYcor,AccCVJS,'FaceColor','green');
    alpha(0.5);
    figure(5);ylabel('log gamma','fontsize',14);
    figure(5);xlabel('log c','fontsize',14);
    figure(5);zlabel('accuracy','fontsize',14);
    str=sprintf('Max CrossValidation Accuracy = %.3f \nMax Test Accuracy = %.3f',max(max(AccCVJS)),accuracy );
    figure(5);title({'Accuracy Topic Model Kernel',str},'fontsize',14);
    colormap winter;
    axis([-20 20 -10 0 0 1]);
    set(gca, 'FontSize', 15);
end


end

function cv=SearchingBestSigma(labelTrain,featureTrain,sigma,log2c)
nTrain=length(labelTrain);
%SimMatTrain=SimMatSymmetricBuilding(featureTrain,sigma);
distance=pdist(featureTrain,@(Xi,Xj)JSD(Xi,Xj));
distance=squareform(distance);
%distance=JSD(featureTrain(1,:),featureTrain);
SimMatTrain=exp(-sqrt(distance)/(sigma*sigma));

cmd = ['-t 4 -q -c ', num2str(2^log2c)];
cv = get_cv_ac(labelTrain, [(1:nTrain)' SimMatTrain], cmd, 3);

%fprintf('cross-validation accuracy =%2.2f% at logsigma=%2.1d log2c=%2.1d',cv,sigma,log2c);

end

function d=Jensen_Shannon_Divergence(P,Q)
epsilon=0.000001;
P=P+epsilon;
Q=Q+epsilon;
M=0.5*(P+Q);
d=P.*log(P)-P.*log(M)+Q.*log(Q)-Q.*log(M);
d=0.5*d;
d=sum(d,2);
end

function out=JSD(XI,XJ)
%XI 1 x n
%XJ m2 x n
%out m2 x 1
m2=size(XJ,1);
%XI2 m2 x n
XI2= repmat(XI,[m2 ,1]);
out=Jensen_Shannon_Divergence(XI2,XJ);
end

function [ac] = get_cv_ac(y,x,param,nr_fold)
len=length(y);
ac = 0;
rand_ind = randperm(len);
for i=1:nr_fold % Cross training : folding
    test_ind=rand_ind([floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold)]');
    train_ind = [1:len]';
    train_ind(test_ind) = [];
    model = ovrtrain(y(train_ind),x(train_ind,:),param);
    [pred,a,decv] = ovrpredict(y(test_ind),x(test_ind,:),model);
    ac = ac + sum(y(test_ind)==pred);
end
ac = ac / len;
%fprintf('Cross-validation Accuracy = %g%%\n', ac * 100);
end

function [model] = ovrtrain(y, x, cmd)

labelSet = unique(y);
labelSetSize = length(labelSet);
models = cell(labelSetSize,1);

for i=1:labelSetSize
    models{i} = svmtrain(double(y == labelSet(i)), x, cmd);
end

model = struct('models', {models}, 'labelSet', labelSet);
end


function [pred, ac, decv] = ovrpredict(y, x, model)
labelSet = model.labelSet;
labelSetSize = length(labelSet);
models = model.models;
decv= zeros(size(y, 1), labelSetSize);

for i=1:labelSetSize
    %[l,a,d] = svmpredict(double(y == labelSet(i)), x, models{i},'-q');
    [T,l,a,d] = evalc('svmpredict(double(y == labelSet(i)), x, models{i})');

    %fprintf(repmat('\b',1,100));
    decv(:, i) = d * (2 * models{i}.Label(1) - 1);
end
[tmp,pred] = max(decv, [], 2);
pred = labelSet(pred);
ac = sum(y==pred) / size(x, 1);
%fprintf(repmat('\b',1,42*10));
end


function [varargout] = process_options(args, varargin)

% Check the number of input arguments
n = length(varargin);
if (mod(n, 2))
    error('Each option must be a string/value pair.');
end

% Check the number of supplied output arguments
if (nargout < (n / 2))
    error('Insufficient number of output arguments given');
elseif (nargout == (n / 2))
    warn = 1;
    nout = n / 2;
else
    warn = 0;
    nout = n / 2 + 1;
end

% Set outputs to be defaults
varargout = cell(1, nout);
for i=2:2:n
    varargout{i/2} = varargin{i};
end

% Now process all arguments
nunused = 0;
for i=1:2:length(args)
    found = 0;
    for j=1:2:n
        if strcmpi(args{i}, varargin{j})
            varargout{(j + 1)/2} = args{i + 1};
            found = 1;
            break;
        end
    end
    if (~found)
        if (warn)
            warning(sprintf('Option ''%s'' not used.', args{i}));
            args{i};
        else
            nunused = nunused + 1;
            unused{2 * nunused - 1} = args{i};
            unused{2 * nunused} = args{i + 1};
        end
    end
end

% Assign the unused arguments
if (~warn)
    if (nunused)
        varargout{nout} = unused;
    else
        varargout{nout} = cell(0);
    end
end

end