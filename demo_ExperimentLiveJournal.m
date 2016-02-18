clear all;     

% this data is a pre-processed data, which is mixture proportion derived by Latent Dirichlet Allocation from orignal 65483 dimension to 50 dimension.
load('LiveJournalData.mat');

% xxTrain is feature matrix for training [N x D]
% yyTrain is label matrix for training [N x 1]
% xxTest is feature matrix for testing [N x D]
% yyTest is label matrix for testing [N x 1]

%% experiment with default parameter (fast and recommended)
%it will takes about 3 seconds
OutputTMK = TopicModelKernel(yyTrain, xxTrain, yyTest, xxTest,'IsDefaultParameter',1);

%% experiment with searching optimal parameter (slow)
%it will takes about 2-3 minutes
%OutputTMK = TopicModelKernel( yyTrain, xxTrain, yyTest, xxTest,'IsDefaultParameter',0,'IsPlot',1);


