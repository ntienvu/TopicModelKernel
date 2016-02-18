% Topic Model Kernel for classification with SVM.
% updated: 31/10/2013
% contact: ntienvu@gmail.com
% cite: T.V. Nguyen, D. Phung, S. Venkatesh, Topic Model Kernel: An Empirical Study Towards Probabilistically Reduced Features For Classification, ICONIP 2013

% the feature data should be nonnegative

% usage: 
(1) experiment with default parameter (fast and recommended)
OutputTMK = TopicModelKernel( labelTrain, featureTrain, labelTest, featureTest);

(2) experiment with searching optimal parameter (slow, not recommended if the input data is huge)
OutputTMK = TopicModelKernel( labelTrain, featureTrain, labelTest, featureTest,'IsDefaultParameter',0,'IsPlot',1);

% input ===================================================================
% yyTrain: feature matrix for training [size of Training Data x 1]
% xxTrain: label matrix for training [size of Training Data x size of Feature] (should be nonnegative)
% yyTest: label matrix for testing [size of Testing of Data x 1]
% xxTest: feature matrix for testing [size of Testing of Data x size of Feature] (should be nonnegative)
% ==========================optional input  ===============================
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

LiveJournalData is a pre-processed data, which is mixture proportion derived by Latent Dirichlet Allocation from orignal 65483 dimension to 50 dimension.
This LiveJournal data includes 8758 posts (label is the community the post belonged) with the original feature size is 65483. The subset of data in "LiveJournalData.mat" comprises of 1000 instances for training and 1000 instances for testing.
We run LDA to extract the lowdimensional feature (size of 50), then used with SVM+TMK for classification.
