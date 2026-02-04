%% ==========================================================================
%  EEG CLASSIFICATION PIPELINE FOR MOTOR IMAGERY DATA
%  ==========================================================================
%  Description:
%    This script implements multiple classifiers for motor imagery EEG
%    classification including traditional ML and deep learning methods:
%    1. LDA (Linear Discriminant Analysis)
%    2. KNN (K-Nearest Neighbors)
%    3. SVM (Support Vector Machine)
%    4. 1D CNN (Convolutional Neural Network)
%    5. LSTM / BiLSTM Networks
%  
%  Input:
%    - fourclass_features.mat: Feature data from Part 2
%    - fourclass_eeg_processed.mat: Raw epochs for deep learning
%  
%  Output:
%    - trained_models.mat: All trained classifier models
%    - classification_results.mat: Performance metrics
%    - classification_report.txt: Detailed report
%  ==========================================================================

clear; close all; clc;
rng('default');  % Reset to default state first
rng(42);         % Then set seed
tic;

fprintf('============================================================\n');
fprintf('     EEG CLASSIFICATION PIPELINE - MOTOR IMAGERY\n');
fprintf('============================================================\n\n');

%% ==================== SECTION 1: LOAD DATA ===============================
fprintf('=== Section 1: Loading Data ===\n');

% Load feature data
FEATURE_FILE = 'fourclass_features.mat';
EPOCH_FILE = 'fourclass_eeg_processed.mat';

if ~isfile(FEATURE_FILE)
    error('Feature file not found: %s\nPlease run feature extraction first.', FEATURE_FILE);
end

load(FEATURE_FILE, 'featureData');

% Extract data
X_all = featureData.allFeatures;          % All features
X_selected = featureData.selectedFeatures; % Selected top features
y = featureData.numericLabels;             % Numeric labels
labels = featureData.labels;               % String labels
uniqueLabels = featureData.uniqueLabels;   % Unique class labels
nClasses = featureData.nClasses;

% Get dimensions
[nSamples, nFeatures] = size(X_all);
nSelectedFeatures = size(X_selected, 2);

fprintf('Data loaded successfully.\n');
fprintf('Total samples: %d\n', nSamples);
fprintf('Total features: %d\n', nFeatures);
fprintf('Selected features: %d\n', nSelectedFeatures);
fprintf('Number of classes: %d\n', nClasses);

% Display class distribution
fprintf('\nClass distribution:\n');
for c = 1:nClasses
    count = sum(y == c);
    fprintf('  Class %d (%s): %d samples (%.1f%%)\n', c, uniqueLabels{c}, count, 100*count/nSamples);
end

% Load epoch data for deep learning
if isfile(EPOCH_FILE)
    load(EPOCH_FILE, 'processedData');
    epochs = processedData.epochs;  % [channels x samples x trials]
    srate = processedData.srate;
    [nChannels, nTimePoints, ~] = size(epochs);
    deepLearningDataAvailable = true;
    fprintf('\nEpoch data loaded for deep learning.\n');
    fprintf('Epoch dimensions: %d channels x %d samples\n', nChannels, nTimePoints);
else
    deepLearningDataAvailable = false;
    fprintf('\nWarning: Epoch file not found. Deep learning will use features.\n');
end

fprintf('\n');

%% ==================== SECTION 2: DATA SPLITTING ==========================
fprintf('=== Section 2: Splitting Data ===\n');

% Stratified train/validation/test split (70/15/15)
cv = cvpartition(y, 'HoldOut', 0.3);

trainIdx = training(cv);
tempIdx = test(cv);

% Split remaining 30% into validation and test (50/50)
yTemp = y(tempIdx);
cv2 = cvpartition(yTemp, 'HoldOut', 0.5);

tempIndices = find(tempIdx);
valIdx = false(nSamples, 1);
testIdx = false(nSamples, 1);
valIdx(tempIndices(training(cv2))) = true;
testIdx(tempIndices(test(cv2))) = true;

% Feature data splits
X_train = X_selected(trainIdx, :);
X_val = X_selected(valIdx, :);
X_test = X_selected(testIdx, :);

y_train = y(trainIdx);
y_val = y(valIdx);
y_test = y(testIdx);

fprintf('Training set: %d samples (%.1f%%)\n', sum(trainIdx), 100*sum(trainIdx)/nSamples);
fprintf('Validation set: %d samples (%.1f%%)\n', sum(valIdx), 100*sum(valIdx)/nSamples);
fprintf('Test set: %d samples (%.1f%%)\n', sum(testIdx), 100*sum(testIdx)/nSamples);

% For deep learning
if deepLearningDataAvailable
    epochs_train = epochs(:, :, trainIdx);
    epochs_val = epochs(:, :, valIdx);
    epochs_test = epochs(:, :, testIdx);
end

fprintf('\n');

%% ==================== SECTION 3: CLASSIFIER INITIALIZATION ===============
fprintf('=== Section 3: Initializing Classifiers ===\n');

% Store results
results = struct();
trainedModels = struct();

% Metrics storage
classifierNames = {'LDA', 'KNN', 'SVM_Linear', 'SVM_RBF', 'CNN1D', 'LSTM', 'BiLSTM'};
nClassifiers = length(classifierNames);

metrics = struct();
metrics.accuracy = zeros(1, nClassifiers);
metrics.precision = zeros(nClassifiers, nClasses);
metrics.recall = zeros(nClassifiers, nClasses);
metrics.f1score = zeros(nClassifiers, nClasses);
metrics.confusionMatrix = cell(1, nClassifiers);
metrics.trainTime = zeros(1, nClassifiers);
metrics.predTime = zeros(1, nClassifiers);

fprintf('Classifiers to train:\n');
for i = 1:nClassifiers
    fprintf('  %d. %s\n', i, classifierNames{i});
end
fprintf('\n');

%% ==================== SECTION 4: LDA CLASSIFIER ==========================
fprintf('=== Section 4: Training LDA Classifier ===\n');

classifierIdx = 1;
tic;

% Train LDA
try
    ldaModel = fitcdiscr(X_train, y_train, ...
        'DiscrimType', 'linear', ...
        'Prior', 'uniform');
    
    trainTime = toc;
    
    % Predict
    tic;
    y_pred_lda = predict(ldaModel, X_test);
    predTime = toc;
    
    % Evaluate
    [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred_lda, nClasses);
    
    % Store results
    metrics.accuracy(classifierIdx) = acc;
    metrics.precision(classifierIdx, :) = prec;
    metrics.recall(classifierIdx, :) = rec;
    metrics.f1score(classifierIdx, :) = f1;
    metrics.confusionMatrix{classifierIdx} = cm;
    metrics.trainTime(classifierIdx) = trainTime;
    metrics.predTime(classifierIdx) = predTime;
    
    trainedModels.LDA = ldaModel;
    
    fprintf('LDA Results:\n');
    fprintf('  Accuracy: %.2f%%\n', acc * 100);
    fprintf('  Training time: %.3f s\n', trainTime);
    fprintf('  Prediction time: %.3f s\n', predTime);
    
catch ME
    fprintf('LDA failed: %s\n', ME.message);
    metrics.accuracy(classifierIdx) = 0;
end

fprintf('\n');

%% ==================== SECTION 5: KNN CLASSIFIER ==========================
fprintf('=== Section 5: Training KNN Classifier ===\n');

classifierIdx = 2;

% Find optimal K using validation set
kValues = [1, 3, 5, 7, 9, 11, 15, 21];
valAccuracy = zeros(length(kValues), 1);

fprintf('Optimizing K parameter...\n');
for i = 1:length(kValues)
    k = kValues(i);
    knnTemp = fitcknn(X_train, y_train, 'NumNeighbors', k, 'Distance', 'euclidean');
    y_val_pred = predict(knnTemp, X_val);
    valAccuracy(i) = sum(y_val_pred == y_val) / length(y_val);
    fprintf('  K=%d: Validation Accuracy = %.2f%%\n', k, valAccuracy(i)*100);
end

[~, bestKIdx] = max(valAccuracy);
bestK = kValues(bestKIdx);
fprintf('Best K: %d\n', bestK);

% Train final KNN with best K
tic;
knnModel = fitcknn(X_train, y_train, ...
    'NumNeighbors', bestK, ...
    'Distance', 'euclidean', ...
    'Standardize', false);  % Already standardized

trainTime = toc;

% Predict
tic;
y_pred_knn = predict(knnModel, X_test);
predTime = toc;

% Evaluate
[acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred_knn, nClasses);

% Store results
metrics.accuracy(classifierIdx) = acc;
metrics.precision(classifierIdx, :) = prec;
metrics.recall(classifierIdx, :) = rec;
metrics.f1score(classifierIdx, :) = f1;
metrics.confusionMatrix{classifierIdx} = cm;
metrics.trainTime(classifierIdx) = trainTime;
metrics.predTime(classifierIdx) = predTime;

trainedModels.KNN = knnModel;
trainedModels.KNN_bestK = bestK;

fprintf('KNN Results (K=%d):\n', bestK);
fprintf('  Accuracy: %.2f%%\n', acc * 100);
fprintf('  Training time: %.3f s\n', trainTime);
fprintf('  Prediction time: %.3f s\n', predTime);
fprintf('\n');

%% ==================== SECTION 6: SVM CLASSIFIER (LINEAR) =================
fprintf('=== Section 6: Training SVM (Linear Kernel) ===\n');

classifierIdx = 3;

% For multi-class, use ECOC (Error-Correcting Output Codes)
tic;

% Linear SVM template
svmTemplate = templateSVM('KernelFunction', 'linear', ...
    'BoxConstraint', 1, ...
    'Standardize', false);

svmLinearModel = fitcecoc(X_train, y_train, ...
    'Learners', svmTemplate, ...
    'Coding', 'onevsall');

trainTime = toc;

% Predict
tic;
y_pred_svm_linear = predict(svmLinearModel, X_test);
predTime = toc;

% Evaluate
[acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred_svm_linear, nClasses);

% Store results
metrics.accuracy(classifierIdx) = acc;
metrics.precision(classifierIdx, :) = prec;
metrics.recall(classifierIdx, :) = rec;
metrics.f1score(classifierIdx, :) = f1;
metrics.confusionMatrix{classifierIdx} = cm;
metrics.trainTime(classifierIdx) = trainTime;
metrics.predTime(classifierIdx) = predTime;

trainedModels.SVM_Linear = svmLinearModel;

fprintf('SVM (Linear) Results:\n');
fprintf('  Accuracy: %.2f%%\n', acc * 100);
fprintf('  Training time: %.3f s\n', trainTime);
fprintf('  Prediction time: %.3f s\n', predTime);
fprintf('\n');

%% ==================== SECTION 7: SVM CLASSIFIER (RBF) ====================
fprintf('=== Section 7: Training SVM (RBF Kernel) ===\n');

classifierIdx = 4;

% Hyperparameter optimization for RBF kernel
fprintf('Optimizing C and Gamma parameters...\n');

% Grid search
cValues = [0.1, 1, 10, 100];
gammaValues = [0.001, 0.01, 0.1, 1];
bestValAcc = 0;
bestC = 1;
bestGamma = 0.1;

for c = cValues
    for gamma = gammaValues
        svmTemp = templateSVM('KernelFunction', 'rbf', ...
            'BoxConstraint', c, ...
            'KernelScale', 1/sqrt(2*gamma));
        
        svmModel = fitcecoc(X_train, y_train, 'Learners', svmTemp, 'Coding', 'onevsall');
        y_val_pred = predict(svmModel, X_val);
        valAcc = sum(y_val_pred == y_val) / length(y_val);
        
        if valAcc > bestValAcc
            bestValAcc = valAcc;
            bestC = c;
            bestGamma = gamma;
        end
    end
end

fprintf('Best C: %.2f, Best Gamma: %.4f (Val Acc: %.2f%%)\n', bestC, bestGamma, bestValAcc*100);

% Train final SVM with best parameters
tic;
svmTemplate = templateSVM('KernelFunction', 'rbf', ...
    'BoxConstraint', bestC, ...
    'KernelScale', 1/sqrt(2*bestGamma), ...
    'Standardize', false);

svmRBFModel = fitcecoc(X_train, y_train, ...
    'Learners', svmTemplate, ...
    'Coding', 'onevsall');

trainTime = toc;

% Predict
tic;
y_pred_svm_rbf = predict(svmRBFModel, X_test);
predTime = toc;

% Evaluate
[acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred_svm_rbf, nClasses);

% Store results
metrics.accuracy(classifierIdx) = acc;
metrics.precision(classifierIdx, :) = prec;
metrics.recall(classifierIdx, :) = rec;
metrics.f1score(classifierIdx, :) = f1;
metrics.confusionMatrix{classifierIdx} = cm;
metrics.trainTime(classifierIdx) = trainTime;
metrics.predTime(classifierIdx) = predTime;

trainedModels.SVM_RBF = svmRBFModel;
trainedModels.SVM_RBF_params = struct('C', bestC, 'gamma', bestGamma);

fprintf('SVM (RBF) Results:\n');
fprintf('  Accuracy: %.2f%%\n', acc * 100);
fprintf('  Training time: %.3f s\n', trainTime);
fprintf('  Prediction time: %.3f s\n', predTime);
fprintf('\n');

%% ==================== SECTION 8: 1D CNN CLASSIFIER =======================
fprintf('=== Section 8: Training 1D CNN ===\n');

classifierIdx = 5;

% Check if Deep Learning Toolbox is available
if ~license('test', 'Neural_Network_Toolbox')
    fprintf('Warning: Deep Learning Toolbox not available. Skipping CNN.\n');
    metrics.accuracy(classifierIdx) = 0;
else
    try
        if deepLearningDataAvailable
            % Prepare data for CNN (use raw epochs)
            % Format: [samples x channels x 1 x trials] for image input
            % Or reshape for sequence input
            
            % Reshape epochs: [timepoints x channels x 1 x trials]
            X_train_cnn = permute(epochs_train, [2, 1, 3]);  % [time x ch x trials]
            X_train_cnn = reshape(X_train_cnn, [nTimePoints, nChannels, 1, sum(trainIdx)]);
            
            X_val_cnn = permute(epochs_val, [2, 1, 3]);
            X_val_cnn = reshape(X_val_cnn, [nTimePoints, nChannels, 1, sum(valIdx)]);
            
            X_test_cnn = permute(epochs_test, [2, 1, 3]);
            X_test_cnn = reshape(X_test_cnn, [nTimePoints, nChannels, 1, sum(testIdx)]);
            
            inputSize = [nTimePoints, nChannels, 1];
        else
            % Use features if epochs not available
            X_train_cnn = reshape(X_train', [nSelectedFeatures, 1, 1, size(X_train, 1)]);
            X_val_cnn = reshape(X_val', [nSelectedFeatures, 1, 1, size(X_val, 1)]);
            X_test_cnn = reshape(X_test', [nSelectedFeatures, 1, 1, size(X_test, 1)]);
            inputSize = [nSelectedFeatures, 1, 1];
        end
        
        % Convert labels to categorical
        y_train_cat = categorical(y_train);
        y_val_cat = categorical(y_val);
        y_test_cat = categorical(y_test);
        
        % Define CNN architecture
        layers = [
            imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
            
            % First Conv Block
            convolution2dLayer([5, 1], 32, 'Padding', 'same', 'Name', 'conv1')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')
            maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Padding', 'same', 'Name', 'pool1')
            dropoutLayer(0.25, 'Name', 'drop1')
            
            % Second Conv Block
            convolution2dLayer([5, 1], 64, 'Padding', 'same', 'Name', 'conv2')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')
            maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Padding', 'same', 'Name', 'pool2')
            dropoutLayer(0.25, 'Name', 'drop2')
            
            % Third Conv Block
            convolution2dLayer([3, 1], 128, 'Padding', 'same', 'Name', 'conv3')
            batchNormalizationLayer('Name', 'bn3')
            reluLayer('Name', 'relu3')
            globalAveragePooling2dLayer('Name', 'gap')
            
            % Fully Connected
            fullyConnectedLayer(64, 'Name', 'fc1')
            reluLayer('Name', 'relu_fc')
            dropoutLayer(0.5, 'Name', 'drop3')
            fullyConnectedLayer(nClasses, 'Name', 'fc2')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'output')
        ];
        
        % Training options
        options = trainingOptions('adam', ...
            'InitialLearnRate', 0.001, ...
            'MaxEpochs', 50, ...
            'MiniBatchSize', 32, ...
            'ValidationData', {X_val_cnn, y_val_cat}, ...
            'ValidationFrequency', 10, ...
            'ValidationPatience', 10, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', true, ...
            'VerboseFrequency', 20, ...
            'Plots', 'none', ...  % Set to 'training-progress' to see plot
            'L2Regularization', 0.001, ...
            'ExecutionEnvironment', 'auto');
        
        % Train CNN
        fprintf('Training CNN (this may take a few minutes)...\n');
        tic;
        cnnModel = trainNetwork(X_train_cnn, y_train_cat, layers, options);
        trainTime = toc;
        
        % Predict
        tic;
        y_pred_cnn_cat = classify(cnnModel, X_test_cnn);
        predTime = toc;
        
        y_pred_cnn = double(y_pred_cnn_cat);
        
        % Evaluate
        [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred_cnn, nClasses);
        
        % Store results
        metrics.accuracy(classifierIdx) = acc;
        metrics.precision(classifierIdx, :) = prec;
        metrics.recall(classifierIdx, :) = rec;
        metrics.f1score(classifierIdx, :) = f1;
        metrics.confusionMatrix{classifierIdx} = cm;
        metrics.trainTime(classifierIdx) = trainTime;
        metrics.predTime(classifierIdx) = predTime;
        
        trainedModels.CNN1D = cnnModel;
        
        fprintf('CNN Results:\n');
        fprintf('  Accuracy: %.2f%%\n', acc * 100);
        fprintf('  Training time: %.2f s\n', trainTime);
        fprintf('  Prediction time: %.3f s\n', predTime);
        
    catch ME
        fprintf('CNN training failed: %s\n', ME.message);
        metrics.accuracy(classifierIdx) = 0;
    end
end

fprintf('\n');

%% ==================== SECTION 9: LSTM CLASSIFIER =========================
fprintf('=== Section 9: Training LSTM Network ===\n');

classifierIdx = 6;

if ~license('test', 'Neural_Network_Toolbox')
    fprintf('Warning: Deep Learning Toolbox not available. Skipping LSTM.\n');
    metrics.accuracy(classifierIdx) = 0;
else
    try
        % Prepare sequence data for LSTM
        % Format: cell array of [features x timepoints] sequences
        
        if deepLearningDataAvailable
            % Use raw epochs
            X_train_lstm = cell(sum(trainIdx), 1);
            X_val_lstm = cell(sum(valIdx), 1);
            X_test_lstm = cell(sum(testIdx), 1);
            
            for i = 1:sum(trainIdx)
                X_train_lstm{i} = epochs_train(:, :, i);  % [channels x time]
            end
            for i = 1:sum(valIdx)
                X_val_lstm{i} = epochs_val(:, :, i);
            end
            for i = 1:sum(testIdx)
                X_test_lstm{i} = epochs_test(:, :, i);
            end
            
            inputDim = nChannels;
        else
            % Use features (reshape as sequence)
            seqLength = 10;  % Divide features into 10 time steps
            featsPerStep = floor(nSelectedFeatures / seqLength);
            
            X_train_lstm = cell(size(X_train, 1), 1);
            X_val_lstm = cell(size(X_val, 1), 1);
            X_test_lstm = cell(size(X_test, 1), 1);
            
            for i = 1:size(X_train, 1)
                temp = X_train(i, 1:seqLength*featsPerStep);
                X_train_lstm{i} = reshape(temp, [featsPerStep, seqLength]);
            end
            for i = 1:size(X_val, 1)
                temp = X_val(i, 1:seqLength*featsPerStep);
                X_val_lstm{i} = reshape(temp, [featsPerStep, seqLength]);
            end
            for i = 1:size(X_test, 1)
                temp = X_test(i, 1:seqLength*featsPerStep);
                X_test_lstm{i} = reshape(temp, [featsPerStep, seqLength]);
            end
            
            inputDim = featsPerStep;
        end
        
        y_train_cat = categorical(y_train);
        y_val_cat = categorical(y_val);
        y_test_cat = categorical(y_test);
        
        % Define LSTM architecture
        numHiddenUnits = 64;
        
        layers = [
            sequenceInputLayer(inputDim, 'Name', 'input')
            lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm1')
            dropoutLayer(0.3, 'Name', 'drop1')
            fullyConnectedLayer(32, 'Name', 'fc1')
            reluLayer('Name', 'relu1')
            dropoutLayer(0.3, 'Name', 'drop2')
            fullyConnectedLayer(nClasses, 'Name', 'fc2')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'output')
        ];
        
        % Training options
        options = trainingOptions('adam', ...
            'InitialLearnRate', 0.001, ...
            'MaxEpochs', 50, ...
            'MiniBatchSize', 32, ...
            'ValidationData', {X_val_lstm, y_val_cat}, ...
            'ValidationFrequency', 10, ...
            'ValidationPatience', 10, ...
            'GradientThreshold', 1, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', true, ...
            'VerboseFrequency', 20, ...
            'Plots', 'none', ...
            'ExecutionEnvironment', 'auto');
        
        % Train LSTM
        fprintf('Training LSTM (this may take a few minutes)...\n');
        tic;
        lstmModel = trainNetwork(X_train_lstm, y_train_cat, layers, options);
        trainTime = toc;
        
        % Predict
        tic;
        y_pred_lstm_cat = classify(lstmModel, X_test_lstm);
        predTime = toc;
        
        y_pred_lstm = double(y_pred_lstm_cat);
        
        % Evaluate
        [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred_lstm, nClasses);
        
        % Store results
        metrics.accuracy(classifierIdx) = acc;
        metrics.precision(classifierIdx, :) = prec;
        metrics.recall(classifierIdx, :) = rec;
        metrics.f1score(classifierIdx, :) = f1;
        metrics.confusionMatrix{classifierIdx} = cm;
        metrics.trainTime(classifierIdx) = trainTime;
        metrics.predTime(classifierIdx) = predTime;
        
        trainedModels.LSTM = lstmModel;
        
        fprintf('LSTM Results:\n');
        fprintf('  Accuracy: %.2f%%\n', acc * 100);
        fprintf('  Training time: %.2f s\n', trainTime);
        fprintf('  Prediction time: %.3f s\n', predTime);
        
    catch ME
        fprintf('LSTM training failed: %s\n', ME.message);
        metrics.accuracy(classifierIdx) = 0;
    end
end

fprintf('\n');

%% ==================== SECTION 10: BiLSTM CLASSIFIER ======================
fprintf('=== Section 10: Training BiLSTM Network ===\n');

classifierIdx = 7;

if ~license('test', 'Neural_Network_Toolbox')
    fprintf('Warning: Deep Learning Toolbox not available. Skipping BiLSTM.\n');
    metrics.accuracy(classifierIdx) = 0;
else
    try
        % Use same data as LSTM
        
        % Define BiLSTM architecture
        numHiddenUnits = 64;
        
        layers = [
            sequenceInputLayer(inputDim, 'Name', 'input')
            bilstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'bilstm1')
            dropoutLayer(0.3, 'Name', 'drop1')
            fullyConnectedLayer(32, 'Name', 'fc1')
            reluLayer('Name', 'relu1')
            dropoutLayer(0.3, 'Name', 'drop2')
            fullyConnectedLayer(nClasses, 'Name', 'fc2')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'output')
        ];
        
        % Training options (same as LSTM)
        options = trainingOptions('adam', ...
            'InitialLearnRate', 0.001, ...
            'MaxEpochs', 50, ...
            'MiniBatchSize', 32, ...
            'ValidationData', {X_val_lstm, y_val_cat}, ...
            'ValidationFrequency', 10, ...
            'ValidationPatience', 10, ...
            'GradientThreshold', 1, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', true, ...
            'VerboseFrequency', 20, ...
            'Plots', 'none', ...
            'ExecutionEnvironment', 'auto');
        
        % Train BiLSTM
        fprintf('Training BiLSTM (this may take a few minutes)...\n');
        tic;
        bilstmModel = trainNetwork(X_train_lstm, y_train_cat, layers, options);
        trainTime = toc;
        
        % Predict
        tic;
        y_pred_bilstm_cat = classify(bilstmModel, X_test_lstm);
        predTime = toc;
        
        y_pred_bilstm = double(y_pred_bilstm_cat);
        
        % Evaluate
        [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred_bilstm, nClasses);
        
        % Store results
        metrics.accuracy(classifierIdx) = acc;
        metrics.precision(classifierIdx, :) = prec;
        metrics.recall(classifierIdx, :) = rec;
        metrics.f1score(classifierIdx, :) = f1;
        metrics.confusionMatrix{classifierIdx} = cm;
        metrics.trainTime(classifierIdx) = trainTime;
        metrics.predTime(classifierIdx) = predTime;
        
        trainedModels.BiLSTM = bilstmModel;
        
        fprintf('BiLSTM Results:\n');
        fprintf('  Accuracy: %.2f%%\n', acc * 100);
        fprintf('  Training time: %.2f s\n', trainTime);
        fprintf('  Prediction time: %.3f s\n', predTime);
        
    catch ME
        fprintf('BiLSTM training failed: %s\n', ME.message);
        metrics.accuracy(classifierIdx) = 0;
    end
end

fprintf('\n');

%% ==================== SECTION 11: CROSS-VALIDATION =======================
fprintf('=== Section 11: K-Fold Cross-Validation ===\n');

nFolds = 5;
fprintf('Performing %d-fold cross-validation for traditional classifiers...\n', nFolds);

cvPartition = cvpartition(y, 'KFold', nFolds);

% Cross-validation results
cvResults = struct();
cvResults.LDA = zeros(nFolds, 1);
cvResults.KNN = zeros(nFolds, 1);
cvResults.SVM_Linear = zeros(nFolds, 1);
cvResults.SVM_RBF = zeros(nFolds, 1);

for fold = 1:nFolds
    fprintf('  Fold %d/%d...\n', fold, nFolds);
    
    % Get fold indices
    trainFoldIdx = training(cvPartition, fold);
    testFoldIdx = test(cvPartition, fold);
    
    X_train_fold = X_selected(trainFoldIdx, :);
    X_test_fold = X_selected(testFoldIdx, :);
    y_train_fold = y(trainFoldIdx);
    y_test_fold = y(testFoldIdx);
    
    % LDA
    ldaCV = fitcdiscr(X_train_fold, y_train_fold, 'DiscrimType', 'linear');
    y_pred_cv = predict(ldaCV, X_test_fold);
    cvResults.LDA(fold) = sum(y_pred_cv == y_test_fold) / length(y_test_fold);
    
    % KNN
    knnCV = fitcknn(X_train_fold, y_train_fold, 'NumNeighbors', bestK);
    y_pred_cv = predict(knnCV, X_test_fold);
    cvResults.KNN(fold) = sum(y_pred_cv == y_test_fold) / length(y_test_fold);
    
    % SVM Linear
    svmLinearCV = fitcecoc(X_train_fold, y_train_fold, ...
        'Learners', templateSVM('KernelFunction', 'linear'), 'Coding', 'onevsall');
    y_pred_cv = predict(svmLinearCV, X_test_fold);
    cvResults.SVM_Linear(fold) = sum(y_pred_cv == y_test_fold) / length(y_test_fold);
    
    % SVM RBF
    svmRBFCV = fitcecoc(X_train_fold, y_train_fold, ...
        'Learners', templateSVM('KernelFunction', 'rbf', 'BoxConstraint', bestC, ...
            'KernelScale', 1/sqrt(2*bestGamma)), 'Coding', 'onevsall');
    y_pred_cv = predict(svmRBFCV, X_test_fold);
    cvResults.SVM_RBF(fold) = sum(y_pred_cv == y_test_fold) / length(y_test_fold);
end

% Display CV results
fprintf('\nCross-Validation Results (%d-fold):\n', nFolds);
fprintf('  LDA:        %.2f%% (+/- %.2f%%)\n', mean(cvResults.LDA)*100, std(cvResults.LDA)*100);
fprintf('  KNN:        %.2f%% (+/- %.2f%%)\n', mean(cvResults.KNN)*100, std(cvResults.KNN)*100);
fprintf('  SVM Linear: %.2f%% (+/- %.2f%%)\n', mean(cvResults.SVM_Linear)*100, std(cvResults.SVM_Linear)*100);
fprintf('  SVM RBF:    %.2f%% (+/- %.2f%%)\n', mean(cvResults.SVM_RBF)*100, std(cvResults.SVM_RBF)*100);
fprintf('\n');

%% ==================== SECTION 12: RESULTS COMPARISON =====================
fprintf('=== Section 12: Comparing All Classifiers ===\n');

% Find best classifier
[bestAcc, bestIdx] = max(metrics.accuracy);
fprintf('\n========================================\n');
fprintf('CLASSIFICATION RESULTS SUMMARY\n');
fprintf('========================================\n\n');

fprintf('%-15s | %8s | %8s | %10s | %10s\n', 'Classifier', 'Accuracy', 'F1-Score', 'Train Time', 'Pred Time');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:nClassifiers
    if metrics.accuracy(i) > 0
        avgF1 = mean(metrics.f1score(i, :));
        fprintf('%-15s | %7.2f%% | %8.4f | %8.3f s | %8.4f s\n', ...
            classifierNames{i}, metrics.accuracy(i)*100, avgF1, ...
            metrics.trainTime(i), metrics.predTime(i));
    else
        fprintf('%-15s | %8s | %8s | %10s | %10s\n', ...
            classifierNames{i}, 'N/A', 'N/A', 'N/A', 'N/A');
    end
end

fprintf('%s\n', repmat('-', 1, 60));
fprintf('\nBest Classifier: %s (Accuracy: %.2f%%)\n', classifierNames{bestIdx}, bestAcc*100);
fprintf('\n');

%% ==================== SECTION 13: VISUALIZATION ==========================
fprintf('=== Section 13: Generating Visualizations ===\n');

figure('Name', 'Classification Results', 'Position', [50, 50, 1600, 1000]);

% --- Subplot 1: Accuracy Comparison ---
subplot(2, 3, 1);
validIdx = metrics.accuracy > 0;
barData = metrics.accuracy(validIdx) * 100;
bar(barData, 'FaceColor', [0.3, 0.6, 0.9]);
set(gca, 'XTickLabel', classifierNames(validIdx), 'XTickLabelRotation', 45);
ylabel('Accuracy (%)');
title('Classification Accuracy Comparison');
grid on;
ylim([0, 100]);

% Add value labels on bars
for i = 1:length(barData)
    text(i, barData(i) + 2, sprintf('%.1f%%', barData(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

% --- Subplot 2: Training Time Comparison ---
subplot(2, 3, 2);
trainTimes = metrics.trainTime(validIdx);
bar(trainTimes, 'FaceColor', [0.9, 0.4, 0.3]);
set(gca, 'XTickLabel', classifierNames(validIdx), 'XTickLabelRotation', 45);
ylabel('Time (seconds)');
title('Training Time Comparison');
grid on;

% --- Subplot 3: F1-Score by Class (Best Model) ---
subplot(2, 3, 3);
bar(metrics.f1score(bestIdx, :), 'FaceColor', [0.4, 0.8, 0.4]);
set(gca, 'XTickLabel', uniqueLabels, 'XTickLabelRotation', 45);
ylabel('F1-Score');
title(sprintf('F1-Score by Class (%s)', classifierNames{bestIdx}));
grid on;
ylim([0, 1]);

% --- Subplot 4: Confusion Matrix (Best Model) ---
subplot(2, 3, 4);
cm = metrics.confusionMatrix{bestIdx};
confusionchart(cm, uniqueLabels);
title(sprintf('Confusion Matrix (%s)', classifierNames{bestIdx}));

% --- Subplot 5: Cross-Validation Results ---
subplot(2, 3, 5);
cvData = [cvResults.LDA, cvResults.KNN, cvResults.SVM_Linear, cvResults.SVM_RBF] * 100;
boxplot(cvData, 'Labels', {'LDA', 'KNN', 'SVM-L', 'SVM-RBF'});
ylabel('Accuracy (%)');
title('Cross-Validation Results');
grid on;

% --- Subplot 6: Precision-Recall by Class ---
subplot(2, 3, 6);
hold on;
barWidth = 0.35;
x = 1:nClasses;
bar(x - barWidth/2, metrics.precision(bestIdx, :), barWidth, 'FaceColor', [0.2, 0.6, 0.8], 'DisplayName', 'Precision');
bar(x + barWidth/2, metrics.recall(bestIdx, :), barWidth, 'FaceColor', [0.8, 0.4, 0.2], 'DisplayName', 'Recall');
hold off;
set(gca, 'XTick', x, 'XTickLabel', uniqueLabels, 'XTickLabelRotation', 45);
ylabel('Score');
title(sprintf('Precision & Recall (%s)', classifierNames{bestIdx}));
legend('Location', 'best');
grid on;
ylim([0, 1]);

% Save figure
saveas(gcf, 'classification_results.png');
fprintf('Visualization saved to: classification_results.png\n');

% Additional figure: All confusion matrices
figure('Name', 'All Confusion Matrices', 'Position', [100, 100, 1400, 800]);
plotIdx = 0;
for i = 1:nClassifiers
    if metrics.accuracy(i) > 0
        plotIdx = plotIdx + 1;
        subplot(2, 4, plotIdx);
        cm = metrics.confusionMatrix{i};
        imagesc(cm);
        colorbar;
        colormap(gca, 'parula');
        title(sprintf('%s (%.1f%%)', classifierNames{i}, metrics.accuracy(i)*100));
        xlabel('Predicted');
        ylabel('Actual');
        set(gca, 'XTick', 1:nClasses, 'YTick', 1:nClasses);
        
        % Add text annotations
        for r = 1:nClasses
            for c = 1:nClasses
                text(c, r, num2str(cm(r, c)), 'HorizontalAlignment', 'center', ...
                    'Color', 'white', 'FontWeight', 'bold');
            end
        end
    end
end
saveas(gcf, 'all_confusion_matrices.png');
fprintf('All confusion matrices saved to: all_confusion_matrices.png\n\n');

%% ==================== SECTION 14: SAVE RESULTS ===========================
fprintf('=== Section 14: Saving Results ===\n');

% Classification results
classificationResults = struct();
classificationResults.classifierNames = classifierNames;
classificationResults.metrics = metrics;
classificationResults.cvResults = cvResults;
classificationResults.bestClassifier = classifierNames{bestIdx};
classificationResults.bestAccuracy = bestAcc;
classificationResults.uniqueLabels = uniqueLabels;
classificationResults.nClasses = nClasses;
classificationResults.testLabels = y_test;
classificationResults.trainTestSplit = struct(...
    'nTrain', sum(trainIdx), ...
    'nVal', sum(valIdx), ...
    'nTest', sum(testIdx));

% Save classification results
save('classification_results.mat', 'classificationResults', '-v7.3');
fprintf('Classification results saved to: classification_results.mat\n');

% Save trained models
save('trained_models.mat', 'trainedModels', '-v7.3');
fprintf('Trained models saved to: trained_models.mat\n\n');

%% ==================== SECTION 15: GENERATE REPORT ========================
fprintf('=== Section 15: Generating Classification Report ===\n');

REPORT_FILE = 'classification_report.txt';
fid = fopen(REPORT_FILE, 'w');

fprintf(fid, '================================================================\n');
fprintf(fid, '         EEG CLASSIFICATION REPORT - MOTOR IMAGERY\n');
fprintf(fid, '================================================================\n\n');

fprintf(fid, 'Classification Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'MATLAB Version: %s\n\n', version);

fprintf(fid, '--- DATA SUMMARY ---\n');
fprintf(fid, 'Total samples: %d\n', nSamples);
fprintf(fid, 'Number of features: %d (selected: %d)\n', nFeatures, nSelectedFeatures);
fprintf(fid, 'Number of classes: %d\n', nClasses);
fprintf(fid, 'Training samples: %d (%.1f%%)\n', sum(trainIdx), 100*sum(trainIdx)/nSamples);
fprintf(fid, 'Validation samples: %d (%.1f%%)\n', sum(valIdx), 100*sum(valIdx)/nSamples);
fprintf(fid, 'Test samples: %d (%.1f%%)\n\n', sum(testIdx), 100*sum(testIdx)/nSamples);

fprintf(fid, '--- CLASSIFICATION RESULTS ---\n\n');
fprintf(fid, '%-15s | %8s | %8s | %10s\n', 'Classifier', 'Accuracy', 'Avg F1', 'Train Time');
fprintf(fid, '%s\n', repmat('-', 1, 50));

for i = 1:nClassifiers
    if metrics.accuracy(i) > 0
        avgF1 = mean(metrics.f1score(i, :));
        fprintf(fid, '%-15s | %7.2f%% | %8.4f | %8.2f s\n', ...
            classifierNames{i}, metrics.accuracy(i)*100, avgF1, metrics.trainTime(i));
    else
        fprintf(fid, '%-15s | %8s | %8s | %10s\n', classifierNames{i}, 'N/A', 'N/A', 'N/A');
    end
end

fprintf(fid, '\n--- CROSS-VALIDATION RESULTS (%d-Fold) ---\n\n', nFolds);
fprintf(fid, 'LDA:        %.2f%% (+/- %.2f%%)\n', mean(cvResults.LDA)*100, std(cvResults.LDA)*100);
fprintf(fid, 'KNN (K=%d): %.2f%% (+/- %.2f%%)\n', bestK, mean(cvResults.KNN)*100, std(cvResults.KNN)*100);
fprintf(fid, 'SVM Linear: %.2f%% (+/- %.2f%%)\n', mean(cvResults.SVM_Linear)*100, std(cvResults.SVM_Linear)*100);
fprintf(fid, 'SVM RBF:    %.2f%% (+/- %.2f%%)\n', mean(cvResults.SVM_RBF)*100, std(cvResults.SVM_RBF)*100);

fprintf(fid, '\n--- BEST MODEL DETAILS ---\n\n');
fprintf(fid, 'Best Classifier: %s\n', classifierNames{bestIdx});
fprintf(fid, 'Test Accuracy: %.2f%%\n\n', bestAcc * 100);

fprintf(fid, 'Per-Class Metrics:\n');
fprintf(fid, '%-15s | %10s | %10s | %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
fprintf(fid, '%s\n', repmat('-', 1, 50));
for c = 1:nClasses
    fprintf(fid, '%-15s | %9.4f | %9.4f | %9.4f\n', ...
        uniqueLabels{c}, metrics.precision(bestIdx, c), ...
        metrics.recall(bestIdx, c), metrics.f1score(bestIdx, c));
end

fprintf(fid, '\nConfusion Matrix:\n');
cm = metrics.confusionMatrix{bestIdx};
fprintf(fid, '          ');
for c = 1:nClasses
    fprintf(fid, '%8s ', uniqueLabels{c}(1:min(8, length(uniqueLabels{c}))));
end
fprintf(fid, '\n');
for r = 1:nClasses
    fprintf(fid, '%8s: ', uniqueLabels{r}(1:min(8, length(uniqueLabels{r}))));
    for c = 1:nClasses
        fprintf(fid, '%8d ', cm(r, c));
    end
    fprintf(fid, '\n');
end

fprintf(fid, '\n--- OUTPUT FILES ---\n');
fprintf(fid, '1. classification_results.mat (all metrics)\n');
fprintf(fid, '2. trained_models.mat (trained classifiers)\n');
fprintf(fid, '3. classification_results.png (visualization)\n');
fprintf(fid, '4. all_confusion_matrices.png\n');
fprintf(fid, '5. %s (this report)\n', REPORT_FILE);

fprintf(fid, '\n================================================================\n');
fprintf(fid, '                    END OF REPORT\n');
fprintf(fid, '================================================================\n');

fclose(fid);
fprintf('Report saved to: %s\n', REPORT_FILE);

%% ==================== COMPLETION =========================================
totalTime = toc;

fprintf('\n');
fprintf('============================================================\n');
fprintf('  CLASSIFICATION COMPLETE\n');
fprintf('============================================================\n');
fprintf('Total processing time: %.2f seconds (%.2f minutes)\n', totalTime, totalTime/60);
fprintf('Best classifier: %s\n', classifierNames{bestIdx});
fprintf('Best accuracy: %.2f%%\n', bestAcc * 100);
fprintf('============================================================\n');

%% ==================== HELPER FUNCTIONS ===================================

function [accuracy, precision, recall, f1, confMat] = evaluateClassifier(yTrue, yPred, nClasses)
    % Compute classification metrics
    %
    % Inputs:
    %   yTrue: true labels
    %   yPred: predicted labels
    %   nClasses: number of classes
    %
    % Outputs:
    %   accuracy: overall accuracy
    %   precision: per-class precision
    %   recall: per-class recall
    %   f1: per-class F1-score
    %   confMat: confusion matrix
    
    % Ensure column vectors
    yTrue = yTrue(:);
    yPred = yPred(:);
    
    % Overall accuracy
    accuracy = sum(yTrue == yPred) / length(yTrue);
    
    % Confusion matrix
    confMat = confusionmat(yTrue, yPred);
    
    % Pad confusion matrix if necessary
    if size(confMat, 1) < nClasses
        confMat(nClasses, nClasses) = 0;
    end
    
    % Per-class metrics
    precision = zeros(1, nClasses);
    recall = zeros(1, nClasses);
    f1 = zeros(1, nClasses);
    
    for c = 1:nClasses
        tp = confMat(c, c);
        fp = sum(confMat(:, c)) - tp;
        fn = sum(confMat(c, :)) - tp;
        
        % Precision
        if (tp + fp) > 0
            precision(c) = tp / (tp + fp);
        else
            precision(c) = 0;
        end
        
        % Recall
        if (tp + fn) > 0
            recall(c) = tp / (tp + fn);
        else
            recall(c) = 0;
        end
        
        % F1-Score
        if (precision(c) + recall(c)) > 0
            f1(c) = 2 * precision(c) * recall(c) / (precision(c) + recall(c));
        else
            f1(c) = 0;
        end
    end
end