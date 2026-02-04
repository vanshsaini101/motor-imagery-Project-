%% ==========================================================================
%  EEG EVALUATION, COMPARISON & IMPROVEMENT PIPELINE
%  ==========================================================================
%  Description:
%    This script performs comprehensive evaluation of EEG classification:
%    1. 10-Fold Cross-Validation for all methods
%    2. Feature extraction method comparison
%    3. Classifier comparison
%    4. Improvement techniques (Ensemble, Feature fusion, Hyperparameter tuning)
%    5. Statistical analysis and final reporting
%  
%  Input:
%    - fourclass_features.mat: Feature data from Part 2
%    - fourclass_eeg_processed.mat: Preprocessed epochs
%  
%  Output:
%    - evaluation_results.mat: Complete evaluation metrics
%    - final_report.txt: Comprehensive analysis report
%    - Multiple visualization figures
%  ==========================================================================

clear; close all; clc;

% Initialize random number generator
try
    rng('default');
    rng(42);
catch
    rng(42, 'twister');
end

tic;

fprintf('============================================================\n');
fprintf('  EEG EVALUATION, COMPARISON & IMPROVEMENT PIPELINE\n');
fprintf('============================================================\n\n');

%% ==================== SECTION 1: LOAD ALL DATA ===========================
fprintf('=== Section 1: Loading Data ===\n');

% Load feature data
FEATURE_FILE = 'fourclass_features.mat';
EPOCH_FILE = 'fourclass_eeg_processed.mat';

if ~isfile(FEATURE_FILE)
    error('Feature file not found: %s', FEATURE_FILE);
end

load(FEATURE_FILE, 'featureData');
load(EPOCH_FILE, 'processedData');

% Extract all feature sets
features = struct();
features.bandPower = featureData.bandPower;
features.relativeBandPower = featureData.relativeBandPower;
features.csp = featureData.csp;
features.fbcsp = featureData.fbcsp;
features.wavelet = featureData.wavelet;
features.timeDomain = featureData.timeDomain;
features.hjorth = featureData.hjorth;
features.all = featureData.allFeatures;
features.selected = featureData.selectedFeatures;

% Labels
y = featureData.numericLabels;
uniqueLabels = featureData.uniqueLabels;
nClasses = featureData.nClasses;
nSamples = length(y);

% Epochs for deep learning
epochs = processedData.epochs;
srate = processedData.srate;
[nChannels, nTimePoints, ~] = size(epochs);

% Feature set names
featureSetNames = {'BandPower', 'RelBandPower', 'CSP', 'FBCSP', 'Wavelet', ...
                   'TimeDomain', 'Hjorth', 'AllFeatures', 'SelectedFeatures'};

fprintf('Data loaded successfully.\n');
fprintf('Total samples: %d\n', nSamples);
fprintf('Number of classes: %d\n', nClasses);
fprintf('Feature sets available: %d\n\n', length(featureSetNames));

%% ==================== SECTION 2: 10-FOLD CROSS-VALIDATION SETUP ==========
fprintf('=== Section 2: Setting Up 10-Fold Cross-Validation ===\n');

nFolds = 10;
cvPartition = cvpartition(y, 'KFold', nFolds);

fprintf('Cross-validation: %d-fold stratified\n', nFolds);
fprintf('Samples per fold: ~%d\n\n', round(nSamples/nFolds));

% Classifier templates
classifierNames = {'LDA', 'QDA', 'KNN', 'SVM_Linear', 'SVM_RBF', 'NaiveBayes', 'RandomForest', 'AdaBoost'};
nClassifiers = length(classifierNames);

%% ==================== SECTION 3: FEATURE EXTRACTION COMPARISON ===========
% Compare all feature extraction methods using LDA as baseline classifier

fprintf('=== Section 3: Comparing Feature Extraction Methods with LDA ===\n');

featureSetFields = {'bandPower', 'relativeBandPower', 'csp', 'fbcsp', 'wavelet', ...
                    'timeDomain', 'hjorth', 'all', 'selected'};

nFeatureSets = length(featureSetFields);
featureCompResults = struct();
featureCompResults.accuracy = zeros(nFeatureSets, nFolds);
featureCompResults.f1score = zeros(nFeatureSets, nFolds);
featureCompResults.precision = zeros(nFeatureSets, nFolds, nClasses);
featureCompResults.recall = zeros(nFeatureSets, nFolds, nClasses);

for fs = 1:nFeatureSets
    featureName = featureSetFields{fs};
    X = features.(featureName);
    
    % Normalize features
    X = normalizeFeatures(X);
    
    fprintf('  Evaluating %s (%d features)...\n', featureSetNames{fs}, size(X, 2));
    
    for fold = 1:nFolds
        trainIdx = training(cvPartition, fold);
        testIdx = test(cvPartition, fold);
        
        X_train = X(trainIdx, :);
        X_test = X(testIdx, :);
        y_train = y(trainIdx);
        y_test = y(testIdx);
        
        % Train LDA
        try
            model = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
            y_pred = predict(model, X_test);
            
            % Compute metrics
            [acc, prec, rec, f1, ~] = evaluateClassifier(y_test, y_pred, nClasses);
            
            featureCompResults.accuracy(fs, fold) = acc;
            featureCompResults.f1score(fs, fold) = mean(f1);
            featureCompResults.precision(fs, fold, :) = prec;
            featureCompResults.recall(fs, fold, :) = rec;
        catch
            featureCompResults.accuracy(fs, fold) = 0;
            featureCompResults.f1score(fs, fold) = 0;
        end
    end
end

% Compute mean and std
featureCompResults.meanAccuracy = mean(featureCompResults.accuracy, 2);
featureCompResults.stdAccuracy = std(featureCompResults.accuracy, 0, 2);
featureCompResults.meanF1 = mean(featureCompResults.f1score, 2);
featureCompResults.stdF1 = std(featureCompResults.f1score, 0, 2);

% Find best feature set
[~, bestFeatureIdx] = max(featureCompResults.meanAccuracy);
bestFeatureSet = featureSetFields{bestFeatureIdx};

fprintf('\n--- Feature Extraction Comparison Results ---\n');
fprintf('%-20s | %12s | %12s\n', 'Feature Set', 'Accuracy', 'F1-Score');
fprintf('%s\n', repmat('-', 1, 50));
for fs = 1:nFeatureSets
    fprintf('%-20s | %5.2f +/- %4.2f | %5.2f +/- %4.2f\n', ...
        featureSetNames{fs}, ...
        featureCompResults.meanAccuracy(fs)*100, featureCompResults.stdAccuracy(fs)*100, ...
        featureCompResults.meanF1(fs)*100, featureCompResults.stdF1(fs)*100);
end
fprintf('\nBest Feature Set: %s (%.2f%%)\n\n', featureSetNames{bestFeatureIdx}, ...
    featureCompResults.meanAccuracy(bestFeatureIdx)*100);

%% ==================== SECTION 4: CLASSIFIER COMPARISON ===================
% Compare all classifiers using the best feature set

fprintf('=== Section 4: Comparing All Classifiers ===\n');

% Use best feature set
X_best = features.(bestFeatureSet);
X_best = normalizeFeatures(X_best);

classifierResults = struct();
classifierResults.accuracy = zeros(nClassifiers, nFolds);
classifierResults.f1score = zeros(nClassifiers, nFolds);
classifierResults.precision = zeros(nClassifiers, nFolds, nClasses);
classifierResults.recall = zeros(nClassifiers, nFolds, nClasses);
classifierResults.trainTime = zeros(nClassifiers, nFolds);
classifierResults.predTime = zeros(nClassifiers, nFolds);
classifierResults.confusionMatrix = cell(nClassifiers, 1);

% Aggregate confusion matrices
for c = 1:nClassifiers
    classifierResults.confusionMatrix{c} = zeros(nClasses, nClasses);
end

for fold = 1:nFolds
    fprintf('  Fold %d/%d...\n', fold, nFolds);
    
    trainIdx = training(cvPartition, fold);
    testIdx = test(cvPartition, fold);
    
    X_train = X_best(trainIdx, :);
    X_test = X_best(testIdx, :);
    y_train = y(trainIdx);
    y_test = y(testIdx);
    
    % 1. LDA
    c = 1;
    tic;
    model = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
    classifierResults.trainTime(c, fold) = toc;
    tic;
    y_pred = predict(model, X_test);
    classifierResults.predTime(c, fold) = toc;
    [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred, nClasses);
    classifierResults.accuracy(c, fold) = acc;
    classifierResults.f1score(c, fold) = mean(f1);
    classifierResults.precision(c, fold, :) = prec;
    classifierResults.recall(c, fold, :) = rec;
    classifierResults.confusionMatrix{c} = classifierResults.confusionMatrix{c} + cm;
    
    % 2. QDA
    c = 2;
    try
        tic;
        model = fitcdiscr(X_train, y_train, 'DiscrimType', 'quadratic');
        classifierResults.trainTime(c, fold) = toc;
        tic;
        y_pred = predict(model, X_test);
        classifierResults.predTime(c, fold) = toc;
        [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred, nClasses);
        classifierResults.accuracy(c, fold) = acc;
        classifierResults.f1score(c, fold) = mean(f1);
        classifierResults.precision(c, fold, :) = prec;
        classifierResults.recall(c, fold, :) = rec;
        classifierResults.confusionMatrix{c} = classifierResults.confusionMatrix{c} + cm;
    catch
        classifierResults.accuracy(c, fold) = 0;
    end
    
    % 3. KNN
    c = 3;
    tic;
    model = fitcknn(X_train, y_train, 'NumNeighbors', 5);
    classifierResults.trainTime(c, fold) = toc;
    tic;
    y_pred = predict(model, X_test);
    classifierResults.predTime(c, fold) = toc;
    [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred, nClasses);
    classifierResults.accuracy(c, fold) = acc;
    classifierResults.f1score(c, fold) = mean(f1);
    classifierResults.precision(c, fold, :) = prec;
    classifierResults.recall(c, fold, :) = rec;
    classifierResults.confusionMatrix{c} = classifierResults.confusionMatrix{c} + cm;
    
    % 4. SVM Linear
    c = 4;
    tic;
    svmTemplate = templateSVM('KernelFunction', 'linear', 'Standardize', false);
    model = fitcecoc(X_train, y_train, 'Learners', svmTemplate);
    classifierResults.trainTime(c, fold) = toc;
    tic;
    y_pred = predict(model, X_test);
    classifierResults.predTime(c, fold) = toc;
    [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred, nClasses);
    classifierResults.accuracy(c, fold) = acc;
    classifierResults.f1score(c, fold) = mean(f1);
    classifierResults.precision(c, fold, :) = prec;
    classifierResults.recall(c, fold, :) = rec;
    classifierResults.confusionMatrix{c} = classifierResults.confusionMatrix{c} + cm;
    
    % 5. SVM RBF
    c = 5;
    tic;
    svmTemplate = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
    model = fitcecoc(X_train, y_train, 'Learners', svmTemplate);
    classifierResults.trainTime(c, fold) = toc;
    tic;
    y_pred = predict(model, X_test);
    classifierResults.predTime(c, fold) = toc;
    [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred, nClasses);
    classifierResults.accuracy(c, fold) = acc;
    classifierResults.f1score(c, fold) = mean(f1);
    classifierResults.precision(c, fold, :) = prec;
    classifierResults.recall(c, fold, :) = rec;
    classifierResults.confusionMatrix{c} = classifierResults.confusionMatrix{c} + cm;
    
    % 6. Naive Bayes
    c = 6;
    tic;
    model = fitcnb(X_train, y_train);
    classifierResults.trainTime(c, fold) = toc;
    tic;
    y_pred = predict(model, X_test);
    classifierResults.predTime(c, fold) = toc;
    [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred, nClasses);
    classifierResults.accuracy(c, fold) = acc;
    classifierResults.f1score(c, fold) = mean(f1);
    classifierResults.precision(c, fold, :) = prec;
    classifierResults.recall(c, fold, :) = rec;
    classifierResults.confusionMatrix{c} = classifierResults.confusionMatrix{c} + cm;
    
    % 7. Random Forest
    c = 7;
    tic;
    model = TreeBagger(100, X_train, y_train, 'Method', 'classification', 'OOBPrediction', 'off');
    classifierResults.trainTime(c, fold) = toc;
    tic;
    y_pred_cell = predict(model, X_test);
    y_pred = str2double(y_pred_cell);
    classifierResults.predTime(c, fold) = toc;
    [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred, nClasses);
    classifierResults.accuracy(c, fold) = acc;
    classifierResults.f1score(c, fold) = mean(f1);
    classifierResults.precision(c, fold, :) = prec;
    classifierResults.recall(c, fold, :) = rec;
    classifierResults.confusionMatrix{c} = classifierResults.confusionMatrix{c} + cm;
    
    % 8. AdaBoost
    c = 8;
    tic;
    template = templateTree('MaxNumSplits', 20);
    model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM2', 'Learners', template, 'NumLearningCycles', 100);
    classifierResults.trainTime(c, fold) = toc;
    tic;
    y_pred = predict(model, X_test);
    classifierResults.predTime(c, fold) = toc;
    [acc, prec, rec, f1, cm] = evaluateClassifier(y_test, y_pred, nClasses);
    classifierResults.accuracy(c, fold) = acc;
    classifierResults.f1score(c, fold) = mean(f1);
    classifierResults.precision(c, fold, :) = prec;
    classifierResults.recall(c, fold, :) = rec;
    classifierResults.confusionMatrix{c} = classifierResults.confusionMatrix{c} + cm;
end

% Compute summary statistics
classifierResults.meanAccuracy = mean(classifierResults.accuracy, 2);
classifierResults.stdAccuracy = std(classifierResults.accuracy, 0, 2);
classifierResults.meanF1 = mean(classifierResults.f1score, 2);
classifierResults.stdF1 = std(classifierResults.f1score, 0, 2);
classifierResults.meanTrainTime = mean(classifierResults.trainTime, 2);
classifierResults.meanPredTime = mean(classifierResults.predTime, 2);

% Find best classifier
[~, bestClassifierIdx] = max(classifierResults.meanAccuracy);

fprintf('\n--- Classifier Comparison Results ---\n');
fprintf('%-15s | %12s | %12s | %10s\n', 'Classifier', 'Accuracy', 'F1-Score', 'Train Time');
fprintf('%s\n', repmat('-', 1, 55));
for c = 1:nClassifiers
    fprintf('%-15s | %5.2f +/- %4.2f | %5.2f +/- %4.2f | %7.3f s\n', ...
        classifierNames{c}, ...
        classifierResults.meanAccuracy(c)*100, classifierResults.stdAccuracy(c)*100, ...
        classifierResults.meanF1(c)*100, classifierResults.stdF1(c)*100, ...
        classifierResults.meanTrainTime(c));
end
fprintf('\nBest Classifier: %s (%.2f%%)\n\n', classifierNames{bestClassifierIdx}, ...
    classifierResults.meanAccuracy(bestClassifierIdx)*100);

%% ==================== SECTION 5: IMPROVEMENT TECHNIQUES ==================
fprintf('=== Section 5: Improvement Techniques ===\n');

improvementResults = struct();

%% --- 5.1: Feature Fusion ---
fprintf('\n--- 5.1: Feature Fusion ---\n');

% Combine best performing feature sets
fusedFeatures1 = [features.csp, features.fbcsp];  % CSP + FBCSP
fusedFeatures2 = [features.csp, features.wavelet];  % CSP + Wavelet
fusedFeatures3 = [features.fbcsp, features.bandPower];  % FBCSP + BandPower
fusedFeatures4 = [features.csp, features.fbcsp, features.wavelet];  % Triple fusion

fusionSets = {fusedFeatures1, fusedFeatures2, fusedFeatures3, fusedFeatures4};
fusionNames = {'CSP+FBCSP', 'CSP+Wavelet', 'FBCSP+BandPower', 'CSP+FBCSP+Wavelet'};
nFusionSets = length(fusionSets);

fusionResults = zeros(nFusionSets, nFolds);

for fs = 1:nFusionSets
    X_fused = normalizeFeatures(fusionSets{fs});
    
    for fold = 1:nFolds
        trainIdx = training(cvPartition, fold);
        testIdx = test(cvPartition, fold);
        
        X_train = X_fused(trainIdx, :);
        X_test = X_fused(testIdx, :);
        y_train = y(trainIdx);
        y_test = y(testIdx);
        
        % Use best classifier
        switch classifierNames{bestClassifierIdx}
            case 'LDA'
                model = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
            case 'SVM_RBF'
                model = fitcecoc(X_train, y_train, 'Learners', templateSVM('KernelFunction', 'rbf'));
            case 'RandomForest'
                model = TreeBagger(100, X_train, y_train, 'Method', 'classification');
            otherwise
                model = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
        end
        
        y_pred = predict(model, X_test);
        if iscell(y_pred)
            y_pred = str2double(y_pred);
        end
        
        fusionResults(fs, fold) = sum(y_pred == y_test) / length(y_test);
    end
    
    fprintf('  %s: %.2f%% (+/- %.2f%%)\n', fusionNames{fs}, ...
        mean(fusionResults(fs,:))*100, std(fusionResults(fs,:))*100);
end

[bestFusionAcc, bestFusionIdx] = max(mean(fusionResults, 2));
fprintf('  Best Fusion: %s (%.2f%%)\n', fusionNames{bestFusionIdx}, bestFusionAcc*100);

improvementResults.fusion = struct('names', {fusionNames}, 'accuracy', fusionResults, ...
    'best', fusionNames{bestFusionIdx}, 'bestAccuracy', bestFusionAcc);

%% --- 5.2: Hyperparameter Optimization ---
fprintf('\n--- 5.2: Hyperparameter Optimization ---\n');

% Grid search for SVM
fprintf('  Optimizing SVM hyperparameters...\n');

cValues = [0.01, 0.1, 1, 10, 100];
gammaValues = [0.001, 0.01, 0.1, 1];

svmGridResults = zeros(length(cValues), length(gammaValues));

% Use 3-fold CV for hyperparameter search (faster)
cvHyper = cvpartition(y, 'KFold', 3);

for ci = 1:length(cValues)
    for gi = 1:length(gammaValues)
        tempAcc = zeros(3, 1);
        for fold = 1:3
            trainIdx = training(cvHyper, fold);
            testIdx = test(cvHyper, fold);
            
            X_train = X_best(trainIdx, :);
            X_test = X_best(testIdx, :);
            y_train = y(trainIdx);
            y_test = y(testIdx);
            
            svmTemplate = templateSVM('KernelFunction', 'rbf', ...
                'BoxConstraint', cValues(ci), ...
                'KernelScale', 1/sqrt(2*gammaValues(gi)));
            model = fitcecoc(X_train, y_train, 'Learners', svmTemplate);
            y_pred = predict(model, X_test);
            tempAcc(fold) = sum(y_pred == y_test) / length(y_test);
        end
        svmGridResults(ci, gi) = mean(tempAcc);
    end
end

[maxAcc, maxIdx] = max(svmGridResults(:));
[bestCIdx, bestGIdx] = ind2sub(size(svmGridResults), maxIdx);
bestC = cValues(bestCIdx);
bestGamma = gammaValues(bestGIdx);

fprintf('  Best SVM Parameters: C=%.2f, Gamma=%.4f (Acc=%.2f%%)\n', bestC, bestGamma, maxAcc*100);

% Evaluate optimized SVM with 10-fold CV
optimizedSVMAcc = zeros(nFolds, 1);
for fold = 1:nFolds
    trainIdx = training(cvPartition, fold);
    testIdx = test(cvPartition, fold);
    
    X_train = X_best(trainIdx, :);
    X_test = X_best(testIdx, :);
    y_train = y(trainIdx);
    y_test = y(testIdx);
    
    svmTemplate = templateSVM('KernelFunction', 'rbf', ...
        'BoxConstraint', bestC, ...
        'KernelScale', 1/sqrt(2*bestGamma));
    model = fitcecoc(X_train, y_train, 'Learners', svmTemplate);
    y_pred = predict(model, X_test);
    optimizedSVMAcc(fold) = sum(y_pred == y_test) / length(y_test);
end

fprintf('  Optimized SVM (10-fold): %.2f%% (+/- %.2f%%)\n', ...
    mean(optimizedSVMAcc)*100, std(optimizedSVMAcc)*100);

% Grid search for Random Forest
fprintf('  Optimizing Random Forest hyperparameters...\n');

nTreesValues = [50, 100, 200, 300];
maxDepthValues = [5, 10, 15, 20];

rfGridResults = zeros(length(nTreesValues), length(maxDepthValues));

for ti = 1:length(nTreesValues)
    for di = 1:length(maxDepthValues)
        tempAcc = zeros(3, 1);
        for fold = 1:3
            trainIdx = training(cvHyper, fold);
            testIdx = test(cvHyper, fold);
            
            X_train = X_best(trainIdx, :);
            X_test = X_best(testIdx, :);
            y_train = y(trainIdx);
            y_test = y(testIdx);
            
            model = TreeBagger(nTreesValues(ti), X_train, y_train, ...
                'Method', 'classification', 'MaxNumSplits', maxDepthValues(di));
            y_pred_cell = predict(model, X_test);
            y_pred = str2double(y_pred_cell);
            tempAcc(fold) = sum(y_pred == y_test) / length(y_test);
        end
        rfGridResults(ti, di) = mean(tempAcc);
    end
end

[maxAccRF, maxIdxRF] = max(rfGridResults(:));
[bestTIdx, bestDIdx] = ind2sub(size(rfGridResults), maxIdxRF);
bestNTrees = nTreesValues(bestTIdx);
bestMaxDepth = maxDepthValues(bestDIdx);

fprintf('  Best RF Parameters: nTrees=%d, MaxDepth=%d (Acc=%.2f%%)\n', ...
    bestNTrees, bestMaxDepth, maxAccRF*100);

improvementResults.hyperparameterOpt = struct(...
    'svm_bestC', bestC, 'svm_bestGamma', bestGamma, 'svm_accuracy', mean(optimizedSVMAcc), ...
    'rf_bestNTrees', bestNTrees, 'rf_bestMaxDepth', bestMaxDepth, 'rf_accuracy', maxAccRF);

%% --- 5.3: Ensemble Methods ---
fprintf('\n--- 5.3: Ensemble Methods ---\n');

% Voting Ensemble
fprintf('  Building Voting Ensemble...\n');

votingResults = zeros(nFolds, 1);
stackingResults = zeros(nFolds, 1);

for fold = 1:nFolds
    trainIdx = training(cvPartition, fold);
    testIdx = test(cvPartition, fold);
    
    X_train = X_best(trainIdx, :);
    X_test = X_best(testIdx, :);
    y_train = y(trainIdx);
    y_test = y(testIdx);
    
    nTest = length(y_test);
    
    % Train base models
    model1 = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
    model2 = fitcknn(X_train, y_train, 'NumNeighbors', 5);
    model3 = fitcecoc(X_train, y_train, 'Learners', templateSVM('KernelFunction', 'rbf'));
    model4 = TreeBagger(100, X_train, y_train, 'Method', 'classification');
    model5 = fitcnb(X_train, y_train);
    
    % Get predictions
    pred1 = predict(model1, X_test);
    pred2 = predict(model2, X_test);
    pred3 = predict(model3, X_test);
    pred4_cell = predict(model4, X_test);
    pred4 = str2double(pred4_cell);
    pred5 = predict(model5, X_test);
    
    % Majority Voting
    allPreds = [pred1, pred2, pred3, pred4, pred5];
    votingPred = zeros(nTest, 1);
    for i = 1:nTest
        votingPred(i) = mode(allPreds(i, :));
    end
    
    votingResults(fold) = sum(votingPred == y_test) / nTest;
    
    % Stacking (Meta-learner)
    % Create meta-features from base model probabilities
    [~, score1] = predict(model1, X_train);
    [~, score2] = predict(model2, X_train);
    [~, score3] = predict(model3, X_train);
    [~, score4] = predict(model4, X_train);
    [~, score5] = predict(model5, X_train);
    
    if size(score4, 2) < nClasses
        score4_temp = zeros(size(X_train, 1), nClasses);
        score4_temp(:, 1:size(score4, 2)) = score4;
        score4 = score4_temp;
    end
    
    metaFeaturesTrain = [score1, score2, score3, score4, score5];
    
    % Meta-features for test
    [~, score1_test] = predict(model1, X_test);
    [~, score2_test] = predict(model2, X_test);
    [~, score3_test] = predict(model3, X_test);
    [~, score4_test] = predict(model4, X_test);
    [~, score5_test] = predict(model5, X_test);
    
    if size(score4_test, 2) < nClasses
        score4_temp = zeros(size(X_test, 1), nClasses);
        score4_temp(:, 1:size(score4_test, 2)) = score4_test;
        score4_test = score4_temp;
    end
    
    metaFeaturesTest = [score1_test, score2_test, score3_test, score4_test, score5_test];
    
    % Train meta-learner
    metaModel = fitcecoc(metaFeaturesTrain, y_train, 'Learners', templateSVM('KernelFunction', 'linear'));
    stackingPred = predict(metaModel, metaFeaturesTest);
    
    stackingResults(fold) = sum(stackingPred == y_test) / nTest;
end

fprintf('  Voting Ensemble: %.2f%% (+/- %.2f%%)\n', mean(votingResults)*100, std(votingResults)*100);
fprintf('  Stacking Ensemble: %.2f%% (+/- %.2f%%)\n', mean(stackingResults)*100, std(stackingResults)*100);

improvementResults.ensemble = struct(...
    'voting_accuracy', votingResults, 'voting_mean', mean(votingResults), ...
    'stacking_accuracy', stackingResults, 'stacking_mean', mean(stackingResults));

%% --- 5.4: Weighted Ensemble ---
fprintf('\n--- 5.4: Weighted Voting Ensemble ---\n');

% Use classifier accuracies as weights
weights = classifierResults.meanAccuracy([1, 3, 5, 7, 6]);  % LDA, KNN, SVM_RBF, RF, NB
weights = weights / sum(weights);

weightedVotingResults = zeros(nFolds, 1);

for fold = 1:nFolds
    trainIdx = training(cvPartition, fold);
    testIdx = test(cvPartition, fold);
    
    X_train = X_best(trainIdx, :);
    X_test = X_best(testIdx, :);
    y_train = y(trainIdx);
    y_test = y(testIdx);
    nTest = length(y_test);
    
    % Train base models
    model1 = fitcdiscr(X_train, y_train, 'DiscrimType', 'linear');
    model2 = fitcknn(X_train, y_train, 'NumNeighbors', 5);
    model3 = fitcecoc(X_train, y_train, 'Learners', templateSVM('KernelFunction', 'rbf'));
    model4 = TreeBagger(100, X_train, y_train, 'Method', 'classification');
    model5 = fitcnb(X_train, y_train);
    
    % Get probability scores
    [~, score1] = predict(model1, X_test);
    [~, score2] = predict(model2, X_test);
    [~, score3] = predict(model3, X_test);
    [~, score4] = predict(model4, X_test);
    [~, score5] = predict(model5, X_test);
    
    % Ensure same dimensions
    if size(score4, 2) < nClasses
        temp = zeros(nTest, nClasses);
        temp(:, 1:size(score4, 2)) = score4;
        score4 = temp;
    end
    
    % Weighted average of probabilities
    weightedScores = weights(1)*score1 + weights(2)*score2 + weights(3)*score3 + ...
                     weights(4)*score4 + weights(5)*score5;
    
    [~, weightedPred] = max(weightedScores, [], 2);
    weightedVotingResults(fold) = sum(weightedPred == y_test) / nTest;
end

fprintf('  Weighted Voting: %.2f%% (+/- %.2f%%)\n', ...
    mean(weightedVotingResults)*100, std(weightedVotingResults)*100);

improvementResults.weightedVoting = struct(...
    'accuracy', weightedVotingResults, 'mean', mean(weightedVotingResults), ...
    'weights', weights);

%% --- 5.5: Feature Selection with Ensemble ---
fprintf('\n--- 5.5: Feature Selection + Best Ensemble ---\n');

% Select top features using mutual information
nTopFeatures = min(50, size(X_best, 2));
featureScores = zeros(size(X_best, 2), 1);

for f = 1:size(X_best, 2)
    featureScores(f) = mutualInformation(X_best(:, f), y);
end

[~, topFeatureIdx] = sort(featureScores, 'descend');
topFeatureIdx = topFeatureIdx(1:nTopFeatures);

X_top = X_best(:, topFeatureIdx);

topFeatureEnsembleResults = zeros(nFolds, 1);

for fold = 1:nFolds
    trainIdx = training(cvPartition, fold);
    testIdx = test(cvPartition, fold);
    
    X_train = X_top(trainIdx, :);
    X_test = X_top(testIdx, :);
    y_train = y(trainIdx);
    y_test = y(testIdx);
    nTest = length(y_test);
    
    % Train ensemble
    model1 = fitcdiscr(X_train, y_train);
    model2 = fitcknn(X_train, y_train, 'NumNeighbors', 5);
    model3 = fitcecoc(X_train, y_train, 'Learners', templateSVM('KernelFunction', 'rbf'));
    
    pred1 = predict(model1, X_test);
    pred2 = predict(model2, X_test);
    pred3 = predict(model3, X_test);
    
    allPreds = [pred1, pred2, pred3];
    ensemblePred = zeros(nTest, 1);
    for i = 1:nTest
        ensemblePred(i) = mode(allPreds(i, :));
    end
    
    topFeatureEnsembleResults(fold) = sum(ensemblePred == y_test) / nTest;
end

fprintf('  Top %d Features + Ensemble: %.2f%% (+/- %.2f%%)\n', nTopFeatures, ...
    mean(topFeatureEnsembleResults)*100, std(topFeatureEnsembleResults)*100);

improvementResults.topFeatureEnsemble = struct(...
    'accuracy', topFeatureEnsembleResults, 'mean', mean(topFeatureEnsembleResults), ...
    'nFeatures', nTopFeatures);

%% ==================== SECTION 6: STATISTICAL TESTS =======================
fprintf('\n=== Section 6: Statistical Analysis ===\n');

% Friedman test for classifier comparison
fprintf('--- Friedman Test for Classifier Comparison ---\n');

accMatrix = classifierResults.accuracy';  % [nFolds x nClassifiers]

try
    [pFriedman, tblFriedman, statsFriedman] = friedman(accMatrix, 1, 'off');
    fprintf('  Friedman test p-value: %.6f\n', pFriedman);
    
    if pFriedman < 0.05
        fprintf('  Result: Significant difference exists between classifiers (p < 0.05)\n');
        
        % Post-hoc Nemenyi test (approximation using Wilcoxon)
        fprintf('\n--- Post-hoc Pairwise Comparisons (Wilcoxon) ---\n');
        significantPairs = {};
        
        for i = 1:nClassifiers
            for j = (i+1):nClassifiers
                [pWilcox, ~] = signrank(accMatrix(:, i), accMatrix(:, j));
                if pWilcox < 0.05/nchoosek(nClassifiers, 2)  % Bonferroni correction
                    significantPairs{end+1} = sprintf('%s vs %s (p=%.4f)', ...
                        classifierNames{i}, classifierNames{j}, pWilcox);
                end
            end
        end
        
        if ~isempty(significantPairs)
            fprintf('  Significant pairs:\n');
            for i = 1:length(significantPairs)
                fprintf('    %s\n', significantPairs{i});
            end
        else
            fprintf('  No significant pairwise differences after Bonferroni correction.\n');
        end
    else
        fprintf('  Result: No significant difference between classifiers (p >= 0.05)\n');
    end
catch
    fprintf('  Friedman test not available. Skipping statistical analysis.\n');
end

% Effect size (Cohen's d) between best and second best
[sortedAcc, sortedIdx] = sort(classifierResults.meanAccuracy, 'descend');
best1 = sortedIdx(1);
best2 = sortedIdx(2);

acc1 = classifierResults.accuracy(best1, :);
acc2 = classifierResults.accuracy(best2, :);

pooledStd = sqrt((var(acc1) + var(acc2)) / 2);
cohensD = (mean(acc1) - mean(acc2)) / pooledStd;

fprintf('\n--- Effect Size Analysis ---\n');
fprintf('  Best: %s (%.2f%%)\n', classifierNames{best1}, mean(acc1)*100);
fprintf('  Second: %s (%.2f%%)\n', classifierNames{best2}, mean(acc2)*100);
fprintf('  Cohen''s d: %.3f', cohensD);
if abs(cohensD) < 0.2
    fprintf(' (negligible effect)\n');
elseif abs(cohensD) < 0.5
    fprintf(' (small effect)\n');
elseif abs(cohensD) < 0.8
    fprintf(' (medium effect)\n');
else
    fprintf(' (large effect)\n');
end

%% ==================== SECTION 7: FINAL COMPARISON ========================
fprintf('\n=== Section 7: Final Comparison Summary ===\n');

% Collect all results
allMethods = {};
allAccuracies = [];
allStds = [];

% Base classifiers
for c = 1:nClassifiers
    allMethods{end+1} = classifierNames{c};
    allAccuracies(end+1) = classifierResults.meanAccuracy(c);
    allStds(end+1) = classifierResults.stdAccuracy(c);
end

% Improvements
allMethods{end+1} = 'Best Fusion';
allAccuracies(end+1) = bestFusionAcc;
allStds(end+1) = std(fusionResults(bestFusionIdx, :));

allMethods{end+1} = 'Optimized SVM';
allAccuracies(end+1) = mean(optimizedSVMAcc);
allStds(end+1) = std(optimizedSVMAcc);

allMethods{end+1} = 'Voting Ensemble';
allAccuracies(end+1) = mean(votingResults);
allStds(end+1) = std(votingResults);

allMethods{end+1} = 'Stacking Ensemble';
allAccuracies(end+1) = mean(stackingResults);
allStds(end+1) = std(stackingResults);

allMethods{end+1} = 'Weighted Voting';
allAccuracies(end+1) = mean(weightedVotingResults);
allStds(end+1) = std(weightedVotingResults);

allMethods{end+1} = 'TopFeat+Ensemble';
allAccuracies(end+1) = mean(topFeatureEnsembleResults);
allStds(end+1) = std(topFeatureEnsembleResults);

% Find overall best
[overallBestAcc, overallBestIdx] = max(allAccuracies);

fprintf('\n=== FINAL RANKING ===\n');
[sortedFinalAcc, sortedFinalIdx] = sort(allAccuracies, 'descend');
fprintf('%-5s | %-20s | %12s\n', 'Rank', 'Method', 'Accuracy');
fprintf('%s\n', repmat('-', 1, 45));
for i = 1:length(allMethods)
    idx = sortedFinalIdx(i);
    marker = '';
    if i == 1
        marker = ' ***';
    end
    fprintf('%4d | %-20s | %5.2f +/- %4.2f%%%s\n', i, allMethods{idx}, ...
        allAccuracies(idx)*100, allStds(idx)*100, marker);
end

fprintf('\n*** BEST METHOD: %s (%.2f%%) ***\n\n', allMethods{overallBestIdx}, overallBestAcc*100);

%% ==================== SECTION 8: VISUALIZATION ===========================
fprintf('=== Section 8: Generating Visualizations ===\n');

% Figure 1: Feature Extraction Comparison
figure('Name', 'Feature Extraction Comparison', 'Position', [50, 50, 1200, 500]);

subplot(1, 2, 1);
barData = featureCompResults.meanAccuracy * 100;
errData = featureCompResults.stdAccuracy * 100;
bar(barData);
hold on;
errorbar(1:nFeatureSets, barData, errData, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', featureSetNames, 'XTickLabelRotation', 45);
ylabel('Accuracy (%)');
title('Feature Extraction Methods Comparison (LDA)');
grid on;
ylim([0, 100]);

subplot(1, 2, 2);
boxplot(featureCompResults.accuracy' * 100, 'Labels', featureSetNames);
set(gca, 'XTickLabelRotation', 45);
ylabel('Accuracy (%)');
title('Feature Extraction Methods - Distribution');
grid on;

saveas(gcf, 'feature_comparison.png');
fprintf('  Saved: feature_comparison.png\n');

% Figure 2: Classifier Comparison
figure('Name', 'Classifier Comparison', 'Position', [50, 50, 1400, 600]);

subplot(1, 3, 1);
barData = classifierResults.meanAccuracy * 100;
errData = classifierResults.stdAccuracy * 100;
bar(barData);
hold on;
errorbar(1:nClassifiers, barData, errData, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', classifierNames, 'XTickLabelRotation', 45);
ylabel('Accuracy (%)');
title('Classifier Comparison');
grid on;
ylim([0, 100]);

subplot(1, 3, 2);
boxplot(classifierResults.accuracy' * 100, 'Labels', classifierNames);
set(gca, 'XTickLabelRotation', 45);
ylabel('Accuracy (%)');
title('Classifier Accuracy Distribution');
grid on;

subplot(1, 3, 3);
bar(classifierResults.meanTrainTime);
set(gca, 'XTickLabel', classifierNames, 'XTickLabelRotation', 45);
ylabel('Time (seconds)');
title('Training Time');
grid on;

saveas(gcf, 'classifier_comparison.png');
fprintf('  Saved: classifier_comparison.png\n');

% Figure 3: Confusion Matrices
figure('Name', 'Confusion Matrices', 'Position', [50, 50, 1600, 800]);

for c = 1:min(8, nClassifiers)
    subplot(2, 4, c);
    cm = classifierResults.confusionMatrix{c};
    cmNorm = cm ./ sum(cm, 2);  % Normalize by row
    imagesc(cmNorm);
    colorbar;
    colormap(gca, 'parula');
    caxis([0, 1]);
    title(sprintf('%s (%.1f%%)', classifierNames{c}, classifierResults.meanAccuracy(c)*100));
    xlabel('Predicted');
    ylabel('Actual');
    set(gca, 'XTick', 1:nClasses, 'YTick', 1:nClasses);
    
    % Add text
    for r = 1:nClasses
        for col = 1:nClasses
            text(col, r, sprintf('%d', cm(r, col)), 'HorizontalAlignment', 'center', ...
                'Color', 'white', 'FontWeight', 'bold', 'FontSize', 8);
        end
    end
end

saveas(gcf, 'confusion_matrices_all.png');
fprintf('  Saved: confusion_matrices_all.png\n');

% Figure 4: Improvement Methods Comparison
figure('Name', 'Improvement Methods', 'Position', [50, 50, 1200, 500]);

subplot(1, 2, 1);
improvementNames = {'Voting', 'Stacking', 'Weighted', 'TopFeat+Ens', 'Best Fusion'};
improvementMeans = [mean(votingResults), mean(stackingResults), mean(weightedVotingResults), ...
                    mean(topFeatureEnsembleResults), bestFusionAcc] * 100;
improvementStds = [std(votingResults), std(stackingResults), std(weightedVotingResults), ...
                   std(topFeatureEnsembleResults), std(fusionResults(bestFusionIdx, :))] * 100;

bar(improvementMeans);
hold on;
errorbar(1:5, improvementMeans, improvementStds, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
yline(classifierResults.meanAccuracy(bestClassifierIdx)*100, 'r--', 'Best Single', 'LineWidth', 2);
hold off;
set(gca, 'XTickLabel', improvementNames, 'XTickLabelRotation', 30);
ylabel('Accuracy (%)');
title('Improvement Methods');
grid on;
legend('Accuracy', '', 'Best Single Classifier', 'Location', 'best');

subplot(1, 2, 2);
allImprovementData = [votingResults, stackingResults, weightedVotingResults, ...
                      topFeatureEnsembleResults, fusionResults(bestFusionIdx, :)'] * 100;
boxplot(allImprovementData, 'Labels', improvementNames);
set(gca, 'XTickLabelRotation', 30);
ylabel('Accuracy (%)');
title('Improvement Methods Distribution');
grid on;

saveas(gcf, 'improvement_methods.png');
fprintf('  Saved: improvement_methods.png\n');

% Figure 5: Final Ranking
figure('Name', 'Final Ranking', 'Position', [50, 50, 1000, 600]);

[sortedAcc, sortedIdx] = sort(allAccuracies, 'descend');
sortedStds = allStds(sortedIdx);
sortedNames = allMethods(sortedIdx);

barh(flip(sortedAcc * 100));
hold on;
errorbar(flip(sortedAcc * 100), 1:length(sortedAcc), flip(sortedStds * 100), 'horizontal', ...
    'k', 'LineStyle', 'none', 'LineWidth', 1.5);
hold off;
set(gca, 'YTickLabel', flip(sortedNames));
xlabel('Accuracy (%)');
title('Final Method Ranking');
grid on;
xlim([0, 100]);

% Highlight best
for i = 1:length(sortedAcc)
    color = [0.3, 0.6, 0.9];  % Blue
    if i == length(sortedAcc)  % Best (flipped, so it's at the top)
        color = [0.2, 0.8, 0.2];  % Green
    elseif sortedAcc(length(sortedAcc) - i + 1) < 0.5
        color = [0.9, 0.3, 0.3];  % Red for low accuracy
    end
end

saveas(gcf, 'final_ranking.png');
fprintf('  Saved: final_ranking.png\n');

% Figure 6: Hyperparameter Optimization Results
figure('Name', 'Hyperparameter Optimization', 'Position', [50, 50, 1000, 400]);

subplot(1, 2, 1);
imagesc(svmGridResults * 100);
colorbar;
colormap('hot');
xlabel('Gamma');
ylabel('C');
set(gca, 'XTickLabel', gammaValues, 'YTickLabel', cValues);
title('SVM Grid Search Results');

subplot(1, 2, 2);
imagesc(rfGridResults * 100);
colorbar;
colormap('hot');
xlabel('Max Depth');
ylabel('Num Trees');
set(gca, 'XTickLabel', maxDepthValues, 'YTickLabel', nTreesValues);
title('Random Forest Grid Search Results');

saveas(gcf, 'hyperparameter_optimization.png');
fprintf('  Saved: hyperparameter_optimization.png\n');

%% ==================== SECTION 9: SAVE RESULTS ============================
fprintf('\n=== Section 9: Saving Results ===\n');

evaluationResults = struct();
evaluationResults.featureComparison = featureCompResults;
evaluationResults.classifierComparison = classifierResults;
evaluationResults.improvements = improvementResults;
evaluationResults.finalRanking = struct('methods', {allMethods}, 'accuracies', allAccuracies, ...
    'stds', allStds, 'bestMethod', allMethods{overallBestIdx}, 'bestAccuracy', overallBestAcc);
evaluationResults.statisticalTests = struct('friedmanP', pFriedman, 'cohensD', cohensD);
evaluationResults.metadata = struct('nFolds', nFolds, 'nSamples', nSamples, ...
    'nClasses', nClasses, 'date', datestr(now));

save('evaluation_results.mat', 'evaluationResults', '-v7.3');
fprintf('  Saved: evaluation_results.mat\n');

%% ==================== SECTION 10: GENERATE FINAL REPORT ==================
fprintf('\n=== Section 10: Generating Final Report ===\n');

REPORT_FILE = 'final_evaluation_report.txt';
fid = fopen(REPORT_FILE, 'w');

fprintf(fid, '================================================================================\n');
fprintf(fid, '           EEG MOTOR IMAGERY CLASSIFICATION - FINAL EVALUATION REPORT\n');
fprintf(fid, '================================================================================\n\n');

fprintf(fid, 'Report Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'MATLAB Version: %s\n\n', version);

fprintf(fid, '================================================================================\n');
fprintf(fid, '                              1. DATA SUMMARY\n');
fprintf(fid, '================================================================================\n\n');
fprintf(fid, 'Total Samples: %d\n', nSamples);
fprintf(fid, 'Number of Classes: %d\n', nClasses);
fprintf(fid, 'Cross-Validation: %d-fold stratified\n', nFolds);
fprintf(fid, 'Classes: ');
fprintf(fid, '%s ', uniqueLabels{:});
fprintf(fid, '\n\n');

fprintf(fid, 'Class Distribution:\n');
for c = 1:nClasses
    count = sum(y == c);
    fprintf(fid, '  %s: %d samples (%.1f%%)\n', uniqueLabels{c}, count, 100*count/nSamples);
end

fprintf(fid, '\n================================================================================\n');
fprintf(fid, '                    2. FEATURE EXTRACTION COMPARISON\n');
fprintf(fid, '================================================================================\n\n');

fprintf(fid, '%-25s | %15s | %15s\n', 'Feature Set', 'Accuracy', 'F1-Score');
fprintf(fid, '%s\n', repmat('-', 1, 60));
for fs = 1:nFeatureSets
    fprintf(fid, '%-25s | %6.2f +/- %5.2f | %6.2f +/- %5.2f\n', ...
        featureSetNames{fs}, ...
        featureCompResults.meanAccuracy(fs)*100, featureCompResults.stdAccuracy(fs)*100, ...
        featureCompResults.meanF1(fs)*100, featureCompResults.stdF1(fs)*100);
end
fprintf(fid, '\nBest Feature Set: %s (%.2f%%)\n', featureSetNames{bestFeatureIdx}, ...
    featureCompResults.meanAccuracy(bestFeatureIdx)*100);

fprintf(fid, '\n================================================================================\n');
fprintf(fid, '                       3. CLASSIFIER COMPARISON\n');
fprintf(fid, '================================================================================\n\n');

fprintf(fid, '%-15s | %15s | %15s | %12s\n', 'Classifier', 'Accuracy', 'F1-Score', 'Train Time');
fprintf(fid, '%s\n', repmat('-', 1, 65));
for c = 1:nClassifiers
    fprintf(fid, '%-15s | %6.2f +/- %5.2f | %6.2f +/- %5.2f | %9.3f s\n', ...
        classifierNames{c}, ...
        classifierResults.meanAccuracy(c)*100, classifierResults.stdAccuracy(c)*100, ...
        classifierResults.meanF1(c)*100, classifierResults.stdF1(c)*100, ...
        classifierResults.meanTrainTime(c));
end
fprintf(fid, '\nBest Classifier: %s (%.2f%%)\n', classifierNames{bestClassifierIdx}, ...
    classifierResults.meanAccuracy(bestClassifierIdx)*100);

fprintf(fid, '\n================================================================================\n');
fprintf(fid, '                       4. IMPROVEMENT TECHNIQUES\n');
fprintf(fid, '================================================================================\n\n');

fprintf(fid, '--- 4.1 Feature Fusion ---\n');
for fs = 1:nFusionSets
    fprintf(fid, '  %s: %.2f%% (+/- %.2f%%)\n', fusionNames{fs}, ...
        mean(fusionResults(fs,:))*100, std(fusionResults(fs,:))*100);
end
fprintf(fid, '  Best: %s\n\n', fusionNames{bestFusionIdx});

fprintf(fid, '--- 4.2 Hyperparameter Optimization ---\n');
fprintf(fid, '  SVM: C=%.2f, Gamma=%.4f -> %.2f%%\n', bestC, bestGamma, mean(optimizedSVMAcc)*100);
fprintf(fid, '  Random Forest: nTrees=%d, MaxDepth=%d -> %.2f%%\n\n', ...
    bestNTrees, bestMaxDepth, maxAccRF*100);

fprintf(fid, '--- 4.3 Ensemble Methods ---\n');
fprintf(fid, '  Majority Voting: %.2f%% (+/- %.2f%%)\n', mean(votingResults)*100, std(votingResults)*100);
fprintf(fid, '  Stacking: %.2f%% (+/- %.2f%%)\n', mean(stackingResults)*100, std(stackingResults)*100);
fprintf(fid, '  Weighted Voting: %.2f%% (+/- %.2f%%)\n', mean(weightedVotingResults)*100, std(weightedVotingResults)*100);
fprintf(fid, '  Top Features + Ensemble: %.2f%% (+/- %.2f%%)\n\n', ...
    mean(topFeatureEnsembleResults)*100, std(topFeatureEnsembleResults)*100);

fprintf(fid, '\n================================================================================\n');
fprintf(fid, '                       5. STATISTICAL ANALYSIS\n');
fprintf(fid, '================================================================================\n\n');

fprintf(fid, 'Friedman Test p-value: %.6f\n', pFriedman);
if pFriedman < 0.05
    fprintf(fid, 'Result: Significant difference between classifiers (p < 0.05)\n');
else
    fprintf(fid, 'Result: No significant difference (p >= 0.05)\n');
end
fprintf(fid, '\nEffect Size (Best vs Second Best):\n');
fprintf(fid, '  Cohen''s d: %.3f ', cohensD);
if abs(cohensD) < 0.2
    fprintf(fid, '(negligible)\n');
elseif abs(cohensD) < 0.5
    fprintf(fid, '(small)\n');
elseif abs(cohensD) < 0.8
    fprintf(fid, '(medium)\n');
else
    fprintf(fid, '(large)\n');
end

fprintf(fid, '\n================================================================================\n');
fprintf(fid, '                          6. FINAL RANKING\n');
fprintf(fid, '================================================================================\n\n');

fprintf(fid, '%5s | %-25s | %15s\n', 'Rank', 'Method', 'Accuracy');
fprintf(fid, '%s\n', repmat('-', 1, 50));
for i = 1:length(allMethods)
    idx = sortedFinalIdx(i);
    marker = '';
    if i == 1
        marker = ' <-- BEST';
    end
    fprintf(fid, '%4d | %-25s | %6.2f +/- %4.2f%%%s\n', i, allMethods{idx}, ...
        allAccuracies(idx)*100, allStds(idx)*100, marker);
end

fprintf(fid, '\n================================================================================\n');
fprintf(fid, '                          7. CONCLUSIONS\n');
fprintf(fid, '================================================================================\n\n');

fprintf(fid, 'BEST OVERALL METHOD: %s\n', allMethods{overallBestIdx});
fprintf(fid, 'BEST ACCURACY: %.2f%% (+/- %.2f%%)\n\n', overallBestAcc*100, allStds(overallBestIdx)*100);

fprintf(fid, 'Key Findings:\n');
fprintf(fid, '  1. Best feature extraction: %s\n', featureSetNames{bestFeatureIdx});
fprintf(fid, '  2. Best single classifier: %s\n', classifierNames{bestClassifierIdx});
fprintf(fid, '  3. Improvement from ensemble: %.2f%% -> %.2f%%\n', ...
    classifierResults.meanAccuracy(bestClassifierIdx)*100, overallBestAcc*100);

fprintf(fid, '\n================================================================================\n');
fprintf(fid, '                          8. RECOMMENDATIONS\n');
fprintf(fid, '================================================================================\n\n');

fprintf(fid, 'For Production Use:\n');
fprintf(fid, '  - Use %s for best accuracy\n', allMethods{overallBestIdx});
if overallBestAcc > 0.8
    fprintf(fid, '  - Current accuracy (%.1f%%) is suitable for BCI applications\n', overallBestAcc*100);
else
    fprintf(fid, '  - Consider additional improvements for BCI applications\n');
end

fprintf(fid, '\nFuture Improvements:\n');
fprintf(fid, '  - Try attention mechanisms or Transformers\n');
fprintf(fid, '  - Explore domain adaptation techniques\n');
fprintf(fid, '  - Investigate subject-specific models\n');
fprintf(fid, '  - Apply data augmentation\n');

fprintf(fid, '\n================================================================================\n');
fprintf(fid, '                          9. OUTPUT FILES\n');
fprintf(fid, '================================================================================\n\n');

fprintf(fid, '1. evaluation_results.mat - Complete evaluation data\n');
fprintf(fid, '2. feature_comparison.png - Feature method comparison\n');
fprintf(fid, '3. classifier_comparison.png - Classifier comparison\n');
fprintf(fid, '4. confusion_matrices_all.png - All confusion matrices\n');
fprintf(fid, '5. improvement_methods.png - Improvement techniques\n');
fprintf(fid, '6. final_ranking.png - Final ranking visualization\n');
fprintf(fid, '7. hyperparameter_optimization.png - Grid search results\n');
fprintf(fid, '8. %s - This report\n', REPORT_FILE);

fprintf(fid, '\n================================================================================\n');
fprintf(fid, '                           END OF REPORT\n');
fprintf(fid, '================================================================================\n');

fclose(fid);
fprintf('  Saved: %s\n', REPORT_FILE);

%% ==================== COMPLETION =========================================
totalTime = toc;

fprintf('\n');
fprintf('============================================================\n');
fprintf('  EVALUATION COMPLETE\n');
fprintf('============================================================\n');
fprintf('Total processing time: %.2f seconds (%.2f minutes)\n', totalTime, totalTime/60);
fprintf('\nBEST METHOD: %s\n', allMethods{overallBestIdx});
fprintf('BEST ACCURACY: %.2f%% (+/- %.2f%%)\n', overallBestAcc*100, allStds(overallBestIdx)*100);
fprintf('\nOutput files saved.\n');
fprintf('============================================================\n');

%% ==================== HELPER FUNCTIONS ===================================

function X_norm = normalizeFeatures(X)
    % Z-score normalization
    mu = mean(X, 1);
    sigma = std(X, 0, 1);
    sigma(sigma < eps) = 1;
    X_norm = (X - mu) ./ sigma;
    
    % Handle NaN/Inf
    X_norm(isnan(X_norm) | isinf(X_norm)) = 0;
end

function [accuracy, precision, recall, f1, confMat] = evaluateClassifier(yTrue, yPred, nClasses)
    yTrue = yTrue(:);
    yPred = yPred(:);
    
    accuracy = sum(yTrue == yPred) / length(yTrue);
    
    confMat = confusionmat(yTrue, yPred);
    
    if size(confMat, 1) < nClasses
        confMat(nClasses, nClasses) = 0;
    end
    
    precision = zeros(1, nClasses);
    recall = zeros(1, nClasses);
    f1 = zeros(1, nClasses);
    
    for c = 1:nClasses
        tp = confMat(c, c);
        fp = sum(confMat(:, c)) - tp;
        fn = sum(confMat(c, :)) - tp;
        
        if (tp + fp) > 0
            precision(c) = tp / (tp + fp);
        end
        
        if (tp + fn) > 0
            recall(c) = tp / (tp + fn);
        end
        
        if (precision(c) + recall(c)) > 0
            f1(c) = 2 * precision(c) * recall(c) / (precision(c) + recall(c));
        end
    end
end

function mi = mutualInformation(X, Y)
    % Discretize continuous feature
    nBins = 10;
    [~, edges] = histcounts(X, nBins);
    X_disc = discretize(X, edges);
    X_disc(isnan(X_disc)) = 1;
    
    % Compute mutual information
    uniqueX = unique(X_disc);
    uniqueY = unique(Y);
    
    pXY = zeros(length(uniqueX), length(uniqueY));
    for i = 1:length(uniqueX)
        for j = 1:length(uniqueY)
            pXY(i, j) = sum(X_disc == uniqueX(i) & Y == uniqueY(j)) / length(Y);
        end
    end
    
    pX = sum(pXY, 2);
    pY = sum(pXY, 1);
    
    mi = 0;
    for i = 1:length(uniqueX)
        for j = 1:length(uniqueY)
            if pXY(i, j) > 0
                mi = mi + pXY(i, j) * log2(pXY(i, j) / (pX(i) * pY(j) + eps));
            end
        end
    end
    
    mi = max(0, mi);  % Ensure non-negative
end