%% ==========================================================================
%  EEG FEATURE EXTRACTION PIPELINE FOR MOTOR IMAGERY CLASSIFICATION
%  ==========================================================================
%  Description:
%    This script extracts multiple feature types from preprocessed EEG epochs
%    for motor imagery classification. Features include:
%    1. Band Power Features (Delta, Theta, Alpha, Beta, Gamma)
%    2. Common Spatial Patterns (CSP)
%    3. Filter Bank CSP (FBCSP)
%    4. Wavelet Transform Features
%    5. Time-Domain Statistical Features
%    6. Hjorth Parameters
%  
%  Input:
%    - fourclass_eeg_processed.mat: Preprocessed epochs from Part 1
%  
%  Output:
%    - fourclass_features.mat: Feature matrices ready for classification
%    - fourclass_features_simple.mat: Simplified feature matrix (X, y)
%    - feature_extraction_report.txt: Processing summary
%    - feature_extraction_results_improved.png: Visualization figure
%  ==========================================================================

%% ==================== INITIALIZATION =====================================
% Clear workspace, close all figures, and clear command window
clear; close all; clc;

% Start timer to measure total processing time
tic;

% Display header message
fprintf('============================================================\n');
fprintf('     EEG FEATURE EXTRACTION PIPELINE - MOTOR IMAGERY\n');
fprintf('============================================================\n\n');

%% ==================== SECTION 1: LOAD PREPROCESSED DATA ==================
% This section loads the preprocessed EEG data from the previous pipeline stage
% and extracts essential information including epochs, labels, and sampling rate

fprintf('=== Section 1: Loading Preprocessed Data ===\n');

% Define input file path
INPUT_FILE = 'fourclass_eeg_processed.mat';

% Check if the preprocessed data file exists
if ~isfile(INPUT_FILE)
    error('Preprocessed data file not found: %s\nPlease run the preprocessing script first.', INPUT_FILE);
end

% Load the preprocessed data structure
load(INPUT_FILE, 'processedData');

% Extract data components from the loaded structure
epochs = processedData.epochs;                      % EEG epochs: [channels x samples x trials]
labels = processedData.labels;                      % Class labels: cell array of strings
srate = processedData.srate;                        % Sampling rate in Hz
timeVector = processedData.epochInfo.timeVector;   % Time vector for epochs

% Get dimensions of the epoch data
[nChannels, nSamples, nTrials] = size(epochs);

% Convert string labels to numeric format for CSP computation
% This is necessary because CSP algorithm requires numeric class indicators
uniqueLabels = unique(labels);                      % Get unique class names
nClasses = length(uniqueLabels);                    % Number of classes
numericLabels = zeros(nTrials, 1);                  % Initialize numeric label array

% Assign numeric values to each class
for i = 1:nClasses
    idx = strcmp(labels, uniqueLabels{i});          % Find trials belonging to class i
    numericLabels(idx) = i;                         % Assign numeric label
end

% Display loaded data summary
fprintf('Data loaded successfully.\n');
fprintf('Epochs: %d trials, %d channels, %d samples\n', nTrials, nChannels, nSamples);
fprintf('Sampling rate: %.1f Hz\n', srate);
fprintf('Number of classes: %d\n', nClasses);
fprintf('Classes: ');
fprintf('%s ', uniqueLabels{:});
fprintf('\n\n');

%% ==================== SECTION 2: BAND POWER FEATURES =====================
% Extract power spectral density in standard EEG frequency bands using Welch's method
% Frequency Bands:
%   - Delta: 1-4 Hz   (deep sleep, unconscious states)
%   - Theta: 4-8 Hz   (drowsiness, light sleep, meditation)
%   - Alpha: 8-13 Hz  (relaxed wakefulness, eyes closed)
%   - Beta:  13-30 Hz (active thinking, focus, alertness)
%   - Gamma: 30-40 Hz (higher cognitive functions, perception)

fprintf('=== Section 2: Extracting Band Power Features ===\n');

% Define frequency bands as a structure
bandDefs = struct();
bandDefs.delta = [1, 4];      % Delta band range
bandDefs.theta = [4, 8];      % Theta band range
bandDefs.alpha = [8, 13];     % Alpha band range
bandDefs.beta  = [13, 30];    % Beta band range
bandDefs.gamma = [30, 40];    % Gamma band range

% Get band names and count
bandNames = fieldnames(bandDefs);
nBands = length(bandNames);

% Welch's method parameters for PSD estimation
winLength = round(srate * 0.5);     % Window length: 500ms (in samples)
overlap = round(winLength * 0.5);   % Overlap: 50% of window length
nfft = 2^nextpow2(winLength);       % FFT points: next power of 2 for efficiency

% Pre-allocate feature matrix
% Dimensions: [number of trials x (number of channels * number of bands)]
bandPowerFeatures = zeros(nTrials, nChannels * nBands);

fprintf('Computing band power for %d bands across %d channels...\n', nBands, nChannels);

% Loop through each trial to extract band power features
for trial = 1:nTrials
    epochData = epochs(:, :, trial);    % Extract current trial data
    featureIdx = 0;                      % Reset feature index for this trial
    
    % Loop through each channel
    for ch = 1:nChannels
        signal = epochData(ch, :);      % Get signal from current channel
        
        % Compute Power Spectral Density using Welch's method
        % pwelch returns PSD estimate and corresponding frequency vector
        [pxx, f] = pwelch(signal, hamming(winLength), overlap, nfft, srate);
        
        % Extract power for each frequency band
        for b = 1:nBands
            bandName = bandNames{b};
            freqRange = bandDefs.(bandName);    % Get frequency range for this band
            
            % Find frequency indices within the band range
            freqIdx = (f >= freqRange(1)) & (f <= freqRange(2));
            
            % Compute average band power and convert to log scale
            % Log transformation helps normalize the distribution
            bandPower = mean(pxx(freqIdx));
            
            featureIdx = featureIdx + 1;
            bandPowerFeatures(trial, featureIdx) = log10(bandPower + eps);  % eps prevents log(0)
        end
    end
    
    % Display progress every 100 trials
    if mod(trial, 100) == 0 || trial == nTrials
        fprintf('  Progress: %d/%d trials (%.1f%%)\n', trial, nTrials, 100*trial/nTrials);
    end
end

% Create descriptive feature names for band power features
bandPowerNames = cell(1, nChannels * nBands);
idx = 0;
for ch = 1:nChannels
    for b = 1:nBands
        idx = idx + 1;
        bandPowerNames{idx} = sprintf('Ch%d_%s', ch, bandNames{b});
    end
end

% Display extraction summary
fprintf('Band power features extracted: %d features per trial\n', size(bandPowerFeatures, 2));
fprintf('Feature matrix size: [%d x %d]\n\n', size(bandPowerFeatures));

%% ==================== SECTION 3: RELATIVE BAND POWER =====================
% Compute relative band power as percentage of total power
% This normalizes for overall amplitude differences between trials
% Relative power = Band power / Total power (1-40 Hz)

fprintf('=== Section 3: Computing Relative Band Power ===\n');

% Pre-allocate relative band power matrix
relativeBandPower = zeros(nTrials, nChannels * nBands);

% Loop through each trial
for trial = 1:nTrials
    epochData = epochs(:, :, trial);
    featureIdx = 0;
    
    % Loop through each channel
    for ch = 1:nChannels
        signal = epochData(ch, :);
        
        % Compute PSD using Welch's method
        [pxx, f] = pwelch(signal, hamming(winLength), overlap, nfft, srate);
        
        % Calculate total power in the analysis range (1-40 Hz)
        totalIdx = (f >= 1) & (f <= 40);
        totalPower = sum(pxx(totalIdx));
        
        % Calculate relative power for each band
        for b = 1:nBands
            bandName = bandNames{b};
            freqRange = bandDefs.(bandName);
            freqIdx = (f >= freqRange(1)) & (f <= freqRange(2));
            
            % Sum power in this band and divide by total power
            bandPower = sum(pxx(freqIdx));
            relPower = bandPower / (totalPower + eps);  % eps prevents division by zero
            
            featureIdx = featureIdx + 1;
            relativeBandPower(trial, featureIdx) = relPower;
        end
    end
end

fprintf('Relative band power features extracted: %d features\n\n', size(relativeBandPower, 2));

%% ==================== SECTION 4: COMMON SPATIAL PATTERNS (CSP) ===========
% CSP finds spatial filters that maximize variance ratio between two classes
% For multi-class problems, we use One-vs-Rest (OVR) approach:
%   - For each class, compute CSP filters: current class vs. all other classes
%   - Features are log-variance of spatially filtered signals

fprintf('=== Section 4: Extracting CSP Features ===\n');

% CSP Parameters
nCSPFilters = 3;  % Number of CSP filters per side (total filters = 2 * nCSPFilters)

% Initialize feature storage
cspFeatures = [];
cspFilterNames = {};

% Apply One-vs-Rest CSP for each class
for classIdx = 1:nClasses
    fprintf('  Computing CSP for Class %d vs Rest...\n', classIdx);
    
    % Create binary labels: current class (1) vs all other classes (0)
    binaryLabels = (numericLabels == classIdx);
    
    % Separate epochs by class
    class1Epochs = epochs(:, :, binaryLabels);      % Current class epochs
    class2Epochs = epochs(:, :, ~binaryLabels);     % All other class epochs
    
    % Compute CSP filters using helper function
    [W, lambda] = computeCSP(class1Epochs, class2Epochs, nCSPFilters);
    
    % Extract CSP features for all trials
    nFilters = size(W, 2);
    classCSPFeatures = zeros(nTrials, nFilters);
    
    for trial = 1:nTrials
        epochData = epochs(:, :, trial);
        
        % Apply CSP spatial filters to the epoch
        Z = W' * epochData;  % Spatially filtered signal: [nFilters x nSamples]
        
        % Compute log-variance features (normalized by total variance)
        varZ = var(Z, 0, 2);                        % Variance of each filtered signal
        classCSPFeatures(trial, :) = log(varZ / sum(varZ));  % Normalized log-variance
    end
    
    % Append features to the main feature matrix
    cspFeatures = [cspFeatures, classCSPFeatures];
    
    % Create descriptive feature names
    for f = 1:nFilters
        cspFilterNames{end+1} = sprintf('CSP_C%d_F%d', classIdx, f);
    end
end

fprintf('CSP features extracted: %d features per trial\n', size(cspFeatures, 2));
fprintf('Feature matrix size: [%d x %d]\n\n', size(cspFeatures));

%% ==================== SECTION 5: FILTER BANK CSP (FBCSP) =================
% FBCSP extends CSP by applying it to multiple frequency sub-bands
% This captures frequency-specific spatial patterns that may be discriminative
% Process: Bandpass filter -> CSP -> Log-variance features

fprintf('=== Section 5: Extracting Filter Bank CSP Features ===\n');

% Define filter bank with overlapping frequency sub-bands
% Each row defines [low_frequency, high_frequency] for a sub-band
filterBank = [
    4, 8;       % Sub-band 1: Theta range
    8, 12;      % Sub-band 2: Low Alpha
    12, 16;     % Sub-band 3: High Alpha / Low Beta
    16, 20;     % Sub-band 4: Beta 1
    20, 24;     % Sub-band 5: Beta 2
    24, 28;     % Sub-band 6: Beta 3
    28, 32;     % Sub-band 7: Beta 4
    32, 36;     % Sub-band 8: Low Gamma
    36, 40      % Sub-band 9: High Gamma
];

nSubBands = size(filterBank, 1);    % Number of frequency sub-bands
nFBCSPFilters = 2;                   % Filters per side per band (smaller than regular CSP)

% Initialize feature storage
fbcspFeatures = [];
fbcspNames = {};

fprintf('Processing %d frequency sub-bands...\n', nSubBands);

% Process each frequency sub-band
for band = 1:nSubBands
    lowFreq = filterBank(band, 1);      % Lower cutoff frequency
    highFreq = filterBank(band, 2);     % Upper cutoff frequency
    
    fprintf('  Band %d: %d-%d Hz\n', band, lowFreq, highFreq);
    
    % Design 4th-order Butterworth bandpass filter
    nyquist = srate / 2;                                    % Nyquist frequency
    [bFilt, aFilt] = butter(4, [lowFreq highFreq] / nyquist, 'bandpass');
    
    % Apply bandpass filter to all epochs
    filteredEpochs = zeros(size(epochs));
    for trial = 1:nTrials
        for ch = 1:nChannels
            % filtfilt applies zero-phase filtering (forward and backward)
            filteredEpochs(ch, :, trial) = filtfilt(bFilt, aFilt, epochs(ch, :, trial));
        end
    end
    
    % Apply CSP to the filtered data (One-vs-Rest for multi-class)
    for classIdx = 1:nClasses
        binaryLabels = (numericLabels == classIdx);
        
        % Separate filtered epochs by class
        class1Epochs = filteredEpochs(:, :, binaryLabels);
        class2Epochs = filteredEpochs(:, :, ~binaryLabels);
        
        % Compute CSP filters for this sub-band
        [W, ~] = computeCSP(class1Epochs, class2Epochs, nFBCSPFilters);
        
        % Extract log-variance features
        nFilters = size(W, 2);
        bandCSPFeatures = zeros(nTrials, nFilters);
        
        for trial = 1:nTrials
            Z = W' * filteredEpochs(:, :, trial);   % Apply spatial filters
            varZ = var(Z, 0, 2);                     % Compute variance
            bandCSPFeatures(trial, :) = log(varZ / sum(varZ));  % Normalized log-variance
        end
        
        % Append to feature matrix
        fbcspFeatures = [fbcspFeatures, bandCSPFeatures];
        
        % Create feature names: FBCSP_B[band]_C[class]_F[filter]
        for f = 1:nFilters
            fbcspNames{end+1} = sprintf('FBCSP_B%d_C%d_F%d', band, classIdx, f);
        end
    end
end

fprintf('FBCSP features extracted: %d features per trial\n', size(fbcspFeatures, 2));
fprintf('Feature matrix size: [%d x %d]\n\n', size(fbcspFeatures));

%% ==================== SECTION 6: WAVELET FEATURES ========================
% Extract time-frequency features using Discrete Wavelet Transform (DWT)
% DWT decomposes signal into approximation (low-freq) and detail (high-freq) coefficients
% Features extracted: Energy, Mean, Standard Deviation, Entropy at each level

fprintf('=== Section 6: Extracting Wavelet Features ===\n');

% Wavelet decomposition parameters
waveletName = 'db4';    % Daubechies 4 wavelet (good for EEG analysis)
nLevels = 5;            % Number of decomposition levels

% Calculate number of wavelet features per channel
% Features: 4 statistics (energy, mean, std, entropy) for each level + approximation
nWaveletFeatures = (nLevels + 1) * 4;  % (nLevels detail + 1 approximation) * 4 stats
totalWaveletFeatures = nChannels * nWaveletFeatures;

% Pre-allocate wavelet feature matrix
waveletFeatures = zeros(nTrials, totalWaveletFeatures);
waveletNames = {};

fprintf('Decomposing signals with %s wavelet (%d levels)...\n', waveletName, nLevels);

% Process each trial
for trial = 1:nTrials
    epochData = epochs(:, :, trial);
    featureIdx = 0;
    
    % Process each channel
    for ch = 1:nChannels
        signal = epochData(ch, :);
        
        % Perform multilevel DWT decomposition
        % C: wavelet coefficients, L: bookkeeping vector for coefficient lengths
        [C, L] = wavedec(signal, nLevels, waveletName);
        
        % Extract approximation coefficients at the deepest level
        A = appcoef(C, L, waveletName, nLevels);
        
        % Compute features from approximation coefficients
        featureIdx = featureIdx + 1;
        waveletFeatures(trial, featureIdx) = log(sum(A.^2) + eps);  % Log energy
        featureIdx = featureIdx + 1;
        waveletFeatures(trial, featureIdx) = mean(abs(A));           % Mean absolute value
        featureIdx = featureIdx + 1;
        waveletFeatures(trial, featureIdx) = std(A);                 % Standard deviation
        featureIdx = featureIdx + 1;
        waveletFeatures(trial, featureIdx) = wentropy(A, 'shannon'); % Shannon entropy
        
        % Extract features from detail coefficients at each level
        for level = 1:nLevels
            D = detcoef(C, L, level);   % Detail coefficients at current level
            
            % Compute 4 statistical features for this level
            featureIdx = featureIdx + 1;
            waveletFeatures(trial, featureIdx) = log(sum(D.^2) + eps);  % Log energy
            featureIdx = featureIdx + 1;
            waveletFeatures(trial, featureIdx) = mean(abs(D));           % Mean absolute value
            featureIdx = featureIdx + 1;
            waveletFeatures(trial, featureIdx) = std(D);                 % Standard deviation
            featureIdx = featureIdx + 1;
            waveletFeatures(trial, featureIdx) = wentropy(D, 'shannon'); % Shannon entropy
        end
    end
    
    % Display progress every 100 trials
    if mod(trial, 100) == 0 || trial == nTrials
        fprintf('  Progress: %d/%d trials (%.1f%%)\n', trial, nTrials, 100*trial/nTrials);
    end
end

% Create descriptive wavelet feature names
idx = 0;
for ch = 1:nChannels
    % Approximation coefficient features
    idx = idx + 1; waveletNames{idx} = sprintf('Ch%d_A%d_energy', ch, nLevels);
    idx = idx + 1; waveletNames{idx} = sprintf('Ch%d_A%d_mean', ch, nLevels);
    idx = idx + 1; waveletNames{idx} = sprintf('Ch%d_A%d_std', ch, nLevels);
    idx = idx + 1; waveletNames{idx} = sprintf('Ch%d_A%d_entropy', ch, nLevels);
    
    % Detail coefficient features for each level
    for level = 1:nLevels
        idx = idx + 1; waveletNames{idx} = sprintf('Ch%d_D%d_energy', ch, level);
        idx = idx + 1; waveletNames{idx} = sprintf('Ch%d_D%d_mean', ch, level);
        idx = idx + 1; waveletNames{idx} = sprintf('Ch%d_D%d_std', ch, level);
        idx = idx + 1; waveletNames{idx} = sprintf('Ch%d_D%d_entropy', ch, level);
    end
end

fprintf('Wavelet features extracted: %d features per trial\n', size(waveletFeatures, 2));
fprintf('Feature matrix size: [%d x %d]\n\n', size(waveletFeatures));

%% ==================== SECTION 7: TIME-DOMAIN FEATURES ====================
% Extract statistical features directly from the time-domain signal
% These features capture amplitude characteristics and signal dynamics
% Features: mean, variance, skewness, kurtosis, peak-to-peak, RMS, zero-crossings, line length

fprintf('=== Section 7: Extracting Time-Domain Features ===\n');

% Number of time-domain features per channel
nTimeFeatures = 8;
timeDomainFeatures = zeros(nTrials, nChannels * nTimeFeatures);
timeFeatureNames = {};

% Process each trial
for trial = 1:nTrials
    epochData = epochs(:, :, trial);
    featureIdx = 0;
    
    % Process each channel
    for ch = 1:nChannels
        signal = epochData(ch, :);
        
        % Feature 1: Mean - average signal amplitude
        featureIdx = featureIdx + 1;
        timeDomainFeatures(trial, featureIdx) = mean(signal);
        
        % Feature 2: Variance - signal power/variability
        featureIdx = featureIdx + 1;
        timeDomainFeatures(trial, featureIdx) = var(signal);
        
        % Feature 3: Skewness - asymmetry of amplitude distribution
        featureIdx = featureIdx + 1;
        timeDomainFeatures(trial, featureIdx) = skewness(signal);
        
        % Feature 4: Kurtosis - tailedness of amplitude distribution
        featureIdx = featureIdx + 1;
        timeDomainFeatures(trial, featureIdx) = kurtosis(signal);
        
        % Feature 5: Peak-to-peak amplitude - signal range
        featureIdx = featureIdx + 1;
        timeDomainFeatures(trial, featureIdx) = max(signal) - min(signal);
        
        % Feature 6: RMS (Root Mean Square) - effective amplitude
        featureIdx = featureIdx + 1;
        timeDomainFeatures(trial, featureIdx) = rms(signal);
        
        % Feature 7: Zero crossings - frequency-related measure
        % Count transitions across the mean value
        featureIdx = featureIdx + 1;
        zeroCross = sum(abs(diff(sign(signal - mean(signal)))) > 0);
        timeDomainFeatures(trial, featureIdx) = zeroCross;
        
        % Feature 8: Line length - sum of absolute differences (signal complexity)
        featureIdx = featureIdx + 1;
        timeDomainFeatures(trial, featureIdx) = sum(abs(diff(signal)));
    end
end

% Create descriptive feature names
idx = 0;
featureTypes = {'mean', 'var', 'skew', 'kurt', 'p2p', 'rms', 'zcr', 'linelen'};
for ch = 1:nChannels
    for ft = 1:length(featureTypes)
        idx = idx + 1;
        timeFeatureNames{idx} = sprintf('Ch%d_%s', ch, featureTypes{ft});
    end
end

fprintf('Time-domain features extracted: %d features per trial\n', size(timeDomainFeatures, 2));
fprintf('Feature matrix size: [%d x %d]\n\n', size(timeDomainFeatures));

%% ==================== SECTION 8: HJORTH PARAMETERS =======================
% Hjorth parameters are time-domain descriptors of EEG signals:
%   - Activity: Variance of the signal (represents signal power)
%   - Mobility: Square root of variance ratio of 1st derivative to signal
%               (represents mean frequency)
%   - Complexity: Ratio of mobility of 1st derivative to mobility of signal
%                 (represents bandwidth/frequency spread)

fprintf('=== Section 8: Extracting Hjorth Parameters ===\n');

% Number of Hjorth features per channel
nHjorthFeatures = 3;  % Activity, Mobility, Complexity
hjorthFeatures = zeros(nTrials, nChannels * nHjorthFeatures);
hjorthNames = {};

% Process each trial
for trial = 1:nTrials
    epochData = epochs(:, :, trial);
    featureIdx = 0;
    
    % Process each channel
    for ch = 1:nChannels
        signal = epochData(ch, :);
        
        % Compute first derivative (approximation using differences)
        d1 = diff(signal);
        
        % Compute second derivative
        d2 = diff(d1);
        
        % Hjorth Activity: variance of the signal
        activity = var(signal);
        
        % Hjorth Mobility: sqrt(var(d1) / var(signal))
        mobility = sqrt(var(d1) / (activity + eps));
        
        % Hjorth Complexity: mobility of derivative / mobility of signal
        complexity = sqrt(var(d2) / (var(d1) + eps)) / (mobility + eps);
        
        % Store Hjorth parameters
        featureIdx = featureIdx + 1;
        hjorthFeatures(trial, featureIdx) = activity;
        featureIdx = featureIdx + 1;
        hjorthFeatures(trial, featureIdx) = mobility;
        featureIdx = featureIdx + 1;
        hjorthFeatures(trial, featureIdx) = complexity;
    end
end

% Create descriptive feature names
idx = 0;
hjorthTypes = {'activity', 'mobility', 'complexity'};
for ch = 1:nChannels
    for ht = 1:length(hjorthTypes)
        idx = idx + 1;
        hjorthNames{idx} = sprintf('Ch%d_hjorth_%s', ch, hjorthTypes{ht});
    end
end

fprintf('Hjorth features extracted: %d features per trial\n', size(hjorthFeatures, 2));
fprintf('Feature matrix size: [%d x %d]\n\n', size(hjorthFeatures));

%% ==================== SECTION 9: COMBINE ALL FEATURES ====================
% Concatenate all extracted feature sets into a single feature matrix
% This creates a comprehensive feature representation for classification

fprintf('=== Section 9: Combining All Features ===\n');

% Horizontally concatenate all feature matrices
allFeatures = [bandPowerFeatures, relativeBandPower, cspFeatures, ...
               fbcspFeatures, waveletFeatures, timeDomainFeatures, hjorthFeatures];

% Concatenate all feature names
allFeatureNames = [bandPowerNames, bandPowerNames, cspFilterNames, ...
                   fbcspNames, waveletNames, timeFeatureNames, hjorthNames];

% Create structure to track feature set indices for later reference
featureSetInfo = struct();

% Band Power feature indices
featureSetInfo.bandPower = struct(...
    'startIdx', 1, ...
    'endIdx', size(bandPowerFeatures, 2), ...
    'nFeatures', size(bandPowerFeatures, 2));

% Relative Band Power feature indices
featureSetInfo.relativeBandPower = struct(...
    'startIdx', size(bandPowerFeatures, 2) + 1, ...
    'endIdx', size(bandPowerFeatures, 2) + size(relativeBandPower, 2), ...
    'nFeatures', size(relativeBandPower, 2));

% CSP feature indices
startIdx = featureSetInfo.relativeBandPower.endIdx + 1;
featureSetInfo.csp = struct(...
    'startIdx', startIdx, ...
    'endIdx', startIdx + size(cspFeatures, 2) - 1, ...
    'nFeatures', size(cspFeatures, 2));

% FBCSP feature indices
startIdx = featureSetInfo.csp.endIdx + 1;
featureSetInfo.fbcsp = struct(...
    'startIdx', startIdx, ...
    'endIdx', startIdx + size(fbcspFeatures, 2) - 1, ...
    'nFeatures', size(fbcspFeatures, 2));

% Wavelet feature indices
startIdx = featureSetInfo.fbcsp.endIdx + 1;
featureSetInfo.wavelet = struct(...
    'startIdx', startIdx, ...
    'endIdx', startIdx + size(waveletFeatures, 2) - 1, ...
    'nFeatures', size(waveletFeatures, 2));

% Time-domain feature indices
startIdx = featureSetInfo.wavelet.endIdx + 1;
featureSetInfo.timeDomain = struct(...
    'startIdx', startIdx, ...
    'endIdx', startIdx + size(timeDomainFeatures, 2) - 1, ...
    'nFeatures', size(timeDomainFeatures, 2));

% Hjorth feature indices
startIdx = featureSetInfo.timeDomain.endIdx + 1;
featureSetInfo.hjorth = struct(...
    'startIdx', startIdx, ...
    'endIdx', startIdx + size(hjorthFeatures, 2) - 1, ...
    'nFeatures', size(hjorthFeatures, 2));

% Display summary of combined features
fprintf('Total features combined: %d\n', size(allFeatures, 2));
fprintf('\nFeature breakdown:\n');
fprintf('  Band Power:          %d features\n', size(bandPowerFeatures, 2));
fprintf('  Relative Band Power: %d features\n', size(relativeBandPower, 2));
fprintf('  CSP:                 %d features\n', size(cspFeatures, 2));
fprintf('  FBCSP:               %d features\n', size(fbcspFeatures, 2));
fprintf('  Wavelet:             %d features\n', size(waveletFeatures, 2));
fprintf('  Time-Domain:         %d features\n', size(timeDomainFeatures, 2));
fprintf('  Hjorth:              %d features\n', size(hjorthFeatures, 2));
fprintf('\n');

%% ==================== SECTION 10: FEATURE NORMALIZATION ==================
% Normalize features using Z-score (standardization)
% Z-score: (x - mean) / std
% This ensures all features have zero mean and unit variance,
% which is important for many machine learning algorithms

fprintf('=== Section 10: Normalizing Features ===\n');

% Compute mean and standard deviation across trials (for each feature)
featureMean = mean(allFeatures, 1);
featureStd = std(allFeatures, 0, 1);

% Prevent division by zero for constant features
featureStd(featureStd < eps) = 1;

% Apply Z-score normalization
normalizedFeatures = (allFeatures - featureMean) ./ featureStd;

% Check for and handle any NaN or Inf values
nanMask = isnan(normalizedFeatures) | isinf(normalizedFeatures);
if any(nanMask(:))
    fprintf('Warning: Found %d NaN/Inf values, replacing with 0\n', sum(nanMask(:)));
    normalizedFeatures(nanMask) = 0;
end

% Display normalization verification
fprintf('Features normalized using Z-score.\n');
fprintf('Mean of normalized features: %.6f (expected: ~0)\n', mean(normalizedFeatures(:)));
fprintf('Std of normalized features: %.6f (expected: ~1)\n\n', std(normalizedFeatures(:)));

%% ==================== SECTION 11: FEATURE SELECTION (FISHER SCORE) =======
% Feature selection using Fisher Score (also known as Fisher Discriminant Ratio)
% Fisher Score measures the discriminative power of each feature
% Higher score = better class separation
% Formula: F = Between-class scatter / Within-class scatter

fprintf('=== Section 11: Feature Selection Analysis (Fisher Score) ===\n');

nFeatures = size(normalizedFeatures, 2);
fScores = zeros(1, nFeatures);

% Compute overall mean for all features
overallMean = mean(normalizedFeatures, 1);

% Initialize scatter matrices
Sb = zeros(1, nFeatures);   % Between-class scatter
Sw = zeros(1, nFeatures);   % Within-class scatter

% Compute scatter for each class
for c = 1:nClasses
    % Get data for current class
    classIdx = (numericLabels == c);
    classData = normalizedFeatures(classIdx, :);
    nSamplesClass = sum(classIdx);
    
    % Compute class mean and variance
    classMean = mean(classData, 1);
    classVar = var(classData, 0, 1);
    
    % Update between-class scatter: weighted squared difference from overall mean
    Sb = Sb + nSamplesClass * (classMean - overallMean).^2;
    
    % Update within-class scatter: pooled class variance
    Sw = Sw + (nSamplesClass - 1) * classVar;
end

% Compute Fisher Score: Sb / Sw (with regularization to prevent division by zero)
fScores = Sb ./ (Sw + eps);

% Sort features by Fisher Score in descending order
[sortedScores, sortedIdx] = sort(fScores, 'descend');

% Select top features based on Fisher Score
nTopFeatures = min(100, length(fScores));   % Select top 100 or fewer
topFeatureIdx = sortedIdx(1:nTopFeatures);
selectedFeatures = normalizedFeatures(:, topFeatureIdx);

% Display feature selection results
fprintf('Top %d features selected based on Fisher Score.\n', nTopFeatures);
fprintf('Top 5 most discriminative features:\n');
for i = 1:min(5, nTopFeatures)
    fIdx = sortedIdx(i);
    fName = allFeatureNames{fIdx};
    fprintf(' %d. %s (Score=%.2f)\n', i, fName, sortedScores(i));
end
fprintf('\n');

%% ==================== SECTION 12: SAVE RESULTS ===========================
% Save all extracted features and metadata to MAT files
% Two output files are created:
%   1. Complete feature data with all metadata
%   2. Simplified version with just feature matrix and labels

fprintf('=== Section 12: Saving Feature Data ===\n');

% Create comprehensive output structure
featureData = struct();

% Store all normalized features
featureData.allFeatures = normalizedFeatures;
featureData.allFeatureNames = allFeatureNames;

% Store individual feature sets (before concatenation)
featureData.bandPower = bandPowerFeatures;
featureData.relativeBandPower = relativeBandPower;
featureData.csp = cspFeatures;
featureData.fbcsp = fbcspFeatures;
featureData.wavelet = waveletFeatures;
featureData.timeDomain = timeDomainFeatures;
featureData.hjorth = hjorthFeatures;

% Store selected features and selection info
featureData.selectedFeatures = selectedFeatures;
featureData.selectedFeatureIdx = topFeatureIdx;
featureData.featureScores = fScores;

% Store label information
featureData.labels = labels;
featureData.numericLabels = numericLabels;
featureData.uniqueLabels = uniqueLabels;
featureData.nClasses = nClasses;

% Store metadata
featureData.featureSetInfo = featureSetInfo;
featureData.nTrials = nTrials;
featureData.nChannels = nChannels;
featureData.srate = srate;

% Store normalization parameters (needed for applying to new data)
featureData.normParams = struct('mean', featureMean, 'std', featureStd);

% Store processing information for reproducibility
featureData.processingInfo = struct(...
    'extractionDate', datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
    'matlabVersion', version, ...
    'waveletName', waveletName, ...
    'nWaveletLevels', nLevels, ...
    'filterBankRanges', filterBank, ...
    'nCSPFilters', nCSPFilters, ...
    'nFBCSPFilters', nFBCSPFilters);

% Save complete feature data to file
OUTPUT_FILE = 'fourclass_features.mat';
save(OUTPUT_FILE, 'featureData', '-v7.3');
fprintf('Feature data saved to: %s\n', OUTPUT_FILE);

% Save simplified version for quick loading (just X and y matrices)
X = normalizedFeatures;     % Feature matrix [nTrials x nFeatures]
y = numericLabels;          % Label vector [nTrials x 1]
save('fourclass_features_simple.mat', 'X', 'y', 'uniqueLabels');
fprintf('Simplified data saved to: fourclass_features_simple.mat\n\n');

%% ==================== SECTION 13: VISUALIZATION ==========================
% Generate comprehensive visualization of feature extraction results
% Includes: score distribution, top features, correlation matrix, 
%           band power by class, t-SNE/PCA visualization, variance explained

fprintf('=== Section 13: Generating Visualizations ===\n');

% Create figure with multiple subplots
figure('Name', 'Feature Extraction Results', 'Position', [50, 50, 1600, 900], 'Color', 'w');

% --- Subplot 1: Histogram of Feature Importance Scores ---
subplot(2, 3, 1);
histogram(fScores, 50, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Fisher Score');
ylabel('Count');
title('Distribution of Feature Importance');
grid on; 
axis tight;

% --- Subplot 2: Top 10 Most Discriminative Features (Horizontal Bar Chart) ---
subplot(2, 3, 2);
nShowTop = 10;
topScores = sortedScores(1:nShowTop);
topIndices = sortedIdx(1:nShowTop);

% Create horizontal bar chart (flipped for top-to-bottom order)
barh(flip(topScores), 'FaceColor', [0.8 0.3 0.3]); 
yticks(1:nShowTop);

% Get names of top features and set as y-tick labels
topNames = allFeatureNames(topIndices);
yticklabels(flip(topNames));
xlabel('Fisher Score');
title(['Top ', num2str(nShowTop), ' Discriminative Features']);
set(gca, 'TickLabelInterpreter', 'none', 'FontSize', 8);
grid on;

% --- Subplot 3: Feature Correlation Matrix ---
subplot(2, 3, 3);
nShowCorr = min(40, size(selectedFeatures, 2));     % Limit for visibility
corrMatrix = corrcoef(selectedFeatures(:, 1:nShowCorr));
imagesc(corrMatrix);
c = colorbar;
c.Label.String = 'Correlation Coefficient';
colormap(gca, 'jet');
title(['Correlation Matrix (Top ', num2str(nShowCorr), ' Features)']);
xlabel('Feature Rank'); 
ylabel('Feature Rank');
axis square;

% --- Subplot 4: Mean Band Power by Class ---
subplot(2, 3, 4);
meanBandPower = zeros(nClasses, nBands);

% Calculate mean normalized band power for each class
for c = 1:nClasses
    classIdx = (numericLabels == c);
    for b = 1:nBands
        % Find columns corresponding to each frequency band
        cols = contains(allFeatureNames, bandNames{b}, 'IgnoreCase', true);
        if sum(cols) > 0
            meanBandPower(c, b) = mean(mean(normalizedFeatures(classIdx, cols)));
        end
    end
end

% Create grouped bar chart
bObj = bar(meanBandPower', 'grouped');
set(gca, 'XTickLabel', bandNames);
xlabel('Frequency Band');
ylabel('Mean Z-Score Power');
legend(uniqueLabels, 'Location', 'best', 'FontSize', 8);
title('Band Power Differences by Class (Normalized)');
grid on;

% --- Subplot 5: t-SNE Visualization of Feature Space ---
subplot(2, 3, 5);
hold on;

% Use top selected features for visualization
nVis = min(50, size(selectedFeatures, 2));
visData = selectedFeatures(:, 1:nVis);

try
    % Attempt t-SNE visualization with safe perplexity setting
    % Perplexity must be less than (n_samples - 1) / 3
    effPerp = min(30, floor((size(visData, 1)-1)/3));
    rng(42);    % Set random seed for reproducibility
    Y_plot = tsne(visData, 'Perplexity', effPerp);
    tTitle = 't-SNE Visualization';
    xLbl = 't-SNE 1'; 
    yLbl = 't-SNE 2';
catch
    % Fall back to PCA if t-SNE fails
    [~, sc] = pca(visData);
    Y_plot = sc(:, 1:2);
    tTitle = 'PCA Visualization (t-SNE failed)';
    xLbl = 'PC 1'; 
    yLbl = 'PC 2';
end

% Plot each class with different colors
colors = lines(nClasses);   % Generate distinct colors
for c = 1:nClasses
    idx = numericLabels == c;
    scatter(Y_plot(idx, 1), Y_plot(idx, 2), 40, colors(c, :), 'filled', ...
        'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.8, 'DisplayName', uniqueLabels{c});
end
hold off;

legend('Location', 'best');
title(tTitle); 
xlabel(xLbl); 
ylabel(yLbl);
grid on; 
axis tight;

% --- Subplot 6: PCA Cumulative Explained Variance ---
subplot(2, 3, 6);

% Perform PCA to analyze variance explained
[~, ~, ~, ~, explained] = pca(normalizedFeatures);
cumExplained = cumsum(explained);

% Find number of components for 95% variance
nPCA95 = find(cumExplained >= 95, 1, 'first');

% Plot cumulative variance explained
plot(cumExplained(1:min(50, end)), '-o', 'LineWidth', 2, 'MarkerSize', 4, 'Color', [0 0.5 0]);
yline(95, 'r--', '95% Variance', 'LabelHorizontalAlignment', 'left');
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained (%)');
title('PCA Explained Variance');
grid on; 
ylim([0 105]);

% Save figure to file
saveas(gcf, 'feature_extraction_results_improved.png');
fprintf('Visualization saved to: feature_extraction_results_improved.png\n\n');

%% ==================== SECTION 14: GENERATE REPORT ========================
% Create a text file report summarizing the feature extraction process

fprintf('=== Section 14: Generating Feature Extraction Report ===\n');

REPORT_FILE = 'feature_extraction_report.txt';
fid = fopen(REPORT_FILE, 'w');

% Write report header
fprintf(fid, '================================================================\n');
fprintf(fid, '         EEG FEATURE EXTRACTION REPORT - FOURCLASS DATA\n');
fprintf(fid, '================================================================\n\n');

% Processing information
fprintf(fid, 'Extraction Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'MATLAB Version: %s\n\n', version);

% Input data summary
fprintf(fid, '--- INPUT DATA ---\n');
fprintf(fid, 'Source file: %s\n', INPUT_FILE);
fprintf(fid, 'Number of trials: %d\n', nTrials);
fprintf(fid, 'Number of channels: %d\n', nChannels);
fprintf(fid, 'Number of classes: %d\n', nClasses);
fprintf(fid, 'Sampling rate: %.1f Hz\n\n', srate);

% Feature set details
fprintf(fid, '--- FEATURE SETS ---\n');
fprintf(fid, '1. Band Power Features: %d\n', size(bandPowerFeatures, 2));
fprintf(fid, '   Bands: Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz),\n');
fprintf(fid, '          Beta (13-30Hz), Gamma (30-40Hz)\n');
fprintf(fid, '2. Relative Band Power: %d\n', size(relativeBandPower, 2));
fprintf(fid, '3. CSP Features: %d (OVR approach, %d filters per class)\n', size(cspFeatures, 2), nCSPFilters*2);
fprintf(fid, '4. FBCSP Features: %d (%d frequency sub-bands)\n', size(fbcspFeatures, 2), nSubBands);
fprintf(fid, '5. Wavelet Features: %d (%s wavelet, %d decomposition levels)\n', size(waveletFeatures, 2), waveletName, nLevels);
fprintf(fid, '6. Time-Domain Features: %d\n', size(timeDomainFeatures, 2));
fprintf(fid, '7. Hjorth Parameters: %d\n\n', size(hjorthFeatures, 2));

% Summary statistics
fprintf(fid, '--- TOTAL FEATURES ---\n');
fprintf(fid, 'Total features extracted: %d\n', size(allFeatures, 2));
fprintf(fid, 'Selected features (top by Fisher Score): %d\n', nTopFeatures);
fprintf(fid, 'PCA components for 95%% variance: %d\n\n', nPCA95);

% Class distribution
fprintf(fid, '--- CLASS DISTRIBUTION ---\n');
for c = 1:nClasses
    count = sum(numericLabels == c);
    percentage = 100 * count / nTrials;
    fprintf(fid, '%s: %d trials (%.1f%%)\n', uniqueLabels{c}, count, percentage);
end

% Output files
fprintf(fid, '\n--- OUTPUT FILES ---\n');
fprintf(fid, '1. %s (complete feature data with metadata)\n', OUTPUT_FILE);
fprintf(fid, '2. fourclass_features_simple.mat (simplified X, y matrices)\n');
fprintf(fid, '3. feature_extraction_results_improved.png (visualization)\n');
fprintf(fid, '4. %s (this report)\n', REPORT_FILE);

% Report footer
fprintf(fid, '\n================================================================\n');
fprintf(fid, '                    END OF REPORT\n');
fprintf(fid, '================================================================\n');

fclose(fid);
fprintf('Report saved to: %s\n', REPORT_FILE);

%% ==================== COMPLETION =========================================
% Display final processing summary

processingTime = toc;

fprintf('\n');
fprintf('============================================================\n');
fprintf('  FEATURE EXTRACTION COMPLETE\n');
fprintf('============================================================\n');
fprintf('Total processing time: %.2f seconds (%.2f minutes)\n', processingTime, processingTime/60);
fprintf('Total features extracted: %d\n', size(allFeatures, 2));
fprintf('Selected features: %d\n', nTopFeatures);
fprintf('Output file: %s\n', OUTPUT_FILE);
fprintf('Data is ready for classification.\n');
fprintf('============================================================\n');

%% ==================== HELPER FUNCTIONS ===================================
% Local functions used by the main script

function [W, lambda] = computeCSP(class1Epochs, class2Epochs, nFilters)
    % COMPUTECSP Compute Common Spatial Pattern filters
    %
    % CSP finds spatial filters that maximize the variance of one class
    % while minimizing the variance of the other class.
    %
    % Inputs:
    %   class1Epochs - EEG epochs for class 1 [channels x samples x trials]
    %   class2Epochs - EEG epochs for class 2 [channels x samples x trials]
    %   nFilters     - Number of filters to extract per side
    %
    % Outputs:
    %   W      - CSP projection matrix [channels x 2*nFilters]
    %   lambda - Corresponding eigenvalues
    
    nChannels = size(class1Epochs, 1);
    
    % Initialize covariance matrices
    cov1 = zeros(nChannels, nChannels);
    cov2 = zeros(nChannels, nChannels);
    
    % Get number of trials for each class
    nTrials1 = size(class1Epochs, 3);
    nTrials2 = size(class2Epochs, 3);
    
    % Compute average normalized covariance matrix for class 1
    for i = 1:nTrials1
        epoch = class1Epochs(:, :, i);
        % Normalize by trace to account for amplitude differences
        cov1 = cov1 + (epoch * epoch') / trace(epoch * epoch');
    end
    cov1 = cov1 / nTrials1;
    
    % Compute average normalized covariance matrix for class 2
    for i = 1:nTrials2
        epoch = class2Epochs(:, :, i);
        cov2 = cov2 + (epoch * epoch') / trace(epoch * epoch');
    end
    cov2 = cov2 / nTrials2;
    
    % Add regularization for numerical stability
    regParam = 0.01;
    cov1 = cov1 + regParam * trace(cov1) / nChannels * eye(nChannels);
    cov2 = cov2 + regParam * trace(cov2) / nChannels * eye(nChannels);
    
    % Compute composite covariance matrix
    covComposite = cov1 + cov2;
    
    % Whitening transformation
    [V, D] = eig(covComposite);
    [d, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % Handle small eigenvalues to prevent numerical issues
    d(d < eps) = eps;
    P = diag(1./sqrt(d)) * V';  % Whitening matrix
    
    % Apply whitening to class 1 covariance
    S1 = P * cov1 * P';
    
    % Eigendecomposition of whitened covariance
    [B, lambda] = eig(S1);
    [lambda, idx] = sort(diag(lambda), 'descend');
    B = B(:, idx);
    
    % Compute CSP projection matrix
    W = (P' * B)';
    
    % Select filters: first nFilters (maximize class 1) and last nFilters (maximize class 2)
    nFilters = min(nFilters, floor(nChannels/2));
    filterIdx = [1:nFilters, (nChannels-nFilters+1):nChannels];
    W = W(filterIdx, :)';
    lambda = lambda(filterIdx);
end

function E = wentropy(x, type)
    % WENTROPY Compute wavelet entropy
    %
    % Entropy measures the complexity/information content of the signal
    %
    % Inputs:
    %   x    - Signal vector (wavelet coefficients)
    %   type - Entropy type: 'shannon', 'log energy', or 'energy' (default)
    %
    % Output:
    %   E    - Computed entropy value
    
    % Ensure column vector and remove zero values
    x = x(:);
    x = x(x ~= 0);
    
    % Compute entropy based on specified type
    switch lower(type)
        case 'shannon'
            % Shannon entropy: -sum(p * log(p))
            % where p is the normalized squared magnitude
            p = (x.^2) / sum(x.^2);
            p = p(p > 0);   % Remove zeros to avoid log(0)
            E = -sum(p .* log(p));
            
        case 'log energy'
            % Log energy entropy: sum(log(x^2))
            E = sum(log(x.^2 + eps));
            
        otherwise
            % Default: energy (sum of squared values)
            E = sum(x.^2);
    end
end