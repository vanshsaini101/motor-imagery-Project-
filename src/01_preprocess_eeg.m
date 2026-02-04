%% ==========================================================================
%  EEG PREPROCESSING PIPELINE FOR FOURCLASS MOTOR IMAGERY DATA
%  ==========================================================================
%  Description:
%    This script performs complete preprocessing of FourClass EEG data
%    including filtering, artifact removal (ICA), epoching, and normalization.
%  
%  Input:
%    - FourClass EEG.mat: Raw EEG data file with trigger channel
%  
%  Output:
%    - fourclass_eeg_processed.mat: Preprocessed epochs ready for classification
%    - eeg_processing_report.txt: Processing summary report
%  ==========================================================================

% Clear workspace, close all figures, and clear command window
clear; close all; clc;

% Start timer to measure total processing time
tic;

%% ==================== SECTION 1: DATA LOADING ============================
% Load raw EEG data from .mat file
% Expected structure: data (channels x samples), SR (sampling rate)

fprintf('=== Loading FourClass EEG Data ===\n');

% Define input file path
DATA_FILE = "FourClass EEG.mat";

% Validate file existence before loading
if ~isfile(DATA_FILE)
    error('Data file not found: %s', DATA_FILE);
end

% Load data structure from the .mat file
loadedData = load(DATA_FILE);

% Validate that all required fields exist in loaded data
requiredFields = {'data', 'SR'};
for i = 1:length(requiredFields)
    if ~isfield(loadedData, requiredFields{i})
        error('Missing required field: %s', requiredFields{i});
    end
end

fprintf('Data file loaded successfully.\n');

%% ==================== SECTION 2: CHANNEL SEPARATION ======================
% Separate EEG channels from trigger channel
% Channels 1-20: EEG signals (brain activity)
% Channel 21: Trigger/Event markers (experimental events)

fprintf('=== Separating EEG and Trigger Channels ===\n');

% Define channel configuration constants
TRIGGER_CHANNEL = 21;       % Index of the trigger channel
EEG_CHANNELS = 1:20;        % Indices of EEG recording channels

% Initialize EEG structure to hold data and parameters
EEG = struct();
EEG.data = loadedData.data;
EEG.srate = loadedData.SR;

% Get data dimensions for reference
[nChannels, nSamples] = size(EEG.data);

% Display data information summary
fprintf('Total channels: %d\n', nChannels);
fprintf('EEG channels: 1-%d\n', length(EEG_CHANNELS));
fprintf('Trigger channel: %d\n', TRIGGER_CHANNEL);
fprintf('Sampling rate: %.1f Hz\n', EEG.srate);
fprintf('Recording duration: %.2f seconds\n', nSamples / EEG.srate);

% Extract EEG signals (brain activity) and trigger signal (event markers)
eegSignals = EEG.data(EEG_CHANNELS, :);
triggerSignal = EEG.data(TRIGGER_CHANNEL, :);

% Clear original loaded data to free memory
loadedData = [];

%% ==================== SECTION 3: SIGNAL FILTERING ========================
% Apply bandpass and notch filters to remove noise and artifacts
% Bandpass: 1-40 Hz (removes DC drift and high-frequency noise)
% Notch: 50 Hz (removes power line interference)

fprintf('\n=== Applying Signal Filters ===\n');

% --- 3.1: Bandpass Filter (1-40 Hz) ---
% Purpose: Remove low-frequency drift and high-frequency noise
% Retains frequencies relevant to motor imagery (mu: 8-12 Hz, beta: 13-30 Hz)

BANDPASS_LOW = 1;    % Lower cutoff frequency in Hz
BANDPASS_HIGH = 40;  % Upper cutoff frequency in Hz
FILTER_ORDER = 4;    % Butterworth filter order (determines rolloff steepness)

fprintf('Applying Bandpass filter (%d-%d Hz)...', BANDPASS_LOW, BANDPASS_HIGH);

% Calculate Nyquist frequency (half of sampling rate)
nyquistFreq = EEG.srate / 2;

% Design Butterworth bandpass filter coefficients
[bBand, aBand] = butter(FILTER_ORDER, [BANDPASS_LOW BANDPASS_HIGH] / nyquistFreq, 'bandpass');

% Apply zero-phase filtering (filtfilt) to preserve signal timing
% Note: Data must be transposed for filtfilt, then transposed back
eegSignals = filtfilt(bBand, aBand, eegSignals')';

fprintf(' Done.\n');

% --- 3.2: Notch Filter (50 Hz) ---
% Purpose: Remove power line interference (50 Hz in most countries, 60 Hz in Americas)

NOTCH_FREQ = 50;     % Notch center frequency in Hz
NOTCH_BW = 2;        % Notch bandwidth in Hz (affects sharpness)

fprintf('Applying Notch filter (%d Hz)...', NOTCH_FREQ);

% Design IIR notch filter
% wo: normalized notch frequency, bw: normalized bandwidth
wo = NOTCH_FREQ / nyquistFreq;
bw = NOTCH_BW / nyquistFreq;
[bNotch, aNotch] = iirnotch(wo, bw);

% Apply zero-phase notch filtering to remove power line noise
eegSignals = filtfilt(bNotch, aNotch, eegSignals')';

fprintf(' Done.\n');

%% ==================== SECTION 4: TRIGGER DETECTION =======================
% Identify and extract trigger events from the trigger channel
% Triggers mark the onset of different motor imagery classes
% TARGET TRIGGERS: 46, 174, 110, 238 (each trial lasts 6 seconds)

fprintf('\n=== Detecting Trigger Events ===\n');

% ===================== MAIN CONFIGURATION =====================
% Define target triggers and their corresponding trial durations
TARGET_TRIGGERS = [46, 174, 110, 238];  % Target class trigger values
TRIGGER_DURATIONS = [6, 6, 6, 6];       % Duration of each trial in seconds

% Use only target triggers for epoch extraction
triggerValues = TARGET_TRIGGERS;
nClasses = length(triggerValues);

% Display trigger configuration
fprintf('Target trigger values: ');
fprintf('%d ', triggerValues);
fprintf('\n');
fprintf('Trial duration: %d seconds each\n', TRIGGER_DURATIONS(1));
% ==============================================================

% Pre-allocate cell arrays to store trigger information for each class
triggerPositions = cell(1, nClasses);  % Sample positions of trigger onsets
triggerLabels = cell(1, nClasses);     % Human-readable class labels

% Extract trigger onset positions for each class
for i = 1:nClasses
    currentTrigger = triggerValues(i);
    
    % Create binary mask where trigger value matches current trigger
    triggerMask = (triggerSignal == currentTrigger);
    
    % Detect rising edges (trigger onsets) using differential
    % A rising edge occurs where diff changes from 0 to 1
    diffSignal = diff([0, triggerMask]);
    onsetPositions = find(diffSignal == 1);
    
    % Store trigger information in cell arrays
    triggerPositions{i} = onsetPositions;
    triggerLabels{i} = sprintf('Class%d', currentTrigger);
    
    % Display detection results
    fprintf('Trigger %d (%s): %d events detected\n', ...
        currentTrigger, triggerLabels{i}, length(onsetPositions));
end

%% ==================== SECTION 5: ICA ARTIFACT REMOVAL ====================
% Remove artifacts using Independent Component Analysis (ICA)
% ICA identifies and removes components related to:
%   - Eye blinks and eye movements
%   - Muscle activity (EMG contamination)
%   - Other non-neural sources

fprintf('\n=== ICA Artifact Removal ===\n');

% Check if EEGLAB toolbox is available in the path
eeglabAvailable = exist('pop_runica', 'file') == 2;

if eeglabAvailable
    fprintf('EEGLAB detected. Running ICA...\n');
    
    % Create proper EEGLAB structure using eeg_emptyset if available
    try
        EEG_ica = eeg_emptyset();
    catch
        % If eeg_emptyset doesn't exist, create structure manually
        EEG_ica = struct();
    end
    
    % Fill in required EEGLAB structure fields
    EEG_ica.setname = 'FourClass_ICA';          % Dataset name
    EEG_ica.filename = '';                       % File name (empty for new data)
    EEG_ica.filepath = '';                       % File path (empty for new data)
    EEG_ica.subject = '';                        % Subject identifier
    EEG_ica.group = '';                          % Group identifier
    EEG_ica.condition = '';                      % Experimental condition
    EEG_ica.session = [];                        % Session number
    EEG_ica.comments = 'FourClass Motor Imagery Data';  % Dataset comments
    EEG_ica.nbchan = size(eegSignals, 1);        % Number of channels
    EEG_ica.trials = 1;                          % Number of trials (continuous = 1)
    EEG_ica.pnts = size(eegSignals, 2);          % Number of time points
    EEG_ica.srate = EEG.srate;                   % Sampling rate
    EEG_ica.xmin = 0;                            % Start time in seconds
    EEG_ica.xmax = (EEG_ica.pnts - 1) / EEG_ica.srate;  % End time in seconds
    EEG_ica.times = linspace(EEG_ica.xmin * 1000, EEG_ica.xmax * 1000, EEG_ica.pnts);  % Time vector in ms
    EEG_ica.data = eegSignals;                   % EEG data matrix
    
    % Initialize ICA-related fields (will be populated by ICA)
    EEG_ica.icawinv = [];                        % ICA inverse weight matrix
    EEG_ica.icasphere = [];                      % ICA sphering matrix
    EEG_ica.icaweights = [];                     % ICA weight matrix
    EEG_ica.icachansind = [];                    % Channels used for ICA
    
    % Initialize event-related fields
    EEG_ica.event = [];                          % Event structure
    EEG_ica.urevent = [];                        % Original unreferenced events
    EEG_ica.eventdescription = {};               % Event descriptions
    EEG_ica.epoch = [];                          % Epoch structure
    EEG_ica.epochdescription = {};               % Epoch descriptions
    
    % Initialize analysis-related fields
    EEG_ica.reject = [];                         % Rejection information
    EEG_ica.stats = [];                          % Statistics
    EEG_ica.specdata = [];                       % Spectral data
    EEG_ica.specicaact = [];                     % ICA component spectra
    EEG_ica.splinefile = '';                     % Spline file for topoplots
    EEG_ica.icasplinefile = '';                  % ICA spline file
    EEG_ica.dipfit = [];                         % Dipole fitting results
    EEG_ica.history = '';                        % Processing history
    EEG_ica.saved = 'no';                        % Save status
    EEG_ica.etc = [];                            % Miscellaneous information
    EEG_ica.ref = 'common';                      % Reference scheme
    
    % Create channel location structure with basic information
    EEG_ica.chanlocs = struct('labels', {}, 'type', {}, 'theta', {}, ...
        'radius', {}, 'X', {}, 'Y', {}, 'Z', {}, 'sph_theta', {}, ...
        'sph_phi', {}, 'sph_radius', {}, 'urchan', {}, 'ref', {});
    
    % Populate channel locations with default labels
    for ch = 1:EEG_ica.nbchan
        EEG_ica.chanlocs(ch).labels = sprintf('Ch%d', ch);  % Channel label
        EEG_ica.chanlocs(ch).type = 'EEG';                  % Channel type
        EEG_ica.chanlocs(ch).urchan = ch;                   % Original channel index
    end
    
    % Initialize channel information structure
    EEG_ica.chaninfo = struct('plotrad', [], 'shrink', [], 'nosedir', '+X', ...
        'icachansind', []);
    
    % Run ICA decomposition
    fprintf('Computing ICA decomposition (this may take several minutes)...\n');
    
    try
        % Perform ICA using extended Infomax algorithm
        EEG_ica = pop_runica(EEG_ica, 'icatype', 'runica', 'extended', 1);
        
        % Initialize array to store artifact component indices
        artifactComponents = [];
        ARTIFACT_THRESHOLD = 0.8;  % Probability threshold for artifact classification
        
        % Try automatic component classification using ICLabel plugin
        if exist('pop_iclabel', 'file') == 2
            fprintf('Using ICLabel for automatic artifact detection...\n');
            
            % Run ICLabel classification
            EEG_ica = pop_iclabel(EEG_ica);
            
            % Identify artifact components based on ICLabel classifications
            nComponents = size(EEG_ica.icaweights, 1);
            for comp = 1:nComponents
                % Get classification probabilities for current component
                classifications = EEG_ica.etc.ic_classification.ICLabel.classifications(comp, :);
                
                % ICLabel categories: Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other
                eyeProb = classifications(3);     % Eye artifact probability
                muscleProb = classifications(2);  % Muscle artifact probability
                
                % Mark component as artifact if eye or muscle probability exceeds threshold
                if eyeProb > ARTIFACT_THRESHOLD || muscleProb > ARTIFACT_THRESHOLD
                    artifactComponents = [artifactComponents, comp];
                end
            end
            
            fprintf('Identified %d artifact components via ICLabel.\n', length(artifactComponents));
        else
            % Fallback method: kurtosis-based artifact detection
            fprintf('ICLabel not available. Using kurtosis-based detection...\n');
            
            KURTOSIS_THRESHOLD = 5;  % Components with |kurtosis| > threshold are artifacts
            
            % Compute ICA activations (component time courses)
            icaActivations = EEG_ica.icaweights * EEG_ica.icasphere * eegSignals;
            
            % Calculate kurtosis for each component
            % High kurtosis often indicates non-Gaussian artifacts (e.g., eye blinks)
            kurtosisValues = kurtosis(icaActivations, [], 2);
            
            % Find components exceeding kurtosis threshold
            artifactComponents = find(abs(kurtosisValues) > KURTOSIS_THRESHOLD);
            
            fprintf('Identified %d high-kurtosis components.\n', length(artifactComponents));
        end
        
        % Remove identified artifact components from the data
        if ~isempty(artifactComponents)
            fprintf('Removing artifact components: ');
            fprintf('%d ', artifactComponents);
            fprintf('\n');
            
            % Subtract artifact components from the data
            EEG_ica = pop_subcomp(EEG_ica, artifactComponents, 0);
            
            % Update EEG signals with cleaned data
            eegSignals = EEG_ica.data;
            
            fprintf('Artifact components successfully removed.\n');
        else
            fprintf('No artifact components detected.\n');
        end
        
        % Flag indicating ICA was successfully applied
        icaApplied = true;
        
    catch ME
        % Handle ICA errors gracefully
        fprintf('WARNING: ICA failed with error: %s\n', ME.message);
        fprintf('Continuing without ICA...\n');
        icaApplied = false;
    end
    
else
    % EEGLAB not available - skip ICA processing
    fprintf('WARNING: EEGLAB not found. Skipping ICA.\n');
    fprintf('Install EEGLAB from: https://sccn.ucsd.edu/eeglab/\n');
    icaApplied = false;
end

%% ==================== SECTION 6: EPOCHING ================================
% Segment continuous data into epochs based on trigger events
% Each epoch is a time window around the trigger onset
% Contains baseline period (pre-trigger) and task period (post-trigger)

fprintf('\n=== Epoching Data ===\n');

% Define epoch time window boundaries (in seconds relative to trigger)
EPOCH_PRE = 0.5;   % Time before trigger onset (baseline period)
EPOCH_POST = 6.0;  % Time after trigger onset (task period - 6 second trial)

% Convert time boundaries to sample counts
preSamples = round(EPOCH_PRE * EEG.srate);     % Samples before trigger
postSamples = round(EPOCH_POST * EEG.srate);   % Samples after trigger
epochLength = preSamples + postSamples + 1;    % Total samples per epoch

% Display epoch configuration
fprintf('Epoch window: %.2f to %.2f seconds relative to trigger\n', -EPOCH_PRE, EPOCH_POST);
fprintf('Epoch length: %d samples (%.2f seconds)\n', epochLength, EPOCH_PRE + EPOCH_POST);

% Count total valid epochs for pre-allocation
% An epoch is valid if it doesn't extend beyond recording boundaries
totalEpochs = 0;
for i = 1:nClasses
    positions = triggerPositions{i};
    for j = 1:length(positions)
        startIdx = positions(j) - preSamples;
        endIdx = positions(j) + postSamples;
        
        % Check if epoch fits within recording boundaries
        if startIdx >= 1 && endIdx <= nSamples
            totalEpochs = totalEpochs + 1;
        end
    end
end

% Pre-allocate epoch array for memory efficiency
% Dimensions: channels x time samples x epochs
nEegChannels = length(EEG_CHANNELS);
allEpochs = zeros(nEegChannels, epochLength, totalEpochs);
allLabels = cell(1, totalEpochs);  % Cell array for class labels

% Extract epochs from continuous data
epochCounter = 0;
for classIdx = 1:nClasses
    triggerValue = triggerValues(classIdx);
    positions = triggerPositions{classIdx};
    label = triggerLabels{classIdx};
    
    validCount = 0;  % Counter for valid epochs in this class
    
    for i = 1:length(positions)
        % Calculate epoch boundaries in samples
        startIdx = positions(i) - preSamples;
        endIdx = positions(i) + postSamples;
        
        % Check boundary conditions to avoid index out of bounds
        if startIdx >= 1 && endIdx <= nSamples
            epochCounter = epochCounter + 1;
            
            % Extract epoch data segment
            allEpochs(:, :, epochCounter) = eegSignals(:, startIdx:endIdx);
            
            % Store corresponding class label
            allLabels{epochCounter} = label;
            
            validCount = validCount + 1;
        end
    end
    
    % Report extraction results for this class
    fprintf('Class %d (%s): %d valid epochs extracted\n', triggerValue, label, validCount);
end

% Trim pre-allocated arrays to actual size (in case some epochs were invalid)
allEpochs = allEpochs(:, :, 1:epochCounter);
allLabels = allLabels(1:epochCounter);

% Display final epoch count and dimensions
fprintf('Total epochs extracted: %d\n', epochCounter);
fprintf('Epoch dimensions: %d channels x %d samples\n', size(allEpochs, 1), size(allEpochs, 2));

%% ==================== SECTION 7: NORMALIZATION ===========================
% Apply Z-score normalization to each epoch and channel
% Standardizes data to zero mean and unit variance
% This helps with classifier training and feature extraction

fprintf('\n=== Normalizing Epochs ===\n');

% Pre-allocate array for normalized epochs
normalizedEpochs = zeros(size(allEpochs));
nTotalEpochs = size(allEpochs, 3);

% Normalize each epoch independently
for ep = 1:nTotalEpochs
    % Get data for current epoch
    epochData = allEpochs(:, :, ep);
    
    % Normalize each channel within the epoch separately
    for ch = 1:size(epochData, 1)
        channelData = epochData(ch, :);
        
        % Compute channel statistics
        mu = mean(channelData);     % Channel mean
        sigma = std(channelData);   % Channel standard deviation
        
        % Apply Z-score normalization: z = (x - mean) / std
        if sigma > eps  % Avoid division by zero for constant signals
            normalizedEpochs(ch, :, ep) = (channelData - mu) / sigma;
        else
            % If std is zero, just center the data
            normalizedEpochs(ch, :, ep) = channelData - mu;
        end
    end
    
    % Display progress indicator every 50 epochs
    if mod(ep, 50) == 0 || ep == nTotalEpochs
        fprintf('Progress: %d/%d epochs normalized (%.1f%%)\n', ep, nTotalEpochs, 100*ep/nTotalEpochs);
    end
end

% Verify normalization by checking global statistics
finalMean = mean(normalizedEpochs(:));
finalStd = std(normalizedEpochs(:));

fprintf('Normalization complete.\n');
fprintf('Global mean: %.6f (expected: ~0)\n', finalMean);
fprintf('Global std: %.6f (expected: ~1)\n', finalStd);

%% ==================== SECTION 8: SAVE RESULTS ============================
% Save processed data to .mat file for subsequent analysis
% Uses structured format for easy access to all parameters

fprintf('\n=== Saving Processed Data ===\n');

% Create structured output containing all processed data and metadata
processedData = struct();

% Store epoch data and labels
processedData.epochs = normalizedEpochs;    % Normalized epoch data (channels x samples x epochs)
processedData.labels = allLabels;           % Class labels for each epoch

% Store trigger information
processedData.triggerValues = triggerValues;    % Trigger values for each class
processedData.triggerLabels = triggerLabels;    % Human-readable class labels

% Store recording parameters
processedData.srate = EEG.srate;        % Sampling rate in Hz
processedData.channels = EEG_CHANNELS;  % Channel indices used

% Store epoch timing information
processedData.epochInfo = struct(...
    'preTriggerTime', EPOCH_PRE, ...                                    % Pre-trigger time in seconds
    'postTriggerTime', EPOCH_POST, ...                                  % Post-trigger time in seconds
    'totalDuration', EPOCH_PRE + EPOCH_POST, ...                        % Total epoch duration
    'samplesPerEpoch', epochLength, ...                                 % Number of samples per epoch
    'timeVector', linspace(-EPOCH_PRE, EPOCH_POST, epochLength));       % Time vector for plotting

% Store processing parameters and metadata for reproducibility
processedData.processingInfo = struct(...
    'processingDate', datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...          % Processing timestamp
    'matlabVersion', version, ...                                        % MATLAB version used
    'bandpassFilter', sprintf('%d-%d Hz', BANDPASS_LOW, BANDPASS_HIGH), ... % Bandpass filter settings
    'notchFilter', sprintf('%d Hz', NOTCH_FREQ), ...                    % Notch filter frequency
    'filterOrder', FILTER_ORDER, ...                                     % Filter order
    'icaApplied', icaApplied, ...                                       % Whether ICA was applied
    'normalization', 'per-channel Z-score');                            % Normalization method

% Save to file using v7.3 format for large data support
OUTPUT_FILE = 'fourclass_eeg_processed.mat';
save(OUTPUT_FILE, 'processedData', '-v7.3');

fprintf('Processed data saved to: %s\n', OUTPUT_FILE);

%% ==================== SECTION 9: VISUALIZATION (IMPROVED) ================
% Generate professional quality assessment plots
% Provides visual verification of preprocessing results

fprintf('\n=== Generating Improved Visualization ===\n');

% Get unique class labels and their counts for visualization
uniqueLabels = unique(allLabels, 'stable');
classCounts = zeros(1, length(uniqueLabels));
for i = 1:length(uniqueLabels)
    classCounts(i) = sum(strcmp(allLabels, uniqueLabels{i}));
end

% Create main figure with white background
figure('Name', 'EEG Preprocessing Quality Control', ...
    'Position', [50, 50, 1600, 900], 'Color', 'w');

% Define time vector for epoch plots (in seconds relative to trigger)
timeEpoch = linspace(-EPOCH_PRE, EPOCH_POST, epochLength);

% --- Subplot 1: Experiment Timeline (Trigger Events) ---
% Shows when each trigger type occurred during the recording
subplot(2, 3, 1);
hold on;

% Define colors for each class
stemColors = lines(nClasses);

% Plot stem for each trigger class
for i = 1:nClasses
    % Convert sample positions to time in seconds
    events = triggerPositions{i} / EEG.srate;
    
    % Create stem plot for trigger events
    stem(events, repmat(triggerValues(i), size(events)), ...
        'Color', stemColors(i,:), 'MarkerFaceColor', stemColors(i,:), ...
        'DisplayName', triggerLabels{i});
end
hold off;

% Format subplot
xlabel('Time (seconds)');
ylabel('Trigger ID');
title('Experiment Event Timeline');
legend('Location', 'best', 'FontSize', 8);
grid on;
axis tight;
ylim([0, max(triggerValues) * 1.2]);  % Add space above for legend

% --- Subplot 2: Class Distribution (With Labels) ---
% Bar chart showing number of epochs per class
subplot(2, 3, 2);

% Create bar plot
b = bar(classCounts, 'FaceColor', [0.2, 0.6, 0.8]);
set(gca, 'XTickLabel', uniqueLabels, 'TickLabelInterpreter', 'none');
xtickangle(45);

% Add count labels on top of each bar
xtips = b.XEndPoints;
ytips = b.YEndPoints;
labels = string(b.YData);
text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

% Format subplot
ylabel('Count');
title('Epoch Distribution');
grid on;
ylim([0, max(classCounts)*1.15]);  % Add space for labels

% --- Subplot 3: Representative Epoch (Butterfly Plot) ---
% Shows all channels of a single trial overlaid
subplot(2, 3, 3);

% Find first epoch of class 1 as representative sample
sampleTrialIdx = find(strcmp(allLabels, triggerLabels{1}), 1);

% Plot all channels in gray with transparency
plot(timeEpoch, normalizedEpochs(:, :, sampleTrialIdx)', 'Color', [0.6 0.6 0.6 0.5]);
hold on;

% Overlay mean across channels (Global Field Power approximation)
plot(timeEpoch, mean(normalizedEpochs(:, :, sampleTrialIdx), 1), 'k', 'LineWidth', 2);

% Add vertical line at trigger onset
xline(0, 'r--', 'Onset', 'HandleVisibility', 'off');
hold off;

% Format subplot
title(['Single Trial Quality (Epoch ' num2str(sampleTrialIdx) ')']);
xlabel('Time (s)');
ylabel('Amp (Z-score)');
grid on;
axis tight;

% --- Subplot 4: Smoothed Grand Average ERP ---
% Shows average response for each class
subplot(2, 3, 4);
colors = lines(nClasses);
hold on;

% Calculate and plot grand average for each class
for i = 1:nClasses
    % Find epochs belonging to current class
    classIdx = strcmp(allLabels, uniqueLabels{i});
    
    if any(classIdx)
        % Calculate grand average: mean over all channels and all epochs
        classGrandAvg = mean(mean(normalizedEpochs(:, :, classIdx), 1), 3);
        
        % Apply Gaussian smoothing for cleaner visualization
        smoothAvg = smoothdata(classGrandAvg, 'gaussian', 20);
        
        % Plot smoothed grand average
        plot(timeEpoch, smoothAvg, 'Color', colors(i, :), 'LineWidth', 2, ...
            'DisplayName', uniqueLabels{i});
    end
end

% Add reference lines
xline(0, 'k--', 'HandleVisibility', 'off');   % Trigger onset
yline(0, 'k-', 'HandleVisibility', 'off');    % Zero amplitude
hold off;

% Format subplot
xlabel('Time (s)');
ylabel('Amplitude (Smoothed)');
title('Grand Average ERP (Smoothed)');
legend('Location', 'best', 'FontSize', 8);
grid on;
xlim([-0.5 2.5]);  % Zoom to focus on relevant time window

% --- Subplot 5 & 6: Spectrograms (Time-Frequency Analysis) ---
% Time-frequency representations are crucial for motor imagery analysis
% Shows power changes in mu (8-12 Hz) and beta (13-30 Hz) bands

% Define spectrogram parameters
window = round(0.5 * EEG.srate);      % 500 ms window
noverlap = round(0.45 * EEG.srate);   % 90% overlap for smooth display
nfft = 2^nextpow2(window);            % FFT points (power of 2 for efficiency)
freqLimit = 35;                        % Display up to 35 Hz

% --- Spectrogram for Class 1 ---
subplot(2, 3, 5);

% Find epochs for class 1
class1Idx = strcmp(allLabels, uniqueLabels{1});

% Average across epochs and channels before spectrogram (reduces noise)
meanSignalC1 = mean(mean(normalizedEpochs(:, :, class1Idx), 3), 1);

% Compute spectrogram
[s, f, t, p] = spectrogram(meanSignalC1, window, noverlap, nfft, EEG.srate);

% Display spectrogram with correct time axis
imagesc(t - EPOCH_PRE, f, 10*log10(p));  % Convert to dB scale
axis xy;  % Flip y-axis so low frequencies are at bottom
colormap(gca, 'jet');
colorbar;
ylim([0 freqLimit]);

% Add vertical line at trigger onset
xline(0, 'w--', 'LineWidth', 2);

% Format subplot
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title(['Spectrogram: ' uniqueLabels{1}]);

% --- Spectrogram for Class 2 ---
subplot(2, 3, 6);

% Find epochs for class 2
class2Idx = strcmp(allLabels, uniqueLabels{2});

% Average across epochs and channels
meanSignalC2 = mean(mean(normalizedEpochs(:, :, class2Idx), 3), 1);

% Compute and display spectrogram
[s, f, t, p] = spectrogram(meanSignalC2, window, noverlap, nfft, EEG.srate);
imagesc(t - EPOCH_PRE, f, 10*log10(p));
axis xy;
colormap(gca, 'jet');
colorbar;
ylim([0 freqLimit]);
xline(0, 'w--', 'LineWidth', 2);

% Format subplot
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title(['Spectrogram: ' uniqueLabels{2}]);

% Save figure to file
saveas(gcf, 'preprocessing_results_improved.png');
fprintf('Improved visualization saved to: preprocessing_results_improved.png\n');

%% ==================== SECTION 10: GENERATE REPORT ========================
% Create text report summarizing the preprocessing pipeline
% Documents all parameters for reproducibility

fprintf('\n=== Generating Processing Report ===\n');

% Open report file for writing
REPORT_FILE = 'eeg_processing_report.txt';
fid = fopen(REPORT_FILE, 'w');

% Write report header
fprintf(fid, '================================================================\n');
fprintf(fid, '          EEG PREPROCESSING REPORT - FOURCLASS DATA\n');
fprintf(fid, '================================================================\n\n');

% Write timestamp and version info
fprintf(fid, 'Processing Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'MATLAB Version: %s\n\n', version);

% Write input data information
fprintf(fid, '--- INPUT DATA ---\n');
fprintf(fid, 'Source file: %s\n', DATA_FILE);
fprintf(fid, 'EEG channels: %d-%d\n', EEG_CHANNELS(1), EEG_CHANNELS(end));
fprintf(fid, 'Trigger channel: %d\n', TRIGGER_CHANNEL);
fprintf(fid, 'Sampling rate: %.1f Hz\n', EEG.srate);
fprintf(fid, 'Recording duration: %.2f seconds\n\n', nSamples / EEG.srate);

% Write preprocessing steps summary
fprintf(fid, '--- PREPROCESSING STEPS ---\n');
fprintf(fid, '1. Channel separation (EEG vs Trigger)\n');
fprintf(fid, '2. Bandpass filter: %d-%d Hz (Butterworth, order %d)\n', BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER);
fprintf(fid, '3. Notch filter: %d Hz (power line removal)\n', NOTCH_FREQ);
fprintf(fid, '4. ICA artifact removal: %s\n', ternary(icaApplied, 'Applied', 'Skipped (EEGLAB not found)'));
fprintf(fid, '5. Epoching: %.2f to %.2f s relative to trigger\n', -EPOCH_PRE, EPOCH_POST);
fprintf(fid, '6. Normalization: Per-channel Z-score\n\n');

% Write output data summary
fprintf(fid, '--- OUTPUT DATA ---\n');
fprintf(fid, 'Total epochs: %d\n', size(normalizedEpochs, 3));
fprintf(fid, 'Epoch dimensions: %d channels x %d samples\n', size(normalizedEpochs, 1), size(normalizedEpochs, 2));
fprintf(fid, 'Epoch duration: %.2f seconds\n\n', EPOCH_PRE + EPOCH_POST);

% Write class distribution
fprintf(fid, '--- CLASS DISTRIBUTION ---\n');
for i = 1:length(uniqueLabels)
    count = classCounts(i);
    percentage = 100 * count / length(allLabels);
    fprintf(fid, '%s: %d epochs (%.1f%%)\n', uniqueLabels{i}, count, percentage);
end

% Write output file list
fprintf(fid, '\n--- OUTPUT FILES ---\n');
fprintf(fid, '1. %s (processed epochs)\n', OUTPUT_FILE);
fprintf(fid, '2. %s (this report)\n', REPORT_FILE);
fprintf(fid, '3. preprocessing_results.png (visualization)\n');

% Write report footer
fprintf(fid, '\n================================================================\n');
fprintf(fid, '                    END OF REPORT\n');
fprintf(fid, '================================================================\n');

% Close file
fclose(fid);

fprintf('Report saved to: %s\n', REPORT_FILE);

%% ==================== COMPLETION =========================================
% Display final summary and processing time

% Stop timer and get elapsed time
processingTime = toc;

% Print completion summary
fprintf('\n');
fprintf('============================================================\n');
fprintf('  PREPROCESSING COMPLETE\n');
fprintf('============================================================\n');
fprintf('Total processing time: %.2f seconds\n', processingTime);
fprintf('Output file: %s\n', OUTPUT_FILE);
fprintf('Total epochs: %d\n', size(normalizedEpochs, 3));
fprintf('Data is ready for feature extraction and classification.\n');
fprintf('============================================================\n');

%% ==================== HELPER FUNCTIONS ===================================

function result = ternary(condition, trueVal, falseVal)
    %TERNARY Ternary operator implementation for MATLAB
    %   result = TERNARY(condition, trueVal, falseVal)
    %   Returns trueVal if condition is true, falseVal otherwise
    %
    %   Inputs:
    %       condition - Logical condition to evaluate
    %       trueVal   - Value to return if condition is true
    %       falseVal  - Value to return if condition is false
    %
    %   Output:
    %       result    - Either trueVal or falseVal based on condition
    
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end