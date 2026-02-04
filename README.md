# EEG Motor Imagery Classification

A comprehensive MATLAB pipeline for processing and classifying Motor Imagery (MI) EEG signals using the FourClass dataset. This project implements advanced signal processing, feature extraction (CSP, FBCSP, Wavelet), and machine learning classification (SVM, CNN, LSTM, Ensemble Learning).

## 🚀 Project Overview

The goal of this project is to classify 4 different motor imagery tasks based on EEG signals. The pipeline achieves **98.75% accuracy** using an optimized Linear SVM with FBCSP features.

### Key Features
- **Preprocessing:** Bandpass filtering (1-40Hz), Notch filtering (50Hz), and Z-score normalization.
- **Feature Extraction:** 
  - Common Spatial Patterns (CSP)
  - Filter Bank CSP (FBCSP)
  - Wavelet Transform (db4)
  - Power Spectral Density (PSD)
- **Feature Selection:** Fisher Score analysis.
- **Classification:** 
  - Traditional ML: SVM (Linear/RBF), LDA, KNN, Random Forest.
  - Deep Learning: 1D-CNN, LSTM, BiLSTM.
- **Optimization:** Hyperparameter tuning (Grid Search) and Ensemble Voting/Stacking.

## 📂 Repository Structure

```
├── 01_preprocess_eeg.m          # Step 1: Data loading & cleaning
├── 02_feature_extraction.m      # Step 2: Feature generation & selection
├── 03_train_classifiers.m       # Step 3: Training ML & DL models
├── 04_evaluate_and_optimize.m   # Step 4: Final evaluation & improvements
└── README.md                    # Project documentation
```

## 📊 Results

| Method | Accuracy | F1-Score |
| :--- | :---: | :---: |
| **SVM (Linear) + FBCSP** | **98.75%** | **0.99** |
| Optimized SVM (RBF) | 96.25% | 0.96 |
| Weighted Ensemble | 93.75% | 0.94 |
| KNN | 90.00% | 0.89 |
| LDA | 80.00% | 0.79 |
| 1D-CNN | ~25.00%* | 0.25 |

*\*Note: Deep learning models underperformed due to the small sample size (80 trials).*

## 🛠️ Requirements

- MATLAB R2020b or later.
- Signal Processing Toolbox.
- Statistics and Machine Learning Toolbox.
- Deep Learning Toolbox (optional, for CNN/LSTM).

## ▶️ How to Run

1. Place the `FourClass EEG.mat` dataset in the project root or `data/` folder.
2. Run the scripts in order:
   - `01_preprocess_eeg.m` -> Generates `fourclass_eeg_processed.mat`
   - `02_feature_extraction.m` -> Generates `fourclass_features.mat`
   - `03_train_classifiers.m` -> Trains models & initial evaluation.
   - `04_evaluate_and_optimize.m` -> Performs final comparison & statistical tests.

## 📈 Visualizations

The pipeline generates comprehensive visualization reports including:
- Preprocessing quality checks.
- Feature importance ranking (Fisher Score).
- Confusion matrices for all classifiers.
- Hyperparameter optimization heatmaps.
