# Kaggle-shl-grammar-scoring: Speech Grammar Scoring Engine

This repository contains the machine learning pipeline developed for the **SHL Hiring Assessment 2026** competition. The task focuses on predicting continuous language proficiency/grammar Mean Opinion Scores (MOS) from spoken English speech recordings.

## Project Overview
- **Objective**: Automate speech grammar evaluation, rating conversational English capability on a continuous scale between `1.0` and `5.0`.
- **Challenge**: The official training data archive suffered from zip truncation, leaving exactly **763 verified and readable** audio files out of 769 listed in the metadata. A directory discrepancy was resolved by matching physical file directories directly.
- **Modeling Strategy**: A 5-Fold Stratified Cross-Validation framework was utilized to train models using robust hand-crafted acoustic representations.

## Pipeline & Architecture
1. **Acoustic Preprocessing**: Extracts 35 acoustic elements from 16kHz mono `.wav` recordings utilizing `librosa`:
   - Spectral Centroid, Spectral Rolloff, RMS Energy, and Zero Crossing Rate (means & standard deviations).
   - 13-dimensional Mel-Frequency Cepstral Coefficients (MFCCs) tracking phonetic stability and accent distribution.
2. **Validation Strategy**: Stratified 5-Fold cross-validation binned on continuous targets to ensure uniform target distributions across subsets.
3. **Regression Modeling**: Standard Scaled continuous features fitted across ensembles including Ridge Regression, HistGradientBoosting, and Random Forest Regressors.

## Local Cross-Validation Performance
| Model | OOF RMSE | OOF Pearson Correlation |
| :--- | :---: | :---: |
| **Ridge Regression (Baseline)** | 0.8544 | 0.7257 |
| **HistGradientBoosting** | 0.7916 | 0.7709 |
| **Random Forest Regressor (OOF Ensemble)** | **0.7909** | **0.7713** |

## Repository Structure
```text
Kaggle-shl-grammar-scoring/
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── extract_features.py       # Modular acoustic feature utility
│   ├── train_model.py           # 5-fold Stratified RF Regressor pipeline
│   └── predict.py               # Test-inference engine
```

## Getting Started

### Installation
Ensure you have python installed, and run:
```bash
pip install -r requirements.txt
```

### Usage
To extract features and evaluate models using modular utilities:
```python
from src.extract_features import extract_acoustic_features
from src.train_model import train_cv_pipeline

# Example feature extraction
features = extract_acoustic_features("path/to/audio.wav")
```

## Licensing & Compliance
All official competition files, raw audio tracks, metadata sheets (`train.csv`, `test.csv`), and submission documents have been deliberately omitted via `.gitignore` to comply with the Kaggle rules and competition distribution protocols.

## Interactive Pipeline Notebook
You can run and explore the entire model pipeline interactively:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pavan-Reddy93/Kaggle-shl-grammar-scoring/blob/main/notebooks/shl_grammar_scoring_pipeline.ipynb)
