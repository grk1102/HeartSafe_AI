## HeartSafe_AI

HeartSafe AI is an innovative AI-powered system designed to predict heart attack risk using a hybrid machine learning model combining Random Forest and XGBoost. It leverages SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced datasets and offers a user-friendly interface via Google Colab. This project aims to assist healthcare professionals and individuals in early risk assessment based on 13 health features.

## Features
- Achieves 91.54% accuracy in predicting heart attack risk.
- Analyzes 13 health features including age, cholesterol (`chol`), resting blood pressure (`trestbps`), ECG metrics (`oldpeak`, `slope`), number of blocked vessels (`ca`), and thalassemia (`thal`).
- Outputs a risk probability (e.g., 0.04 for low risk, closer to 1.0 for high risk).
- Handles class imbalance using SMOTE for improved minority class prediction.
- Provides an interactive interface for inputting patient data and visualizing results.

## Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or Google Colab environment

### Dependencies
- `imbalanced-learn` (for SMOTE)
- `numpy`
- `scipy`
- `scikit-learn` (for Random Forest and XGBoost)
