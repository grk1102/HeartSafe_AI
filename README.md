HeartSafe_AI
HeartSafe AI is an innovative AI-powered system designed to predict heart attack risk using a hybrid machine learning model combining Random Forest and XGBoost. It leverages SMOTE to handle imbalanced datasets and offers a user-friendly interface via a Streamlit dashboard or command-line inputs. The system uses the UCI Heart Disease dataset for training and prediction.
Features

Achieves ~91.5% accuracy in predicting heart attack risk.
Analyzes 13 health features: age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs), resting ECG (restecg), max heart rate (thalach), exercise angina (exang), ST depression (oldpeak), ST slope (slope), major vessels (ca), thalassemia (thal).
Outputs a risk probability (0 = low risk, 1 = high risk).
Handles class imbalance using SMOTE.
Provides an interactive Streamlit dashboard for inputting patient data and visualizing results.

Installation
Prerequisites

Python 3.7 or higher
Streamlit for dashboard
Dataset: Heart.csv (UCI Heart Disease dataset)

Dependencies

pandas
numpy
scikit-learn
xgboost
imbalanced-learn
joblib
streamlit

Setup

Clone the repository or download the project files.
Create and activate a virtual environment:python -m venv venv
.\venv\Scripts\activate  # Windows


Install dependencies:pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib streamlit


Ensure Heart.csv is in the project directory.
Train the model to save hybrid_model.joblib and scaler.joblib:python prototype_code.py


Launch the Streamlit dashboard:streamlit run dashboard.py



Usage

Training: Run prototype_code.py to train the model on Heart.csv, save the model/scaler, and test predictions via command-line inputs.
Dashboard: Access the Streamlit dashboard at http://localhost:8501 to input patient data and view risk predictions with visualizations.
Dataset: Uses the UCI Heart Disease dataset (Heart.csv, 303 rows, 13 features + binary target).

Notes

Ensure Heart.csv, hybrid_model.joblib, and scaler.joblib are in the project directory for the dashboard to work.
The model expects numeric inputs within specific ranges (e.g., age 29-77, cholesterol 126-564).
For deployment, push to GitHub and connect to Streamlit Cloud with requirements.txt.
