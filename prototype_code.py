# ---- Import Libraries ----
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from joblib import dump
import warnings

warnings.filterwarnings("ignore")

# ---- Load Dataset ----
file_path = "Heart.csv"  # Adjust if not in same directory
df = pd.read_csv(file_path, encoding='utf-8-sig')
print(df.head())

# ---- Separate Features and Target ----
X = df.drop(columns=['target'])
y = df['target']

# ---- Encode Categorical Columns ----
# Note: UCI dataset features are mostly numeric, but ensure any categorical are encoded
categorical_cols = X.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# ---- Fill Missing Values ----
X = X.fillna(X.mean())

# ---- Feature Scaling ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Handle Class Imbalance ----
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# ---- Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42)

# ---- Initialize Models ----
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                    subsample=0.9, colsample_bytree=0.9, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

# ---- Create Hybrid Voting Model ----
hybrid_model = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf)], voting='soft')

# ---- Train the Model ----
hybrid_model.fit(X_train, y_train)

# ---- Save Model and Scaler ----
dump(hybrid_model, "hybrid_model.joblib")
dump(scaler, "scaler.joblib")

# ---- Predictions ----
y_pred = hybrid_model.predict(X_test)

# ---- Evaluation ----
accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", round(accuracy, 4))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
cv_scores = cross_val_score(hybrid_model, X_resampled, y_resampled, cv=5)
print("📈 Cross-Validated Accuracy: {:.2f} ± {:.2f}".format(cv_scores.mean(), cv_scores.std()))

# ---- Interactive Prediction (Optional for Testing) ----
print("\n🩺 Please provide the following health details for prediction:")
input_prompts = {
    'age': "Age (29-77, e.g., 45): ",
    'sex': "Sex (0 = Female, 1 = Male): ",
    'cp': "Chest Pain Type (0 = typical angina, 1 = atypical, 2 = non-anginal, 3 = asymptomatic): ",
    'trestbps': "Resting Blood Pressure in mm Hg (94-200, e.g., 120): ",
    'chol': "Serum Cholesterol in mg/dL (126-564, e.g., 200): ",
    'fbs': "Fasting Blood Sugar > 120 mg/dL? (0 = No, 1 = Yes): ",
    'restecg': "Resting ECG Results (0 = Normal, 1 = ST-T abnormality, 2 = LVH): ",
    'thalach': "Maximum Heart Rate Achieved (71-202, e.g., 150): ",
    'exang': "Exercise Induced Angina? (0 = No, 1 = Yes): ",
    'oldpeak': "ST Depression (0-6.2, e.g., 1.4): ",
    'slope': "Slope of ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping): ",
    'ca': "Major Vessels Colored (0-3, e.g., 0): ",
    'thal': "Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect): "
}
user_input = []
for feature, prompt in input_prompts.items():
    while True:
        try:
            val = float(input(prompt))
            user_input.append(val)
            break
        except ValueError:
            print(f"⚠️ Invalid input for {feature}. Please enter a numeric value.")
user_df = pd.DataFrame([user_input], columns=input_prompts.keys())
user_df_scaled = scaler.transform(user_df)
prediction = hybrid_model.predict(user_df_scaled)[0]
probability = hybrid_model.predict_proba(user_df_scaled)[0][1]
print("\n🧠 Prediction:", "💔 High Risk of Heart Disease" if prediction == 1 else "❤️ Low Risk of Heart Disease")
print(f"📊 Risk Probability: {probability:.2f}")