# ==============================
# END-TO-END LOAN MODEL PIPELINE
# Random Forest Classifier
# ==============================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


DATA_PATH = "loan_approval_dataset.csv"
MODEL_PATH = "loan_random_forest_pipeline.pkl"


# ------------------------------
# 1. LOAD DATA
# ------------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

for col in ["loan_status", "education", "self_employed"]:
  df[col] = df[col].str.strip()

# ------------------------------
# 2. DEFINE FEATURES & TARGET
# ------------------------------
X = df.drop(columns="loan_status")
y = df["loan_status"].map({"Approved": 1, "Rejected": 0})  # binary target


# ------------------------------
# 3. FEATURE GROUPS
# ------------------------------
numeric_features = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score"
]

categorical_features = [
    "education",
    "self_employed"
]


# ------------------------------
# 4. PREPROCESSING
# ------------------------------
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ------------------------------
# 5. MODEL
# ------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=27,
    n_jobs=-1
)


# ------------------------------
# 6. FULL PIPELINE
# ------------------------------
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", model)
])


# ------------------------------
# 7. TRAIN / TEST SPLIT
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ------------------------------
# 8. TRAIN
# ------------------------------
pipeline.fit(X_train, y_train)


# ------------------------------
# 9. EVALUATION
# ------------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ------------------------------
# 10. SAVE PIPELINE
# ------------------------------
joblib.dump(pipeline, MODEL_PATH)


# ------------------------------
# 11. EXAMPLE INFERENCE
# ------------------------------
sample = pd.DataFrame([{
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 9600000,
    "loan_amount": 29900000,
    "loan_term": 12,
    "cibil_score": 778
}])

prediction = pipeline.predict(sample)[0]
probability = pipeline.predict_proba(sample)[0][1]

print("\nSample Prediction:", "Approved" if prediction == 1 else "Rejected")
print("Approval Probability:", probability)
