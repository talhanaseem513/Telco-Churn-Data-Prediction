import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Clean total charges (some blanks)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", pd.NA), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    # Drop customerID – an identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    # Convert target to 0/1
    df["Churn"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)
    return df

def split_features(df: pd.DataFrame):
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    # Identify dtypes
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, y, numeric_cols, categorical_cols

def build_pipeline(numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            # numeric passthrough (no scaling needed for tree models)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
    return pipe, preprocessor

def main():
    df = load_data()
    X, y, num_cols, cat_cols = split_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe, preprocessor = build_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)

    # Evaluate
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

    # Persist artifacts
    joblib.dump(pipe.named_steps["prep"], ART_DIR / "preprocessor.pkl")
    joblib.dump(pipe.named_steps["clf"], ART_DIR / "model.pkl")

    # Store feature names in original order for the app UI
    feature_names = X.columns.tolist()
    with open(ART_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    print("\nArtifacts saved to ./artifacts")

if __name__ == "__main__":
    main()
