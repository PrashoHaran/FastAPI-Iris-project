# train.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def main():
    # 1) Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Pipeline: scaler + logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])

    # 4) Train
    pipeline.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.3f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

    # 6) Save model bundle
    os.makedirs("models", exist_ok=True)
    bundle = {
        "model": pipeline,
        "class_names": iris.target_names,
        "feature_names": iris.feature_names,
        "metrics": {"test_accuracy": float(acc)}
    }
    joblib.dump(bundle, "models/model.pkl")
    print("Saved: models/model.pkl")

if __name__ == "__main__":
    main()
