import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("data/ai4i2020.csv")

df.drop(["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1, inplace=True)

le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print(classification_report(y_test, preds))

joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully!")


import matplotlib.pyplot as plt
importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, importance)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.show()

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    model,
    X_scaled,
    y,
    cv=5,
    scoring="f1"
)

print("\nCross Validation F1 Scores:")
print(cv_scores)

print("\nAverage F1:")
print(cv_scores.mean())

from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\nROC AUC:", roc_auc)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, preds)

print("\nConfusion Matrix:")
print(cm)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(
    "models/roc_curve.png",
    bbox_inches="tight"
)

plt.show()