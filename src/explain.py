import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

model = joblib.load("models/model.pkl")

df = pd.read_csv("data/ai4i2020.csv")

df.drop(
    [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ],
    axis=1,
    inplace=True
)
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])
X = df.sample(1000, random_state=42)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

print("SHAP Shape:", shap_values.shape)
failure_shap = shap_values[:, :, 1]

shap.summary_plot(
    failure_shap,
    X,
    show=False
)

plt.savefig(
    "models/shap_summary.png",
    bbox_inches="tight",
    dpi=300
)

plt.show()

print("SHAP summary saved to models/shap_summary.png")