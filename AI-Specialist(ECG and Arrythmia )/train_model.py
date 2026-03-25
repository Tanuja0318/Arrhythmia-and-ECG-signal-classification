import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Feature extraction
def extract_features(signal):
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0.6:
            peaks.append(i)

    if len(peaks) < 2:
        return [0,0,0]

    rr = np.diff(peaks)
    return [np.mean(rr), np.std(rr), len(peaks)]


# Load datasets
files = {
    "normal_rhythm.csv": 0,
    "no_arrhythmia_ecg.csv": 0,
    "arrhythmia_irregular.csv": 1,
    "tachycardia_fast_rate.csv": 1,
    "bradycardia_slow_rate.csv": 1
}

X, y = [], []

for file, label in files.items():
    df = pd.read_csv(file)
    signal = df["ecg"].values
    X.append(extract_features(signal))
    y.append(label)

X = np.array(X)
y = np.array(y)

# Models
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

accuracies = {}

# Train + evaluate
for name, model in models.items():
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    accuracies[name] = acc
    print(f"{name}: {acc}")

# Save best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

os.makedirs("models", exist_ok=True)
pickle.dump(best_model, open("models/model.pkl", "wb"))

# Plot graph
plt.bar(accuracies.keys(), accuracies.values())
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.savefig("models/accuracy.png")

print(f"Best model: {best_model_name}")