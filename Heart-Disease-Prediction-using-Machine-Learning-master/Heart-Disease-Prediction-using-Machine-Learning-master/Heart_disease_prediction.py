
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

warnings.filterwarnings('ignore')

# Check files in current directory
print("Files in directory:", os.listdir())

# Load dataset
dataset = pd.read_csv("heart.csv")
print("Dataset shape:", dataset.shape)
print(dataset.head())

# Define features and target
X = dataset.drop("target", axis=1)
y = dataset["target"]

# === Preprocessing ===
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Feature Extraction: PCA ===
pca = PCA(n_components=10)  # reduce to 10 features
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0)

# === Model Training and Evaluation ===
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM (Linear)": svm.SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0)
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    results[name] = acc
    predictions[name] = Y_pred
    print(f"{name} Accuracy: {round(acc * 100, 2)}%")

# Best Model Selection
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n\u2705 Best Performing Model: {best_model_name} with Accuracy = {round(results[best_model_name]*100, 2)}%")
print("\n=== Classification Report ===")
print(classification_report(Y_test, predictions[best_model_name]))

# Accuracy Comparison Plot
df_results = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy (%)": [round(acc * 100, 2) for acc in results.values()]
})
df_results = df_results.sort_values(by="Accuracy (%)", ascending=False)

sns.set(rc={'figure.figsize': (10, 6)})
sns.barplot(x="Model", y="Accuracy (%)", data=df_results, palette="mako")
plt.title("Model Accuracy Comparison (With PCA)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
# Save best model and PCA pipeline
with open('heart_disease_best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('heart_disease_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('heart_disease_pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

print("\n\ud83d\udcc0 Best model, scaler, and PCA transformer saved.")