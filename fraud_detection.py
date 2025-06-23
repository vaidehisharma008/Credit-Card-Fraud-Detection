# fraud_detection.py

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 2. Load Dataset
# 🔁 Replace this with the correct path to your CSV file
# Best and cleanest

df = pd.read_csv("C:/Users/Vaidehi Sharma/Downloads/pythonmlproject/archive (1).zip")



# 3. Preprocess Data
scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
df.drop(["Amount", "Time"], axis=1, inplace=True)

# 4. Define Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 6. Apply SMOTE for Class Balancing
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# 7. Train XGBoost Classifier
model = XGBClassifier(eval_metric='logloss')

model.fit(X_resampled, y_resampled)

# 8. Evaluate Model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 9. Print Evaluation Metrics
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

print(f"\n🏁 ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# 10. Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="XGBoost")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# 11. Simulate Real-Time Detection
import time
print("\n🔍 Real-Time Transaction Simulation:")
sample_stream = X_test.sample(5)
for i, row in sample_stream.iterrows():
    proba = model.predict_proba([row])[0][1]
    label = "Fraud" if proba > 0.5 else "Legit"
    print(f"Transaction {i}: {label} (Score: {proba:.2f})")
    time.sleep(1)
