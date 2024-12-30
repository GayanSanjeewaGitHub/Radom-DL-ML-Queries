import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get predicted probabilities for positive class
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Default threshold (0.5)
y_pred_default = (y_pred_proba >= 0.5).astype(int)
print("Classification Report (Threshold = 0.5):")
print(classification_report(y_test, y_pred_default))

# Custom threshold (e.g., 0.7 - stricter for positives)
y_pred_custom = (y_pred_proba >= 0.7).astype(int)
print("\nClassification Report (Threshold = 0.7):")
print(classification_report(y_test, y_pred_custom))
