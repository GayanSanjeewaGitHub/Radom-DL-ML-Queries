import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Simulated predictions
y_true = np.array([0, 0, 1, 1])  # True labels
perfect_pred = np.array([0.1, 0.4, 0.9, 0.95])  # Perfect predictions
random_pred = np.array([0.5, 0.5, 0.5, 0.5])    # Random predictions
imperfect_pred = np.array([0.3, 0.6, 0.4, 0.8]) # Imperfect predictions

# Calculate ROC curves
perfect_fpr, perfect_tpr, _ = roc_curve(y_true, perfect_pred)
random_fpr, random_tpr, _ = roc_curve(y_true, random_pred)
imperfect_fpr, imperfect_tpr, _ = roc_curve(y_true, imperfect_pred)

# Calculate AUC scores
perfect_auc = auc(perfect_fpr, perfect_tpr)
random_auc = auc(random_fpr, random_tpr)
imperfect_auc = auc(imperfect_fpr, imperfect_tpr)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(perfect_fpr, perfect_tpr, label=f'Perfect Model (AUC = {perfect_auc:.2f})', color='green')
plt.plot(random_fpr, random_tpr, label=f'Random Model (AUC = {random_auc:.2f})', color='blue')
plt.plot(imperfect_fpr, imperfect_tpr, label=f'Imperfect Model (AUC = {imperfect_auc:.2f})', color='orange')
plt.plot([0, 1], [0, 1], 'r--', label='No Skill (AUC = 0.50)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves')
plt.legend()
plt.grid()
plt.show()
