from sklearn.metrics import r2_score
import numpy as np

# Actual data
y_true = [3, -0.5, 2, 7, 4.2]
# Predicted data (good model)
y_pred_good = [2.5, 0.0, 2, 8, 4.1]
# Predicted data (poor model, worse than the mean)
y_pred_bad = [10, 10, 10, 10, 10]

# Calculate R^2
r2_good = r2_score(y_true, y_pred_good)  # Should be positive and close to 1
r2_bad = r2_score(y_true, y_pred_bad)    # Should be negative

print("R² (Good Model):", r2_good)
print("R² (Bad Model):", r2_bad)

# R² (Good Model): 0.948
# R² (Bad Model): -14.0
