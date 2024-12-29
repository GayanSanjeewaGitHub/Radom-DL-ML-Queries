from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# True values
y_true = [100, 200, 300, 400, 100000]  # Outlier in last value
# Predicted values
y_pred = [110, 190, 310, 390, 90000]

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("MSE:", mse)       # Heavily influenced by the outlier
print("MAE:", mae)       # Less influenced by the outlier
print("RMSE:", rmse)     # Similar to MSE but on the original scale
