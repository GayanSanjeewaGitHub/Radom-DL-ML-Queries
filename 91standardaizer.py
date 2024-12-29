import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create a sample dataset with features of varied magnitudes
# Feature 1 has values in the range of 0-10, and Feature 2 in the range of 0-10,000
data = {
    'Feature1': np.random.uniform(1, 10, 100000),
    'Feature2': np.random.uniform(1000, 10000, 100000),
    'Target': np.random.uniform(50, 100, 100000)
}

df = pd.DataFrame(data)

# Separate features and target
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Without Standardization: Train a linear regression model
model_no_scaling = LinearRegression()
model_no_scaling.fit(X_train, y_train)

# Predict and evaluate the model without scaling
y_pred_no_scaling = model_no_scaling.predict(X_test)
mse_no_scaling = mean_squared_error(y_test, y_pred_no_scaling)
print(f"Mean Squared Error without scaling: {mse_no_scaling}")

# Standardize the features to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# With Standardization: Train a linear regression model
model_with_scaling = LinearRegression()
model_with_scaling.fit(X_train_scaled, y_train)

# Predict and evaluate the model with scaling
y_pred_with_scaling = model_with_scaling.predict(X_test_scaled)
mse_with_scaling = mean_squared_error(y_test, y_pred_with_scaling)
print(f"Mean Squared Error with scaling: {mse_with_scaling}")

# Observe the differences in MSE to understand the impact of standardization.
# Larger magnitudes in features can bias the model if not scaled properly.
