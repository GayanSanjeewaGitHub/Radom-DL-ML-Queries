import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create a dataset with features of varied magnitudes
data = {
    'Feature1': np.random.uniform(1, 10, 100),
    'Feature2': np.random.uniform(1000, 10000, 100),
    'Target': np.random.uniform(50, 100, 100)
}

df = pd.DataFrame(data)
X = df[['Feature1', 'Feature2']].values  # Convert to NumPy array
y = df['Target'].values  # Convert to NumPy array

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Without scaling: Train an SVR model
model_no_scaling = SVR(kernel='linear')
model_no_scaling.fit(X_train, y_train)

# Predict and evaluate the model without scaling
y_pred_no_scaling = model_no_scaling.predict(X_test)
mse_no_scaling = mean_squared_error(y_test, y_pred_no_scaling)
print(f"Mean Squared Error without scaling: {mse_no_scaling}")

# With scaling: Standardize the features and train an SVR model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_with_scaling = SVR(kernel='linear')
model_with_scaling.fit(X_train_scaled, y_train)

# Predict and evaluate the model with scaling
y_pred_with_scaling = model_with_scaling.predict(X_test_scaled)
mse_with_scaling = mean_squared_error(y_test, y_pred_with_scaling)
print(f"Mean Squared Error with scaling: {mse_with_scaling}")
