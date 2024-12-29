# Import necessary libraries
from pycaret.classification import *

# Sample dataset
from sklearn.datasets import make_classification
import pandas as pd

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.9, 0.1], random_state=42)
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
data['target'] = y

# Start PyCaret setup
clf_setup = setup(data=data, target='target', session_id=42, silent=True, log_experiment=False)

# Compare models based on F1-score
best_model = compare_models(sort='F1')

# Print the best model
print(f"The best model based on F1-score is:\n{best_model}")
