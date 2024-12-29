# Import necessary libraries
from pycaret.classification import *

# Sample dataset
from sklearn.datasets import make_classification
import pandas as pd

# Create a dataset
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.7, 0.3], random_state=42)
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
print(data.head(5))
data['target'] = y

# Step 1: PyCaret Setup
clf_setup = setup(data=data, target='target', session_id=42, silent=True, log_experiment=False)

# Step 2: Create the KNN Model
knn_model = create_model('knn')

# Step 3: Hyperparameter Tuning for Precision
tuned_knn = tune_model(knn_model, optimize='Precision')

# Step 4: Evaluate the Tuned Model
evaluate_model(tuned_knn)

# Step 5: Save the Tuned Model
save_model(tuned_knn, 'tuned_knn_precision')
