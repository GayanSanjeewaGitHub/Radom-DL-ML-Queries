import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Step 1: Generate a synthetic dataset
X, y = make_classification(
    n_samples=500,  # Number of samples
    n_features=10,  # Total number of features
    n_informative=8,  # Number of informative features
    n_redundant=2,  # Number of redundant features
    n_classes=2,  # Binary classification
    random_state=42  # Reproducibility
)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize the KNN model
knn = KNeighborsClassifier()

# Step 4: Define the hyperparameter grid
param_grid = {'n_neighbors': np.arange(5, 25)}  # Test values for n_neighbors from 5 to 24

# Step 5: Perform GridSearchCV
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
#5 cross validations are happening 
#accuracy used as the matric for the validation

# Step 6: Print the best parameters and evaluate the model
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Score:", grid_search.best_score_)

# Step 7: Test the model with the best hyperparameters
best_knn = grid_search.best_estimator_  # Get the model with the best parameters
y_pred = best_knn.predict(X_test)

# Step 8: Evaluate on the test set
print("\nClassification Report:")
print(classification_report(y_test, y_pred))