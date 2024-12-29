import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score

# 1. Create a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Define a range of var_smoothing values to explore
param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}  # 100 values between 1 and 1e-9

# 4. Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()

# 5. Perform GridSearchCV to tune var_smoothing and optimize recall
gs = GridSearchCV(estimator=gnb, param_grid=param_grid, scoring='recall', cv=5, verbose=1)
gs.fit(X_train, y_train)

# 6. Display the best var_smoothing found by GridSearchCV
print(f"Best var_smoothing: {gs.best_params_['var_smoothing']}")

# 7. Plot recall values for different var_smoothing
recall_scores = gs.cv_results_['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(param_grid['var_smoothing'], recall_scores, marker='o', linestyle='-', color='b')
plt.xscale('log')
plt.xlabel('var_smoothing (log scale)')
plt.ylabel('Recall')
plt.title('Effect of var_smoothing on Recall')
plt.grid(True)
plt.show()

# 8. Evaluate the final model on the test set using the best var_smoothing
best_gnb = gs.best_estimator_
y_pred = best_gnb.predict(X_test)
test_recall = recall_score(y_test, y_pred)
print(f"Test Recall with best var_smoothing: {test_recall}")
