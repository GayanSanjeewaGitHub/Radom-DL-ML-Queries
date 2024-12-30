from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a fully grown tree
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

# Print the tree structure
print("Full Tree:\n", export_text(full_tree, feature_names=[f"Feature{i}" for i in range(X.shape[1])]))

# Simulated subtree replacement: Manually limit depth of a subtree
pruned_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
pruned_tree.fit(X_train, y_train)

# Print the pruned tree structure
print("\nPruned Tree:\n", export_text(pruned_tree, feature_names=[f"Feature{i}" for i in range(X.shape[1])]))

# Evaluate both trees
y_pred_full = full_tree.predict(X_test)
y_pred_pruned = pruned_tree.predict(X_test)

print("\nAccuracy of Full Tree:", accuracy_score(y_test, y_pred_full))
print("Accuracy of Pruned Tree (Simulated Subtree Replacement):", accuracy_score(y_test, y_pred_pruned))
