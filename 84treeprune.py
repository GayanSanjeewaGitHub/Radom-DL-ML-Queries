from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=10000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pre-pruned Decision Tree
pre_pruned_tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=42)
pre_pruned_tree.fit(X_train, y_train)

# Evaluate pre-pruned tree
y_pred = pre_pruned_tree.predict(X_test)
print("Pre-Pruned Tree Accuracy:", accuracy_score(y_test, y_pred))



# Fully grown Decision Tree
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

# Check accuracy of fully grown tree
y_pred_full = full_tree.predict(X_test)
print("Fully Grown Tree Accuracy:", accuracy_score(y_test, y_pred_full))

# Simulated Subtree Replacement: Replace a redundant subtree
# In practice, this requires manual evaluation or specialized libraries.
simplified_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
simplified_tree.fit(X_train, y_train)

# Evaluate simplified tree (as if a subtree was replaced)
y_pred_simplified = simplified_tree.predict(X_test)
print("Simplified Tree Accuracy (Simulated Subtree Replacement):", accuracy_score(y_test, y_pred_simplified))


#Backward Pruning Example
# Simulating backward pruning by pruning a fully grown tree using cross-validation.

from sklearn.model_selection import cross_val_score
import numpy as np

# Function to prune tree by limiting depth
def prune_tree(X_train, y_train, X_test, y_test, max_depth_values):
    best_depth = None
    best_score = 0
    
    for depth in max_depth_values:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(tree, X_train, y_train, cv=5)
        avg_score = np.mean(scores)
        
        if avg_score > best_score:
            best_score = avg_score
            best_depth = depth
    
    # Train with the best depth
    pruned_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    pruned_tree.fit(X_train, y_train)
    return pruned_tree

# Prune the tree using backward pruning
pruned_tree = prune_tree(X_train, y_train, X_test, y_test, max_depth_values=range(1, 11))

# Evaluate the pruned tree
y_pred_pruned = pruned_tree.predict(X_test)
print("Backward Pruned Tree Accuracy:", accuracy_score(y_test, y_pred_pruned))
