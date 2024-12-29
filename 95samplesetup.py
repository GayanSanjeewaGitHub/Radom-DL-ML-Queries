from pycaret.classification import *

# Example dataset
from sklearn.datasets import make_classification
import pandas as pd

# Create a dataset
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.7, 0.3], random_state=42)
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
data['Loan_Status'] = y  # Target variable

# Setup with all possible parameters
clf_setup = setup(
    data=data,                      # The dataset
    target='Loan_Status',           # The target column
    train_size=0.7,                 # Proportion of the dataset for training
    test_data=None,                 # Optionally provide a test set
    session_id=123,                 # Random seed for reproducibility
    sampling=True,                  # Enable/disable random under-sampling/over-sampling
    categorical_features=None,      # List of categorical features
    categorical_imputation='mode',  # Imputation method for categorical features
    ordinal_features=None,          # Ordinal feature mappings
    high_cardinality_features=None, # Features to consider as high cardinality
    high_cardinality_method='frequency', # Method to encode high cardinality features
    numeric_features=None,          # List of numeric features
    numeric_imputation='mean',      # Imputation method for numeric features
    date_features=None,             # Date features to parse
    ignore_features=None,           # Features to ignore during modeling
    normalize=False,                # Normalize features
    normalize_method='zscore',      # Normalization method
    transformation=False,           # Apply transformation to features
    transformation_method='yeo-johnson', # Transformation method
    handle_unknown_categorical=True, # Handle unknown categorical levels
    unknown_categorical_method='least_frequent', # Method for unknown categorical levels
    pca=False,                      # Enable PCA
    pca_method='linear',            # PCA method
    pca_components=None,            # Number of components to keep in PCA
    ignore_low_variance=False,      # Ignore features with low variance
    combine_rare_levels=False,      # Combine rare levels in categorical features
    rare_level_threshold=0.1,       # Threshold for rare levels
    bin_numeric_features=None,      # List of numeric features to bin
    remove_outliers=False,          # Remove outliers
    outliers_threshold=0.05,        # Proportion of outliers to remove
    remove_multicollinearity=False, # Remove correlated features
    multicollinearity_threshold=0.9,# Threshold for correlation
    create_clusters=False,          # Create clusters
    cluster_iter=20,                # Number of iterations for clustering
    polynomial_features=False,      # Create polynomial features
    polynomial_degree=2,            # Degree for polynomial features
    trigonometric_features=False,   # Create trigonometric features
    group_features=None,            # Group features based on domain knowledge
    group_names=None,               # Names for grouped features
    feature_selection=False,        # Enable feature selection
    feature_selection_threshold=0.8,# Threshold for feature selection
    feature_interaction=False,      # Create feature interactions
    feature_ratio=False,            # Create feature ratios
    fix_imbalance=False,            # Fix class imbalance
    fix_imbalance_method=None,      # Method for fixing imbalance
    data_split_shuffle=True,        # Shuffle data before splitting
    data_split_stratify=True,       # Stratify data during split
    fold_strategy='kfold',          # Cross-validation strategy
    fold=10,                        # Number of folds for CV
    fold_shuffle=False,             # Shuffle folds
    fold_groups=None,               # Grouping for CV
    custom_pipeline=None,           # Custom transformers
    html=True,                      # Generate HTML reports
    verbose=True,                   # Display output logs
    profile=False                   # Profile the dataset
)
