from sklearn.metrics import confusion_matrix, classification_report

# Sample ground truth and predictions
y_true = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]  # Actual labels
y_pred = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # Predicted labels

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Print the confusion matrix details
print("Confusion Matrix:")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Not Spam", "Spam"]))
