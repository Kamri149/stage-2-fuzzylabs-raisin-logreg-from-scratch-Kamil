import csv
import numpy as np


def load_csv(path: str, label_col: str, newline="", encoding="utf-8-sig", delimiter=",", **kwargs):
    """
    Loads a CSV file, processes the features and labels, and returns the feature matrix (X), 
    label array (y), feature names, and label-to-index mapping.

    Parameters:
    - path (str): The file path to the CSV file.
    - label_col (str): The name of the column containing the labels (for classification or regression tasks).
    - newline (str, optional): The newline character(s) used in the file (default is "").
    - encoding (str, optional): The encoding used in the file (default is "utf-8-sig").
    - delimiter (str, optional): The delimiter used in the CSV file (default is ",").
    - **kwargs: Additional arguments passed to `csv.DictReader`, such as quoting, doublequote, etc.

    Returns:
    - X (np.ndarray): A NumPy array of shape (n_samples, n_features) containing the feature values.
    - y (np.ndarray): A NumPy array of shape (n_samples,) containing the labels (could be numeric or categorical).
    - feature_names (list): A list of the column names used as features (excluding the label column).
    - mapping (dict, optional): A dictionary mapping the class labels to numeric values (only for classification tasks).

    Raises:
    - ValueError: If the CSV is empty, or if there are issues with label columns (e.g., non-unique labels in binary classification).

    Example:
    >>> X, y, feature_names, mapping = load_csv("data.csv", label_col="Label")
    >>> print(X.shape)  # (number_of_samples, number_of_features)
    >>> print(feature_names)  # ['Feature1', 'Feature2', ...]
    >>> print(mapping)  # {'Class1': 0, 'Class2': 1}
    """
    # Open the CSV file with the specified encoding and other options
    with open(path, "r", newline=newline, encoding=encoding, **kwargs) as f:
        # Create a CSV reader that returns each row as a dictionary with column names as keys
        reader = csv.DictReader(f, delimiter=delimiter)
        # Convert the reader into a list of rows (each row is a dictionary)
        rows = list(reader)

    # If there are no rows in the CSV, raise an error
    if not rows:
        raise ValueError("Empty CSV")

    # Extract the feature column names (exclude the label column)
    feature_names = [c for c in rows[0].keys() if c != label_col]

    # Convert each row's feature values to a float and store them in a NumPy array (X)
    X = np.array([[float(r[c]) for c in feature_names] for r in rows], dtype=np.float64)

    # Extract the label values from the specified label column (label_col)
    labels = [r[label_col] for r in rows]

    # Check if the labels are numeric or categorical and handle accordingly
    try:
        # Try converting labels to numeric (for regression or numeric classification)
        y = np.array([float(label) for label in labels], dtype=np.float64)
        mapping = None  # No mapping needed for continuous labels (regression tasks)
    except ValueError:
        # Handle categorical labels deterministically (for classification)
        classes = sorted(set(labels))  # Get unique classes
        if len(classes) < 2:
            raise ValueError(f"Expected at least 2 unique classes for classification, got {len(classes)}.")
        mapping = {classes[i]: i for i in range(len(classes))}
        y = np.array([mapping[label] for label in labels], dtype=np.int64)

    return X, y, feature_names, mapping