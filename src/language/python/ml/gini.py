import numpy as np


def calculate_gini(labels):
    classes = np.unique(labels)
    gini = 1.0

    for c in classes:
        proportion = np.mean(labels == c)
        gini -= proportion ** 2

    return gini


def evaluate_split(dataset, feature_index, threshold):
    left = [sample for sample in dataset if sample[feature_index] <= threshold]
    right = [sample for sample in dataset if sample[feature_index] > threshold]

    left_labels = [sample[-1] for sample in left]
    right_labels = [sample[-1] for sample in right]

    left_proportion = len(left_labels) / len(dataset)
    right_proportion = len(right_labels) / len(dataset)

    left_gini = calculate_gini(left_labels)
    right_gini = calculate_gini(right_labels)

    weighted_gini = (left_proportion * left_gini) + (right_proportion * right_gini)

    return weighted_gini, left, right


def find_best_split(dataset):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    best_left = None
    best_right = None

    num_features = len(dataset[0]) - 1  # Exclude the last column (target)

    for feature_index in range(num_features):
        values = [sample[feature_index] for sample in dataset]
        unique_values = np.unique(values)

        for threshold in unique_values:
            weighted_gini, left, right = evaluate_split(dataset, feature_index, threshold)

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_index
                best_threshold = threshold
                best_left = left
                best_right = right

    return best_feature, best_threshold, best_left, best_right


# Step 1: Define the dataset
data = [
    [3, 2, 5, 1],
    [5, 7, 7, 0],
    [8, 6, 9, 1],
    [1, 3, 2, 0],
    [6, 4, 8, 1],
    [4, 5, 6, 0]
]

# Step 2: Calculate the Gini index for the root node
labels = [sample[-1] for sample in data]
root_gini = calculate_gini(labels)
print("Root Gini Index:", root_gini)

# Step 3: Find the best split (first decision rule)
best_feature, best_threshold, left, right = find_best_split(data)
print("First Decision Rule:")
print("Selected Feature:", best_feature)
print("Threshold:", best_threshold)
print("Left Subset:", left)
print("Right Subset:", right)
