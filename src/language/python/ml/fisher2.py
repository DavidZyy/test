import numpy as np
import matplotlib.pyplot as plt

# Define class 1 and class 2 samples
class1_samples = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 5]])
class2_samples = np.array([[1, 0], [2, 1], [3, 1], [3, 2], [5, 3], [6, 5]])

# Calculate mean vectors
m1 = np.mean(class1_samples, axis=0)
m2 = np.mean(class2_samples, axis=0)

# Calculate within-class covariance matrices
S1 = np.cov(class1_samples.T)
S2 = np.cov(class2_samples.T)

# Calculate between-class scatter matrix
SB = np.outer((m1 - m2), (m1 - m2))

# Calculate within-class scatter matrix
SW = S1 + S2

# Compute Fisher discriminant projection vector
w = np.linalg.inv(SW).dot(m1 - m2)

# Project samples onto the Fisher discriminant projection vector
projected_class1 = class1_samples.dot(w)
projected_class2 = class2_samples.dot(w)

# Plot the projected points
plt.figure()
plt.scatter(projected_class1, np.zeros_like(projected_class1), label='Class 1', marker='o')
plt.scatter(projected_class2, np.zeros_like(projected_class2), label='Class 2', marker='x')
plt.xlabel('Projection onto Fisher Discriminant Vector')
plt.legend()
plt.title('Fisher Discriminant Analysis')
plt.show()

print("Fisher Discriminant Projection Vector (w):", w)
