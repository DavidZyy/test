import numpy as np
from cvxopt import matrix, solvers

# 数据点和标签
X = np.array([[4, 4], [5, 4], [2, 2]])
y = np.array([1, 1, -1])

# 内积矩阵
K = np.dot(X, X.T)
P = matrix(np.outer(y, y) * K)
q = matrix(np.ones(X.shape[0]) * -1)
G = matrix(np.diag(np.ones(X.shape[0]) * -1))
h = matrix(np.zeros(X.shape[0]))
A = matrix(y, (1, X.shape[0]), 'd')
b = matrix(0.0)

# 求解二次规划问题
solution = solvers.qp(P, q, G, h, A, b)
alphas = np.ravel(solution['x'])

# 计算权重向量 w
w = np.sum(alphas * y[:, None] * X, axis=0)

# 计算偏置 b
# 选择支持向量（alpha > 1e-5 的点）
sv = alphas > 1e-5
b = y[sv] - np.dot(X[sv], w)
b = b[0]

print("Support vectors:\n", X[sv])
print("w:", w)
print("b:", b)
