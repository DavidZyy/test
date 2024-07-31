import numpy as np
import matplotlib.pyplot as plt

# 定义类别1和类别2的样本数据
X1 = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 5]])
X2 = np.array([[1, 0], [2, 1], [3, 1], [3, 2], [5, 3], [6, 5]])

# 步骤1：计算均值向量
m1 = np.mean(X1, axis=0, keepdims=True)
m2 = np.mean(X2, axis=0, keepdims=True)
# 步骤2：计算类内散布矩阵
S1 = (X1 - m1).T @ (X1 - m1)
S2 = (X2 - m2).T @ (X2 - m2)
Sw = S1 + S2
# 步骤3：计算 Fisher 判别投影向量
Sw_inv = np.linalg.inv(Sw)

w = Sw_inv @ (m1 - m2).T
# 步骤4：计算判别函数
def discriminant_function(x):
    return np.dot(w, x)

w_x = w[0, 0]
w_y = w[1, 0]

# 步骤5：绘制判别投影向量和样本点的投影坐标
plt.scatter(X1[:, 0], X1[:, 1], color='blue', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', label='Class 2')
plt.plot([0, 8*w_y], [0, -8*w_x], scalex=False, scaley=False, color='green', linestyle='--', label='Fisher Discriminant')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Projection of Samples using Fisher Discriminant')
plt.legend()
plt.grid()
# plt.grid(True, "both", "both", alpha=0.5)
ax = plt.gca()
ax.set_aspect(1)
plt.show()

# 步骤6：计算每个样本点投影后的坐标
# projection_X1 = [discriminant_function(x) for x in X1]
# projection_X2 = [discriminant_function(x) for x in X2]

projection_X1 = X1.dot(w)
projection_X2 = X2.dot(w)
#
print("类别1样本投影坐标:", projection_X1)
print("类别2样本投影坐标:", projection_X2)
#
# plt.figure()
# plt.scatter(projection_X1, np.zeros_like(projection_X1), label='Class 1', marker='o')
# plt.scatter(projection_X2, np.zeros_like(projection_X2), label='Class 2', marker='x')
# plt.xlabel('Projection onto Fisher Discriminant Vector')
# plt.legend()
# plt.title('Fisher Discriminant Analysis')
# plt.show()
#
# print("Fisher Discriminant Projection Vector (w):", w)
