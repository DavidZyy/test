from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

# 定义参数（可以根据需要修改）
L = 32  # 模型总层数
G = 12  # GPU 最大容量
k = 1.4   # GPU 加载速度是 CPU 计算速度的 k 倍
# k = 2   # GPU 加载速度是 CPU 计算速度的 k 倍
epsilon = 0.001  # n 的惩罚系数，确保 n 尽量小

# 创建整数规划问题，目标是最小化总时间
prob = LpProblem("Minimize_Total_Time", LpMinimize)

# 定义变量
n_max = L  # 最大可能的加载次数（上界估计）
# n_max = 100  # 最大可能的加载次数（上界估计）

g0 = LpVariable("g0", lowBound=0, upBound=G, cat="Integer")  # GPU 初始层数
g = [LpVariable(f"g_{i}", lowBound=0, upBound=G, cat="Integer") for i in range(n_max)]  # GPU 每次加载层数
c = [LpVariable(f"c_{i}", lowBound=0, upBound=L, cat="Integer") for i in range(n_max)]  # CPU 每次计算层数
t = [LpVariable(f"t_{i}", lowBound=0) for i in range(n_max)]  # 每次操作时间
n = LpVariable("n", lowBound=0, upBound=n_max, cat="Integer")  # 操作次数

# 目标函数：最小化总时间
prob += lpSum(t[i] for i in range(n_max)) + epsilon * n

# 约束条件
# 1. 总计算层数约束：g0 + sum(g_i) + sum(c_i) = L
prob += g0 + lpSum(g[i] for i in range(n_max)) + lpSum(c[i] for i in range(n_max)) == L

# 2. GPU 容量约束
prob += g0 <= G  # 初始层数约束
for i in range(n_max):
    prob += g0 + g[i] <= G  # 每次加载时的容量约束

# 3. 时间约束
for i in range(n_max):
    prob += t[i] >= g[i] * (1.0 / k)  # GPU 加载时间
    prob += t[i] >= c[i]      # CPU 计算时间

# 4. 只计算前 n 次操作（通过大 M 方法控制）
M = 1000  # 一个足够大的数
active = [LpVariable(f"active_{i}", cat="Binary") for i in range(n_max)]  # 二进制变量，表示该次操作是否激活
for i in range(n_max):
    # 如果 i < n，则 active[i] = 1；如果 i >= n，则 active[i] = 0
    prob += active[i] <= (n - i - 1 + M) / M  # 约束 active[i] 在 i < n 时为 1
    # 限制 g[i], c[i] 在未激活时为 0（通过 M 约束）
    prob += g[i] <= M * active[i]
    prob += c[i] <= M * active[i]
    # 未激活时 t[i] 不计入目标（这里通过目标函数已隐式处理）

# 求解问题
prob.solve()

# 输出结果
print(f"状态: {LpStatus[prob.status]}")
if LpStatus[prob.status] == "Optimal":
    print(f"总时间: {prob.objective.value()}")
    print(f"初始层数 g0: {g0.value()}")
    n_val = int(n.value())
    print(f"操作次数 n: {n_val}")
    for i in range(n_val):
    # for i in range(n_max):
        print(f"第 {i+1} 次操作: g_{i} = {g[i].value()}, c_{i} = {c[i].value()}, 时间 t_{i} = {t[i].value()}")
else:
    print("未找到最优解")
