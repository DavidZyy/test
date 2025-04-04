from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

# 参数
L = 10  # 总层数
G = 4   # GPU 容量
k = 2   # GPU 加载速度倍数

# 创建问题
prob = LpProblem("Minimize_Total_Time", LpMinimize)

# 变量
g_b = LpVariable("g_b", lowBound=0, upBound=G, cat="Integer")  # GPU 初始层数
g = LpVariable("g", lowBound=0, upBound=G, cat="Integer")  # 每次 GPU 加载层数
c = LpVariable("c", lowBound=0, upBound=L, cat="Integer")  # 每次 CPU 计算层数
n = LpVariable("n", lowBound=0, upBound=L, cat="Integer")  # 重复次数
g_e = LpVariable("g_e", lowBound=0, upBound=G ,cat="Integer")  # 最后一次 GPU 加载层数
c_e = LpVariable("c_e", lowBound=0, upBound=L, cat="Integer")  # 最后一次 CPU 计算层数
t = LpVariable("t", lowBound=0)  # 每次操作时间
t_e = LpVariable("t_e", lowBound=0)  # 最后一次操作时间

# 目标函数：最小化 T = n * t + t_e
prob += n * t + t_e

# 约束
# 1. 总计算层数约束
prob += g_b + n * (g + c) + g_e + c_e == L

# 2. GPU 容量约束
prob += g_b <= G
prob += g_b + g <= G
prob += g_b + g_e <= G

# 3. 时间约束
prob += t >= g / k
prob += t >= c
prob += t_e >= g_e / k
prob += t_e >= c_e

# 求解
prob.solve()

# 输出
print(f"状态: {LpStatus[prob.status]}")
if LpStatus[prob.status] == "Optimal":
    print(f"总时间: {prob.objective.value()}")
    print(f"初始层数 g_b: {g_b.value()}")
    print(f"重复次数 n: {n.value()}")
    print(f"每次操作: g = {g.value()}, c = {c.value()}, 时间 t = {t.value()}")
    print(f"最后一次操作: g_e = {g_e.value()}, c_e = {c_e.value()}, 时间 t_e = {t_e.value()}")
    # 验证总层数
    total_layers = g_b.value() + n.value() * (g.value() + c.value()) + g_e.value() + c_e.value()
    print(f"总计算层数: {total_layers} (应等于 {L})")
else:
    print("未找到最优解")
