from pulp import LpProblem, LpMinimize, LpVariable, LpStatus

# 参数
L = 32  # 总层数
G = 12   # GPU 容量
k = 1   # GPU 加载速度倍数

# 枚举 n 的范围
n_max = L  # 最大可能的重复次数
best_T = float('inf')  # 记录最优总时间
best_solution = None   # 记录最优解

# 枚举 n 从 0 到 n_max
for n_val in range(n_max + 1):
    # 创建线性问题
    prob = LpProblem(f"Minimize_Total_Time_n_{n_val}", LpMinimize)

    g_b = LpVariable("g_b", lowBound=0, upBound=G, cat="Integer")  # GPU 初始层数
    g = LpVariable("g", lowBound=0, upBound=G, cat="Integer")  # 每次 GPU 加载层数
    c = LpVariable("c", lowBound=0, upBound=L, cat="Integer")  # 每次 CPU 计算层数
    g_e = LpVariable("g_e", lowBound=0, upBound=G ,cat="Integer")  # 最后一次 GPU 加载层数
    c_e = LpVariable("c_e", lowBound=0, upBound=L, cat="Integer")  # 最后一次 CPU 计算层数
    t = LpVariable("t", lowBound=0)  # 每次操作时间
    t_e = LpVariable("t_e", lowBound=0)  # 最后一次操作时间

    # 目标函数：T = n_val * t + t_e（n_val 是常数）
    prob += n_val * t + t_e
    # prob += n_val * t + t_e - 0.001 * g_b

    # 约束
    # 1. 总计算层数约束
    prob += g_b + n_val * (g + c) + g_e + c_e == L

    # 2. GPU 容量约束
    prob += g_b <= G
    prob += g_b + g <= G
    prob += g_b + g_e <= G

    # 3. 时间约束
    prob += t >= g * (1.0 / k)
    prob += t >= c
    prob += t_e >= g_e * (1.0 / k)
    prob += t_e >= c_e

    # 求解
    prob.solve()

    # 检查结果
    if LpStatus[prob.status] == "Optimal":
        T = prob.objective.value()
        if T < best_T:
            best_T = T
            best_solution = {
                "n": n_val,
                "g_b": g_b.value(),
                "g": g.value(),
                "c": c.value(),
                "t": t.value(),
                "g_e": g_e.value(),
                "c_e": c_e.value(),
                "t_e": t_e.value(),
                "T": T
            }

# 输出最优解
if best_solution:
    print("找到最优解:")
    print(f"总时间: {best_solution['T']}")
    print(f"初始层数 g_b: {best_solution['g_b']}")
    print(f"重复次数 n: {best_solution['n']}")
    print(f"每次操作: g = {best_solution['g']}, c = {best_solution['c']}, 时间 t = {best_solution['t']}")
    print(f"最后一次操作: g_e = {best_solution['g_e']}, c_e = {best_solution['c_e']}, 时间 t_e = {best_solution['t_e']}")
    # 验证总层数
    total_layers = (best_solution['g_b'] + 
                    best_solution['n'] * (best_solution['g'] + best_solution['c']) + 
                    best_solution['g_e'] + best_solution['c_e'])
    print(f"总计算层数: {total_layers} (应等于 {L})")
else:
    print("未找到可行解")
