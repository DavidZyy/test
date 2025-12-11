import sys

def solve_min_cumulative_sum(A, k):
    """
    给定一个数字序列 A 和一个整数 k，计算选择 k 个数字的最小累计值总和。
    同时返回达到该最小和所选择的数字的 (1-based) 索引。

    Args:
        A (list[int]): 数字序列 (a_1, a_2, ...)
        k (int): 需要选择的数字数量

    Returns:
        tuple: (min_sum, selected_indices)
               min_sum (float): 最小的累计值总和 (如果无解则为 inf)
               selected_indices (list[int]): 达到最小和所选择的数字的 1-based 索引
    """
    n = len(A)

    if k == 0:
        return 0, []
    if k > n:
        return float('inf'), []

    # --- 1. 预计算 (使用 1-based 索引使逻辑更清晰) ---
    # P 和 PP 数组大小为 n+1，索引 0 留空 (值为0)
    # P[i] = A[0] + ... + A[i-1] (即 a_1 ... a_i 的和)
    # PP[i] = P[1] + ... + P[i]
    
    P = [0] * (n + 1)
    PP = [0] * (n + 1)
    
    for i in range(1, n + 1):
        P[i] = P[i - 1] + A[i - 1]
        
    for i in range(1, n + 1):
        PP[i] = PP[i - 1] + P[i]

    # --- 2. DP 初始化 ---
    # dp[i][j]: 考虑前 i 个数字 (a_1...a_i)，选择 j 个的最小成本
    dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]

    # choice[i][j] = L: 
    # L = 0  -> 达到 dp[i][j] 的最优解是 *不* 选择 a_i
    # L > 0  -> 达到 dp[i][j] 的最优解是选择 a_i 作为长度为 L 的块的结尾
    choice = [[-1] * (k + 1) for _ in range(n + 1)]

    # 基础情况: 选择 0 个元素，成本永远是 0
    for i in range(n + 1):
        dp[i][0] = 0
        choice[i][0] = 0

    # for j in range(k + 1):
        # dp[0][j] = 0

    # --- 3. DP 迭代 ---
    for i in range(1, n + 1):  # 遍历每个数字 a_i (即 A[i-1])
        for j in range(1, k + 1):  # 遍历选择的数量 j
            
            # 情况 1: 不选择 a_i
            # 成本继承自 dp[i-1][j]
            dp[i][j] = dp[i-1][j]
            choice[i][j] = 0  # 标记为 "跳过"
            
            # 情况 2: 选择 a_i 作为长度为 L (L >= 1) 的块的结尾
            # L 的最大值是 min(i, j)
            # i: 块不能比 a_1...a_i 更长
            # j: 块不能选择超过 j 个总数
            for L in range(1, min(i, j) + 1):
                # 这个块是 a_{i-L+1} ... a_i
                # 我们需要的前一个状态是 dp[i-L][j-L]
                
                prev_i = i - L
                prev_j = j - L
                
                # 检查前一个状态是否可达
                if dp[prev_i][prev_j] == float('inf'):
                    continue

                # 检查前一个状态是否选择了 a_{i-L}
                # 这个地方有错误！！不应该这么做！！会漏掉情况!
                # if choice[prev_i][prev_j] > 0:
                #     continue

                a = dp[prev_i][prev_j]
                if choice[prev_i][prev_j] > 0:
                    a = dp[prev_i-1][prev_j]
                
                # 计算这个新块 (a_{i-L+1} ... a_i) 的成本
                # Cost(s, e) = (PP[e] - PP[s-1]) - L * P[s-1]
                # 这里 s = i-L+1, e = i, s-1 = i-L
                block_cost = (PP[i] - PP[prev_i]) - L * P[prev_i]
                
                # 总成本 = 前一个状态的成本 + 新块的成本
                cost_with_block = a + block_cost
                
                # 如果这个新成本更优
                if cost_with_block < dp[i][j]:
                    dp[i][j] = cost_with_block
                    choice[i][j] = L  # 记录我们选择了长度为 L 的块

    # --- 4. 获取最终答案 ---
    min_sum = dp[n][k]

    # --- 5. 回溯路径以找到解 ---
    selected_indices = []
    
    if min_sum == float('inf'):
        return float('inf'), []

    curr_i = n
    curr_j = k
    
    while curr_i > 0 and curr_j > 0:
        L = choice[curr_i][curr_j]
        
        if L == 0:
            # 我们没有选择 a_i, 向前移动一个元素
            curr_i -= 1
        elif L > 0:
            # 我们选择了一个长度为 L 的块，结尾是 a_i
            # 块的 1-based 索引是 (curr_i - L + 1) 到 curr_i
            start_index = curr_i - L + 1
            end_index = curr_i
            
            for idx in range(end_index, start_index - 1, -1):
            # for idx in range(start_index, end_index + 1):
                selected_indices.append(idx)
            
            # 跳到这个块之前的状态
            curr_i = curr_i - L
            curr_j = curr_j - L
        else:
            # 理论上不应该发生，如果发生说明有bug或状态不可达
            break
            
    # 因为我们是从后往前添加的，所以需要反转
    selected_indices.reverse()
    
    return min_sum, selected_indices


# 示例 1: A = [10, 1, 2, 20, 30, 1], k = 3
# 预期最优解：选择 a_2=1, a_3=2, a_6=1
# 块 {a_2, a_3}: (1) + (1+2) = 4
# 块 {a_6}: (1)
# 总和 = 5
# A1 = [10, 1, 2, 20, 30, 1]
# k1 = 3
# min_sum1, indices1 = solve_min_cumulative_sum(A1, k1)
# print(f"序列 A1: {A1}, k = {k1}")
# print(f"最小总和: {min_sum1}")
# print(f"选择的索引 (1-based): {indices1}")
# print("-" * 20)

# 示例 2: A = [10, 10, 1, 1, 1, 10, 10], k = 4
# 预期最优解：选择中间的 {1, 1, 1} 和另一个 10
# 块 {a_3, a_4, a_5} = {1, 1, 1}: (1) + (1+1) + (1+1+1) = 1 + 2 + 3 = 6
# 块 {a_2} = {10}: 10
# 总和 = 16
# (如果选 {a_1} = 10，总和也是 16)
# A2 = [10, 10, 1, 1, 1, 10, 10]
# k2 = 4
# min_sum2, indices2 = solve_min_cumulative_sum(A2, k2)
# print(f"序列 A2: {A2}, k = {k2}")
# print(f"最小总和: {min_sum2}")
# print(f"选择的索引 (1-based): {indices2}")
# print("-" * 20)

# 示例 3: (来自您的提问) A = [a1, a2, a3, a4, a5, a6, a7, a8]
# 选择 a1, a3, a4, a5, a7, a8 (k=6)
# 假设 A = [10, 99, 1, 1, 1, 99, 1, 1] (设置 a2, a6 为高成本)
# A3 = [10, 99, 1, 1, 1, 99, 1, 1]
# k3 = 6
# min_sum3, indices3 = solve_min_cumulative_sum(A3, k3)
# print(f"序列 A3: {A3}, k = {k3}")
# print(f"最小总和: {min_sum3}")
# print(f"选择的索引 (1-based): {indices3}")
# 预期成本:
# 块 {a1}: 10
# 块 {a3, a4, a5}: (1) + (1+1) + (1+1+1) = 1 + 2 + 3 = 6
# 块 {a7, a8}: (1) + (1+1) = 1 + 2 = 3
# 总和 = 10 + 6 + 3 = 19
# 预期索引: [1, 3, 4, 5, 7, 8]


A3 = [10, 99, 1, 1, 1, 99, 1, 1, 2, 2, 6, 7, 2, 9]
k3 = 10
# A3 = [10, 99, 1, 1, 1, 99, 1, 1, 2, 2]
# k3 = 8
# A3 = [10, 99, 1, 1, 1, 99, 1, 1, 2, 2, 6, 7]
# k3 = 8
min_sum3, indices3 = solve_min_cumulative_sum(A3, k3)
print(f"序列 A3: {A3}, k = {k3}")
print(f"最小总和: {min_sum3}")
print(f"选择的索引 (1-based): {indices3}")
