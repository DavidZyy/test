import sys

def solve_min_cost_k_selection(a_list, k):
    """
    使用动态规划 (O(n^2 * k)) 解决最小代价k选择问题。
    
    参数:
        a_list (list): 原始数字序列 (a_1, a_2, ..., a_n)。
        k (int): 需要选择的数字总数。
        
    返回:
        tuple: (min_cost, selected_indices)
                 min_cost (float): 最小的总代价。
                 selected_indices (list): 最小代价对应的（1-based）下标列表。
    """
    
    n = len(a_list)
    
    # --- 0. 处理边界情况 ---
    if k == 0:
        return 0, []
    if k > n:
        # 不可能选择比序列还长的数字
        return float('inf'), []
    
    # 为了方便计算，我们将序列a变为1-indexed
    # a[1] 对应 a_1, a[2] 对应 a_2, ...
    a = [0] + a_list
    
    # --- 1. 预处理：计算前缀和 P 和前缀和的前缀和 PP ---
    
    # P[i] = a[1] + ... + a[i]
    P = [0] * (n + 1)
    # PP[i] = P[1] + ... + P[i]
    PP = [0] * (n + 1)
    
    for i in range(1, n + 1):
        P[i] = P[i-1] + a[i]
        PP[i] = PP[i-1] + P[i]

    # --- 辅助函数：O(1) 计算一个段的代价 ---
    def get_segment_cost(s, e):
        """
        计算选择连续段 a[s...e] (均为1-based) 的总代价 C(s, e)。
        C(s, e) = (PP[e] - PP[s-1]) - (e - s + 1) * P[s-1]
        """
        if s > e:
            return 0
        cost = (PP[e] - PP[s-1]) - (e - s + 1) * P[s-1]
        return cost

    # --- 2. DP 初始化 ---
    infinity = float('inf')
    
    # dp[i][j][state]: 考虑了前 i 个数，总共选择了 j 个数，
    #                  且第 i 个数的状态为 state (0=未选, 1=已选) 时的最小代价。
    dp = [[[infinity, infinity] for _ in range(k + 1)] for _ in range(n + 1)]
    
    # path0[i][j] = 导致 dp[i][j][0] 最小代价的前一个状态 (0 或 1)
    path0 = [[-1 for _ in range(k + 1)] for _ in range(n + 1)]
    
    # path1[i][j] = 导致 dp[i][j][1] 最小代价的段的起始位置 's'
    path1 = [[-1 for _ in range(k + 1)] for _ in range(n + 1)]

    # 基础情况：考虑0个数，选择0个，状态为0，代价为0
    dp[0][0][0] = 0

    # --- 3. DP 循环 ---
    for i in range(1, n + 1):
        for j in range(k + 1):
            
            # --- 状态 0: a[i] 未被选中 ---
            # 代价来自 dp[i-1][j][0] 或 dp[i-1][j][1]
            cost_from_0 = dp[i-1][j][0]
            cost_from_1 = dp[i-1][j][1]
            
            if cost_from_0 <= cost_from_1:
                dp[i][j][0] = cost_from_0
                path0[i][j] = 0  # 记录从 state 0 转移而来
            else:
                dp[i][j][0] = cost_from_1
                path0[i][j] = 1  # 记录从 state 1 转移而来
                
            # --- 状态 1: a[i] 被选中 ---
            # a[i] 是某个段 (s...i) 的结尾 (1 <= s <= i)
            # 我们需要遍历所有可能的起始点 s
            
            for s in range(1, i + 1):
                segment_len = i - s + 1
                prev_k = j - segment_len
                
                # 检查是否满足选择k个的条件
                if prev_k >= 0:
                    # 这个段的代价
                    segment_cost = get_segment_cost(s, i)
                    
                    # 形成这个新段，要求 a[s-1] 必须未被选中 (state 0)
                    # 之前的代价 = dp[s-1][prev_k][0]
                    prev_cost = dp[s-1][prev_k][0]
                    
                    if prev_cost != infinity:
                        total_cost = prev_cost + segment_cost
                        if total_cost < dp[i][j][1]:
                            dp[i][j][1] = total_cost
                            path1[i][j] = s  # 记录这个最优段的起始点 s

    # --- 4. 找到最终答案 ---
    min_cost = min(dp[n][k][0], dp[n][k][1])
    
    if min_cost == infinity:
        return float('inf'), [] # 无法选出k个 (例如 k > n，虽然前面已处理)

    last_state = 0 if dp[n][k][0] <= dp[n][k][1] else 1

    # --- 5. 路径回溯 ---
    selected_indices = []
    curr_i = n
    curr_k = k
    curr_state = last_state

    while curr_i > 0:
        if curr_state == 0:
            # 第 i 个未被选中，回溯到 i-1
            prev_state = path0[curr_i][curr_k]
            if prev_state == -1:
                break # 到达初始状态
            curr_i -= 1
            curr_state = prev_state
        
        elif curr_state == 1:
            # 第 i 个被选中，它是一个段 (s...i) 的结尾
            s = path1[curr_i][curr_k]
            if s == -1:
                break # 异常
            
            segment_len = curr_i - s + 1
            
            # 将这个段的所有下标 (1-based) 加入结果
            for idx in range(s, curr_i + 1):
                selected_indices.append(idx)
            
            # 跳转到这个段开始之前
            curr_k -= segment_len
            curr_i = s - 1
            curr_state = 0 # 段的开始 s 之前必须是 state 0
        
        if curr_k == 0:
            break

    selected_indices.sort()
    return min_cost, selected_indices

# --- 示例 ---
if __name__ == "__main__":
    
    # 示例 1: 简单的例子
    # a = [10, 2, 3]  (k = 2)
    # 方案 1: 选 10, 2。段: [10], [2]。
    #    代价(10) = a[1] = 10
    #    代价(2) = a[2] = 2
    #    总代价 = 12
    # 方案 2: 选 10, 3。段: [10], [3]。
    #    代价(10) = a[1] = 10
    #    代价(3) = a[3] = 3
    #    总代价 = 13
    # 方案 3: 选 2, 3。段: [2, 3]。
    #    代价(2) = a[2] = 2
    #    代价(3) = a[2] + a[3] = 2 + 3 = 5
    #    总代价 = 2 + 5 = 7
    # 方案 4: 选 10, 2, 3 (k=3)。段: [10], [2, 3]。
    #    代价(10) = 10
    #    代价(2) = 2
    #    代价(3) = 2 + 3 = 5
    #    总代价 = 17
    # 方案 5: 选 10, 2, 3 (k=3)。段: [10, 2, 3]。
    #    代价(10) = 10
    #    代价(2) = 10 + 2 = 12
    #    代价(3) = 10 + 2 + 3 = 15
    #    总代价 = 37
    
    # a1 = [10, 2, 3]
    # k1 = 2
    a1 = [10, 1, 2, 20, 30, 1]
    k1 = 3
    cost1, indices1 = solve_min_cost_k_selection(a1, k1)
    print(f"序列: {a1}, k: {k1}")
    print(f"最小代价: {cost1}")
    print(f"选择下标 (1-based): {indices1}")
    print("-" * 20)

    # 示例 2: 题目描述中的例子
    # 假设 a = [6, 4, 1, 10, 3], k = 3
    # 尝试选择: 1, 3, 4 (下标) -> [6], [1], [10]
    #    代价(6) = 6
    #    代价(1) = 1
    #    代价(10) = 10
    #    总代价 = 17
    # 尝试选择: 2, 3, 4 (下标) -> [4, 1, 10]
    #    代价(4) = a[2] = 4
    #    代价(1) = a[2] + a[3] = 4 + 1 = 5
    #    代价(10) = a[2] + a[3] + a[4] = 4 + 1 + 10 = 15
    #    总代价 = 4 + 5 + 15 = 24
    
    # a2 = [6, 4, 1, 10, 3]
    # k2 = 3
    a2 = [10, 10, 1, 1, 1, 10, 10]
    k2 = 4
    cost2, indices2 = solve_min_cost_k_selection(a2, k2)
    print(f"序列: {a2}, k: {k2}")
    print(f"最小代价: {cost2}")
    print(f"选择下标 (1-based): {indices2}")
    print("-" * 20)

    # 示例 3: 另一个例子
    # a3 = [1, 100, 1, 1, 100]
    # k3 = 3
    # a3 = [10, 99, 1, 1, 1, 99, 1, 1]
    # k3 = 6
    # cost3, indices3 = solve_min_cost_k_selection(a3, k3)
    # print(f"序列: {a3}, k: {k3}")
    # print(f"最小代价: {cost3}")
    # print(f"选择下标 (1-based): {indices3}")
    # print("-" * 20)

    
    

    a3 = [10, 99, 1, 1, 1, 99, 1, 1, 2, 2, 6, 7, 2, 9]
    k3 = 10
    cost3, indices3 = solve_min_cost_k_selection(a3, k3)
    print(f"序列: {a3}, k: {k3}")
    print(f"最小代价: {cost3}")
    print(f"选择下标 (1-based): {indices3}")
    print("-" * 20)

