import math

def min_total_cost_and_indices(a, k):
    """
    输入:
        a: list or 1-based-indexable sequence of numbers (we'll treat it as 1-based inside)
        k: 需要选择的元素个数
    返回:
        (min_cost, chosen_indices_list_sorted)
    时间复杂度: O(n^2 * k)
    """
    n = len(a)
    # convert to 1-based for方便计算
    A = [0] + list(a)
    # 前缀和 S1: sum a_i, S2: sum a_i * i
    S1 = [0] * (n + 1)
    S2 = [0] * (n + 1)
    for i in range(1, n + 1):
        S1[i] = S1[i-1] + A[i]
        S2[i] = S2[i-1] + A[i] * i

    def cost(s, e):
        """cost of choosing the full contiguous block s..e (包含端点)"""
        # cost(s,e) = (e+1)*(sum_{t=s..e} a_t) - sum_{t=s..e} a_t * t
        sum_a = S1[e] - S1[s-1]
        sum_atimes = S2[e] - S2[s-1]
        return (e + 1) * sum_a - sum_atimes

    INF = 10**30
    # dp_end[e][j] 最后一个段以 e 结尾，选了 j 个
    # dp_best[e][j] 前 e 个位置选 j 个的最优（可不以 e 结尾）
    dp_end = [ [INF] * (k+1) for _ in range(n+1) ]
    dp_best = [ [INF] * (k+1) for _ in range(n+1) ]

    # 回溯指针
    # prev[some] 用于重建：当 dp_end[e][j] 来自 dp_best[s-1][j-L] + cost(s,e) 时，记录 s
    prev_start = [ [ -1 ] * (k+1) for _ in range(n+1) ]
    # dp_best 来源：0 表示来自 dp_best[e-1], 1 表示来自 dp_end[e]
    best_from_end = [ [ False ] * (k+1) for _ in range(n+1) ]

    dp_best[0][0] = 0
    for j in range(1, k+1):
        dp_best[0][j] = INF

    # DP
    for e in range(1, n+1):
        # 拷贝不选 e 的情况
        for j in range(0, k+1):
            # 先假设不以 e 结尾，继承 dp_best[e-1][j]
            dp_best[e][j] = dp_best[e-1][j]
            best_from_end[e][j] = False

        # 枚举段起点 s，使得段为 s..e，长度 L = e-s+1
        for s in range(1, e+1):
            L = e - s + 1
            if L > k:
                # 这段长度已经超过 k，后面的 s 更小 L 更大，可继续但这里跳不过
                # 但 s 从 1 到 e，L decreases as s increases; we can't break safely
                pass
            # 对所有 j >= L: 更新 dp_end[e][j]
            c = cost(s, e)
            # 遍历 j 从 L..k
            for j in range(L, k+1):
                prev_val = dp_best[s-1][j - L]
                if prev_val >= INF:
                    continue
                cand = prev_val + c
                if cand < dp_end[e][j]:
                    dp_end[e][j] = cand
                    prev_start[e][j] = s

        # 更新 dp_best[e][j] = min(dp_best[e-1][j], dp_end[e][j])
        for j in range(0, k+1):
            if dp_end[e][j] < dp_best[e][j]:
                dp_best[e][j] = dp_end[e][j]
                best_from_end[e][j] = True

    min_cost = dp_best[n][k]
    if min_cost >= INF/2:
        return (None, [])  # 无可行解（例如 k>n）
    # 回溯找到具体选中的区间，然后展开为索引
    chosen_intervals = []
    e = n
    j = k
    while e > 0 and j > 0:
        if best_from_end[e][j]:
            # dp_best[e][j] 来自 dp_end[e][j]，说明有一个段以 e 结尾
            s = prev_start[e][j]
            if s == -1:
                # Should not happen
                raise RuntimeError("回溯错误：找不到段起点")
            chosen_intervals.append((s, e))
            L = e - s + 1
            j -= L
            e = s - 1
        else:
            # dp_best[e][j] 来自 dp_best[e-1][j]，跳过位置 e
            e -= 1
    # 若 j==0，done. 若 j>0 无解（已处理）
    # 将区间展开为选中下标（按升序）
    chosen_indices = []
    for s, e in reversed(chosen_intervals):
        for idx in range(s, e+1):
            chosen_indices.append(idx)
    return (min_cost, chosen_indices)


# 测试一个小例子
if __name__ == "__main__":
    # 示例：a = [1,2,3,4,5,6,7,8], n=8, 选 k = 4
    # a = [1,2,3,4,5,6,7,8]
    # k = 4
    # a = [10, 1, 2, 20, 30, 1]
    # k = 3
    a = [10, 10, 1, 1, 1, 10, 10]
    k = 4
    cost, inds = min_total_cost_and_indices(a, k)
    print("min cost =", cost)
    print("chosen indices =", inds)
