import torch
import time

# 设置随机种子以确保结果可重复
torch.manual_seed(42)

# 输入和模型
x = torch.randn (2000, 2000, device='cuda')
w1 = torch.randn(2000, 2000, device='cuda')
w2 = torch.randn(2000, 2000, device='cuda')

# 测试 1：不使用CUDA流（默认流）
def no_stream():
    start_time = time.time()
    y1 = x @ w1  # 矩阵乘法1
    y2 = x @ w2  # 矩阵乘法2
    y = y1 + y2  # 合并结果
    torch.cuda.synchronize()  # 确保GPU计算完成
    end_time = time.time()
    return end_time - start_time

# 测试 2：使用CUDA流
def with_streams():
    # 创建两个CUDA流
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    start_time = time.time()
    with torch.cuda.stream(stream1):
        y1 = x @ w1  # 矩阵乘法1
    with torch.cuda.stream(stream2):
        y2 = x @ w2  # 矩阵乘法2
    torch.cuda.synchronize()  # 等待所有流完成
    y = y1 + y2  # 合并结果
    end_time = time.time()
    return end_time - start_time



# 运行多次取平均值以减少随机波动
num_runs = 1000

with_streams_time = 0
no_stream_time = 0

for _ in range(num_runs):
    no_stream_time += no_stream()
    with_streams_time += with_streams()

with_streams_time /= num_runs
no_stream_time /= num_runs


# 输出结果
print(f"不使用CUDA流平均时间: {no_stream_time:.6f} 秒")
print(f"使用CUDA流平均时间: {with_streams_time:.6f} 秒")
print(f"时间差异: {(no_stream_time - with_streams_time):.6f} 秒")
