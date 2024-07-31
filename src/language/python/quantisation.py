import random

# Define the range
a = 1.5
b = 5.5

# Generate a list of 100 random floats in the range [a, b]
random_floats = [random.uniform(a, b) for _ in range(100)]

# Print the list of random floats
# print(random_floats)

def calcScaleZeroPoint(min_val, max_val, num_bits):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmin - min_val / scale

    zero_point = int(zero_point)

    return scale, zero_point


def quantize_tensor(x):
    min_val, max_val = min(x), max(x)
    num_bits = 8

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)

    q_x = [int(e / scale) + zero_point for e in x]
    return q_x

a = quantize_tensor(random_floats)
print(random_floats)
print(a)
