import numpy as np

# Define the 3x3 input array
# input_array = np.array([[1, 2, 3],
#                         [4, 5, 6],
#                         [7, 8, 9]], dtype=np.float32)

input_array = np.array([[14, 28, 42],
                        [56, 71, 85],
                        [99, 113, 127]], dtype=np.float32)

# Define the 2x2 kernel
# kernel = np.array([[1, 2],
#                    [3, 4]], dtype=np.float32)

kernel = np.array([[32, 64],
                   [96, 127]], dtype=np.float32)

fc_weight = np.array([[1, 2],
                   [3, 4]], dtype=np.float32)

# Define the stride
stride = 1

# Get the dimensions of the input and the kernel
input_height, input_width = input_array.shape
kernel_height, kernel_width = kernel.shape

# Calculate the dimensions of the output
output_height = (input_height - kernel_height) // stride + 1
output_width = (input_width - kernel_width) // stride + 1

# Initialize the output array
output_array = np.zeros((output_height, output_width), dtype=np.float32)

# Perform the convolution operation
for i in range(0, output_height):
    for j in range(0, output_width):
        # Extract the region of interest from the input array
        region = input_array[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
        # Compute the convolution (element-wise multiplication and sum)
        output_array[i, j] = np.sum(region * kernel)

print("Input Array:")
print(input_array)
print("\nKernel:")
print(kernel)
print("\nOutput Array:")
print(output_array)

outcome = np.sum(output_array * fc_weight)
print(outcome)