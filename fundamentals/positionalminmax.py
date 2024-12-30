import torch
tensor = torch.arange(10, 100, 10)

print(f"Tensor: {tensor}")

print(f"Index of max value: {tensor.argmax()}")
print(f"Index of min value: {tensor.argmin()}")