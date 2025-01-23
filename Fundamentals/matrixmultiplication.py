import torch

tensor = torch.tensor([1, 2, 3])

print(f"tensor mul tensor: {torch.matmul(tensor, tensor)}")
print(f"tensor mul tensor: {tensor @ tensor }")

print(f"tensor * tensor: {tensor * tensor}")
print(f"tensor * tensor: {tensor.mul(tensor)}")

value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
print(f"tensor * tensor: {value}")