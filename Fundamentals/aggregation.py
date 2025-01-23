import torch
x = torch.arange(start=0, end=100,step=10,dtype=torch.float32)
print(x)

print(f"Minimum: {x.min()}")
print(f"Minimum: {torch.min(x)}")

print(f"Maximum: {x.max()}")
print(f"Maximum: {torch.max(x)}")

print(f"Mean: {x.mean()}")
print(f"Mean: {torch.mean(x)}")

print(f"Sum: {x.sum()}")
print(f"Sum: {torch.sum(x)}")
