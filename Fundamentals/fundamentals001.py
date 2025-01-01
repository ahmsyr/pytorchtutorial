import torch
scalar = torch.tensor(200)
print(scalar)
print(scalar.dtype)
print(scalar.ndim)
print(scalar.item())

# Vector
vector = torch.tensor([7,8,9])
print(vector)
print(vector.dtype)
print(vector.ndim)
print(vector.shape)
print(vector.tolist())

# Matrix
matrix = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(matrix)
print(matrix.dtype)
print(matrix.ndim)
print(matrix.shape)
print(matrix.tolist())
