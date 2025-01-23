import torch

# Create a tensor of all zeros
zeros = torch.zeros(size=(224,224,3))
print(zeros.shape, zeros.dtype)

# Create a tensor of all ones

ones = torch.ones(size=(2,3,1))
print(ones)
print(ones.shape, ones.dtype)
