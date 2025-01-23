import torch

zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)
print(zero_to_ten.dtype)

ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros)
