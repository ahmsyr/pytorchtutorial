import torch

float_32_tensor = torch.tensor([1.0,2.4,4.6],
                               dtype=None,
                               device=None,
                               requires_grad=False)

print(float_32_tensor)
print(float_32_tensor.dtype)
print(float_32_tensor.device)

float_16_tensor = torch.tensor([1.0,2.4,4.6],
                               dtype=torch.float16,
                               device=None,
                               requires_grad=False)

print(float_16_tensor)
print(float_16_tensor.dtype)
print(float_16_tensor.device)
