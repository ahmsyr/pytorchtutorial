import torch

tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)


tensor_float16 = tensor.type(torch.float16)
print(tensor_float16.dtype)

tensor_int8 = tensor.type(torch.int8)
print(tensor_int8.dtype)


