import torch

x = torch.rand(size=(224,224,3))

print(x.shape)

x_permuted = x.permute(2,0,1)
print(x_permuted.shape)