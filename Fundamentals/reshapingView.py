import torch

x = torch.arange(1., 8.)
print(x, x. shape)

x_reshaped = x.reshape(1, 1, 7)
print(x_reshaped, x_reshaped.shape)

print(x, x. shape)

z = x.view(1,7)
print(z, z.shape)

z[:,0] = 5

print(z,x)
