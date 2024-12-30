import torch

x = torch.arange(1., 10.)
y = torch.arange(5., 14.)

x_stacked = torch.stack([x,x,x,y],dim=0)

print(x_stacked)
print(x_stacked.shape)


