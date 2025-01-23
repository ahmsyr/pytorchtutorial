import torch

x = torch.arange(0.0, 10.0)

x_reshaped = x.reshape(1,2,5)

print(x_reshaped.shape)

x_squeezed = x_reshaped.squeeze()
print(x_squeezed.shape)


x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(x_unsqueezed.shape)

x_unsqueezed = x_unsqueezed.unsqueeze(dim=0)
print(x_unsqueezed.shape)

print(x_reshaped)
print(x_squeezed)
print(x_unsqueezed)


