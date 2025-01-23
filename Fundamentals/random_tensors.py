import torch

random_tensor =  torch.rand(size=(3,4))
print(random_tensor)
print(random_tensor.dtype)

random_image_size_tensor = torch.rand(size=(224,224,3))
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)