import torch
import numpy as np

array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array).type(dtype=torch.float32)
print(array)
print(tensor)

array = array + 1
print(array)
print(tensor)

tensor = torch.ones(7)
numpy_array = tensor.numpy()
print(tensor)
print(numpy_array)

tensor = tensor + 1
print(tensor)
print(numpy_array)