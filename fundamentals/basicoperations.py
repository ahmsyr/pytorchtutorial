import torch

tensor = torch.tensor([1,2,3])
print(tensor + 10)
print(tensor * 10)
print(tensor /2)
print(tensor ** 2)
print(tensor - 10)

print( torch.multiply(tensor, 10))
print(torch.add(tensor, 10))

print( f"{tensor} * {tensor} = {tensor * tensor}")
