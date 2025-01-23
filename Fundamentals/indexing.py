import torch

x = torch.arange(1,10).reshape(1,3,3)
print(x)
print(x.shape)

print(f"First square bracket:\n {x[0]}")
print(f"second square bracket:\n {x[0][0]}")
print(f"second square bracket:\n {x[0][1]}")
print(f"second square bracket:\n {x[0][2]}")

# print(f"second square bracket:\n {x[1][1]}")  # Error

print(f"third square bracket:\n {x[0][1][2]}")  # 6

print(x[:,0])
print(x[:,1])
print(x[:,:,0])
print(x[:,:,1])

print(x[:,0,2].item())  #3

print(x[0,1,:])