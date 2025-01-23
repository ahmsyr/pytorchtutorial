import torch

tensor_A = torch.tensor([[1, 2],
                        [3, 4],
                        [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 8],
                        [9, 10],
                        [11, 12]], dtype=torch.float32) 


print(f"Shape of tensor_A: {tensor_A.shape}")
print(f"Shape of tensor_B: {tensor_B.T.shape}")

print(f"tensor_A @ tensor_B:\n {tensor_A @ tensor_B.T}")
print(f"Shape tensor_A @ tensor_B:\n {(tensor_A @ tensor_B.T).shape}")

# Y = A . X + B

# torch.nn.Linear(in_features=2, out_features=1)





