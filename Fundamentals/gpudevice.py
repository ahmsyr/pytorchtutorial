import torch

print(torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

print(torch.cuda.device_count())

tensor = torch.tensor([1, 2, 3])

print(tensor, tensor.device)


# Move tensor to GPU (if available)
tensor_gpu = tensor.to(device)
print(tensor_gpu, tensor_gpu.device)


#numpyTensor = tensor_gpu.numpy()
tensor_back_to_cpu = tensor_gpu.cpu()
numpyarray = tensor_back_to_cpu.numpy()

print(numpyarray)

numpyarray = tensor_gpu.cpu().numpy()
print(numpyarray)

