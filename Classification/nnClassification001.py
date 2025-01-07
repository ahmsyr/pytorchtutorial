# Make Data

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
from sklearn.model_selection import train_test_split

n_samples = 10000

X, y = make_circles(n_samples=n_samples,
                    noise=0.03,
                    random_state=42)

plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdBu)
plt.show()

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print(X_train[:5], y_train[:5])

# Build Model 
from torch import nn
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        y1 = self.relu(self.layer_1(x))
        y2 = self.relu(self.layer_2(y1))
        y3 = self.relu(self.layer_3(y2))
        return y3
        # return self.layer_3(self.layer_2(self.layer_1(x)))

device = "cuda" if torch.cuda.is_available() else "cpu"
model_NN = CircleModel().to(device=device)

print(model_NN)

# Setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_NN.parameters(), lr=0.1)

# Fit the model
torch.manual_seed(42)
epochs = 6300

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
# y_logits = model_NN(X_train).squeeze()
# print(torch.round(torch.sigmoid(y_logits)))

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) *100
    return acc

import requests
from pathlib import Path

# Download helper functions from github (Learn PyTorch)
if Path("helper_functions.py").is_file():
    print("helper_functions.py is already exists, skipping download")
else:
    print(" Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", 'wb') as f:
        f.write(request.content)

from helper_functions import plot_decision_boundary

for epoch in range(epochs):
    ### Training
    model_NN.train()
    # 1. Forward pass
    y_logits = model_NN(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits after applying sigmoid
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    
    # 4. Loss backward
    loss.backward()
    
    # 5. Optimizer step
    optimizer.step()
    
    ### Testing
    model_NN.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_NN(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
        
    # Print out what's happenning
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss:{loss:.5f}, Accuracy: {acc:.2f}% | Test Loss:{test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
    
 
model_NN.eval()
with torch.inference_mode():
    y_pred = torch.round(torch.sigmoid(model_NN(X_test))).squeeze()

print(y_pred[:10], y[:10])

# Plot decision boundaries for training and test sets
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_NN, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_NN, X_test, y_test)

plt.show()



