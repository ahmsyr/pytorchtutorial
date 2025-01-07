import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
print(X_blob[:5], y_blob[:5])

# 3. Split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# 4. Plot data
plt.figure(figsize=(10,7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()

# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

from torch import nn

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) *100
    return acc


# Build model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """

        Args:
            input_features (_type_): Number of input features 
            output_features (_type_): Number of output features
            hidden_units (int, optional): Number of hidden units. Defaults to 8.
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units, ),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
        
    def forward(self, x):
        return self.linear_layer_stack(x)

# Create an instance of BlobModel
blob_model = BlobModel(input_features=NUM_FEATURES,
                       output_features=NUM_CLASSES,
                       hidden_units=8).to(device=device)



# Create loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(blob_model.parameters(),
                            lr=0.1)
print(blob_model)

print(blob_model(X_blob_train.to(device))[:5])

print(blob_model(X_blob_train.to(device))[0].shape, NUM_CLASSES)


# Make prediction logits with model
y_logits = blob_model(X_blob_test.to(device))
y_pred_probs = torch.softmax(y_logits, dim=1)
print(torch.sum(y_pred_probs[0]))
print(torch.argmax(y_pred_probs[0]))

# Fit the model
torch.manual_seed(RANDOM_SEED)

# Set number of epochs
epochs = 100

# Put data to target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    ### Training
    blob_model.train()
    
    # 1. Forward pass
    y_logits = blob_model(X_blob_train)
    # go from logits -> probabilities -> prediction labels
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_preds)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    
    # 4 Loss backwards
    loss.backward()
    
    # 5. Optimizer step
    optimizer.step()
    
    ### Testing
    blob_model.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = blob_model(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        # 2. Calculate test loss and accuracy
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_pred)
        
    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f} Test Acc: {test_acc:.2f}%")
    
# Make predictions
blob_model.eval()
with torch.inference_mode():
    y_logits = blob_model(X_blob_test)
    
# Find probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
# convert into labels
y_preds = y_pred_probs.argmax(dim=1)

print(f"Predictions: {y_preds[:10]} \nLabels: {y_blob_test[:10]}")
print(f"Test accuracy: {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}")

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

plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(blob_model, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(blob_model, X_blob_test, y_blob_test)
plt.show()
