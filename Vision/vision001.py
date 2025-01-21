
import torchmetrics
import mlxtend




import torch
from torch import nn

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

#  Import matplotlib fgor visualization
import matplotlib.pyplot as plt

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Check versions
print(f"Pytorch version: {torch.__version__}\n torchvision version:{torchvision.__version__}")

# Setup training data
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None,
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),

)

image, label = train_data[0]
print(image, label)

print(image.shape)
print(len(train_data.data), len(train_data.targets))
print(len(test_data.data), len(test_data.targets))

class_names = train_data.classes
print(class_names)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(label=label)

plt.show()

torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows, cols = 4,4

for i in range(1,rows * cols +1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
    
plt.show()

from torch.utils.data import DataLoader

BATCH_SIZE = 32

train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

print(f"Dataloaders; {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)

# Show a sample
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis("Off")
plt.show()
print(f"Image size: {img.shape}")
print(f"Label: {label}, lable size: {label.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

from timeit import default_timer as timer
def print_train_time(start: float, end: float, device: torch.device = device):
    """_summary_

    Args:
        start (float): start time
        end (float): end time
        device (torch.device, optional): Computing device.
        
    Returns: 
        float: time between start and end in seconds
    """
    total_time = end - start
    print(f"Train time on {device} : {total_time:.3f} seconds")
    return total_time

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        data_loader (torch.utils.data.DataLoader): _description_
        loss_fn (torch.nn.Module): _description_
        accuracy_fn (_type_): _description_
        
    Returns:
    (dict): Results of model predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))
            
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
            
    return{"model_name": model.__class__.__name__,
           "model_loss": loss.item(),
           "model_acc": acc}
    

    
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # Send data to device
        X, y = X.to(device), y.to(device)
        
        # 1. Forward pass
        y_pred = model(X)
        
        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Loss backward
        loss.backward()
        
        # 5. Optimizer step()
        optimizer.step()
        
    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f} %")
    
    
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, 
                                   y_pred=test_pred.argmax(dim=1))
            
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /=  len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")
    
import requests
from pathlib import Path

# Download helper functions
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py","wb") as f:
        f.write(request.content)
        
# Import accuracy metric
from helper_functions import accuracy_fn



# TinyVGG

# Create a convolutional neural network
class FashionMNISTModel(nn.Module):
    """_summary_

    TinyVGG from:
    https://poloclub.github.io/cnn-explainer
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )
        
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        #print(x.shape)
        x = self.block_2(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x
    
torch.manual_seed(42)
model_cnn = FashionMNISTModel(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device=device)

print(model_cnn)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_cnn.parameters(),
                            lr=0.1)

torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
train_time_start_model_cnn = timer()

# Train and test model
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch}\n-----------")
    train_step(data_loader=train_dataloader,
               model=model_cnn,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(data_loader=test_dataloader,
              model=model_cnn,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
train_time_end_model_cnn=timer()
total_train_time_model_cnn = print_train_time(start=train_time_start_model_cnn,
                                              end=train_time_end_model_cnn,
                                              device=device)

model_cnn_results = eval_model(
    model=model_cnn,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
print(model_cnn_results)


def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_props = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device=device)
            # Forward pass
            pred_logit = model(sample)
            
            # Get prediction probability
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            
            # Get prob off GPU 
            pred_props.append(pred_prob.cpu())
            
    return torch.stack(pred_props)

import random
random.seed(42)
test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data),k=9):
    test_samples.append(sample)
    test_labels.append(label)
    
# View the first test sample shape and label
print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]}({class_names[test_labels[0]]})")

pred_probs = make_predictions(model=model_cnn,
                              data=test_samples)

# Turn the prediction probabilities into labels
pred_classes = pred_probs.argmax(dim=1)
print(test_labels, pred_classes)

# Plot predictions
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze(), cmap="gray")
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")
    else:
        plt.title(title_text, fontsize=10, c="r")
        
    plt.axis(False)
plt.show()

from tqdm.auto import tqdm
y_preds = []
model_cnn.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions"):
        X, y = X.to(device), y.to(device)
        y_logits = model_cnn(X)
        y_pred = torch.softmax(y_logits.squeeze(), dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())
# Concate4nate list of predictions into tensor        
y_pred_tensor = torch.cat(y_preds)

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# Setup Confusion matrix instance 
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10,7)
)
plt.show()

from pathlib import Path
# Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

# Create model save path
MODEL_NAME = "pytorch_computer_vision_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_cnn.state_dict(),
           f=MODEL_SAVE_PATH)

# Loading Model
loaded_model_cnn = FashionMNISTModel(input_shape=1,
                                     hidden_units=10,
                                     output_shape=10)

loaded_model_cnn.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model_cnn = loaded_model_cnn.to(device)

# Evaluate loaded model
torch.manual_seed(42)
loaded_model_cnn_results = eval_model(model=loaded_model_cnn,
                                      data_loader=test_dataloader,
                                      loss_fn=loss_fn,
                                      accuracy_fn=accuracy_fn)

print(loaded_model_cnn_results)
print(model_cnn_results)

print(torch.isclose(torch.tensor(model_cnn_results["model_loss"]),
                    torch.tensor(loaded_model_cnn_results["model_loss"]),
                    atol=1e-8,
                    rtol=0.0001))