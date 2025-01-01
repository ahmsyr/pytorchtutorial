

what_were_covering = {1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}


import torch
from torch import nn
import matplotlib.pyplot as plt

print(torch.__version__)





    
weight = 0.7
bias = 0.3

# Create Data
start = 0
end = 1
step = 0.02

X = torch.arange(start=start,end=end, step=step).unsqueeze(dim=1)
y= weight * X + bias

print(X[:10])
print(y[:10])

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test),len(y_test))

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """_summary_

    Args:
        train_data (_type_, optional): _description_. Defaults to X_train.
        train_labels (_type_, optional): _description_. Defaults to y_train.
        test_data (_type_, optional): _description_. Defaults to X_test.
        test_labels (_type_, optional): _description_. Defaults to y_test.
        predictions (_type_, optional): _description_. Defaults to None.
        
        Plots training data, test data and compares predictions
    """
    plt.figure(figsize=(10,7))
    
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    
    #Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    
    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
        
    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()
    
plot_predictions()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float
                                                ))
        self.bias = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float
                                                ))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
    
torch.manual_seed(42)

model_0 = LinearRegressionModel()

print(list(model_0.parameters()))

print(model_0.state_dict())

with torch.inference_mode():
    y_preds = model_0(X_test)
    
print(f"Number of testing samples: {len(X_test)}")
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values: \n {y_preds}")

plot_predictions(predictions=y_preds)

print(y_test - y_preds)


# Create Loss function
loss_fn = nn.L1Loss()

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01) # learning rate


torch.manual_seed(42)

# Set number of epochs
epochs = 200

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training
    
    # Put the model in training mode
    model_0.train()
    
    # 1. Forward pass on train data using forward() method inside model
    y_pred = model_0(X_train)
    
    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)
    
    # 3. Zero grad of the optimizer
    optimizer.zero_grad()
    
    # 4. loss backwards
    loss.backward()
    
    # 5. Progress the optimizer
    optimizer.step()
    
    ### Testing
    
    # Put the model in evaluation mode
    model_0.eval()
    
    with torch.inference_mode():
        # 1. Forward pass on test data
        test_pred = model_0(X_test)
        
        # 2. Calculate loss on test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float))
        
        # Print out what's happenning
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} MAE Train Loss: {loss} | MAE Test Loss: {test_loss}") 
            
# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train Losss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

print( "The model learned the weight and bias as follows:")
print(model_0.state_dict())
print(" The original values for weights and bias are:")
print(f"weight: {weight}, bias: {bias}")


# 1. Set the model in evaluation mode
model_0.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
    # 3. Make sure the calculations are done with the model and data on the same devie(CPU)
    y_preds = model_0(X_test)
    
print(y_preds)
        
plot_predictions(predictions=y_preds)  


from pathlib import Path

# 1. Create model directory
MODEL_PATH = Path("models")  
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to : {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)

torch.manual_seed(42)
# Loading model
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH,weights_only=True))

# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)
    
print(y_preds == loaded_model_preds)

print(loaded_model_0.state_dict())
plot_predictions(predictions=loaded_model_preds)

    
    