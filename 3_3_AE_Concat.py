#=================================================
#   ML_Project__Auditory Attention Detection (on a part of KULeuven Dataset)
#   3_3_AE_Concat
#   Foad Moslem - PhD Student - Aerodynamics
#   Using Python 3.11.4 & Spyder IDE
#=================================================

#%%
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


#% Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


#%% AE_Concat_Function

# Define the Bi-LSTM network class
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(BiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Four fully connected layers
        self.fc1 = nn.Linear(hidden_size*2, 128) # *2 for bidirectional
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, num_classes)
        
        # Activation functions
        self.dropout = nn.Dropout(dropout_rate)
    
    
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # *2 for bidirectional
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # Pass through the fully connected layers with relu activation and dropout
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = torch.relu(out)
        out = self.dropout(out)
        # Apply softmax activation to the last layer
        out = self.fc4(out)
        out = torch.softmax(out, dim=1)
        
        return out


#%% Import Our Dataset
Dataset = torch.load('./Dataset.pt')

X = Dataset[0]
y = []
for i in range(0, len(Dataset[0])):
# for i in range(0, 1): #dor_test
    y.append(torch.full((len(Dataset[0][i]),), Dataset[1][i]))

# Concatenate the list of tensors into a single tensor
X = torch.cat(X, dim=0)
y = torch.cat(y, dim=0)

# Split the data into train, validation, and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)
# X_test, X_val, y_test, y_val = train_test_split(X_train, y_train, train_size=0.5, random_state=42)


#%% Define Hyperparameters, model, loss function, and optimizer

# Define the parameters and hyperparameters
input_size = 64 # The number of features in the EEG data
hidden_size = 32 # The number of hidden units in the LSTM
num_layers = 1 # The number of layers in the LSTM
num_classes = 2 # The number of classes to classify (e.g. emotions)
dropout_rate = 0.25 # The dropout rate for the fully connected layers

batch_size = 32 # The size of each batch of data
learning_rate = 5e-4 # The learning rate for the optimizer

num_epochs = 80 # The number of epochs to train the model
seq_length = 10 # The length of each sequence of data

# Set the device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model instance and move it to device
model = BiLSTM(input_size, hidden_size, num_layers, num_classes, dropout_rate).to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#%% Generate Random Tensor For test
# # Generate some random data for demonstration purposes
# X_train = torch.randn(batch_size, seq_length, input_size).to(device) # Input data
# y_train = torch.randint(0, num_classes, (batch_size,)).to(device) # Labels
# X_test = torch.randn(batch_size, seq_length, input_size).to(device) # New input data
# y_test = torch.randint(0, num_classes, (batch_size,)).to(device) # New labels

#%%
# Train the model
for epoch in range(num_epochs):
    ### Forward pass
    # Forward pass the input through the model
    outputs = model(X_train)
    # Calculate the loss
    loss = criterion(outputs, y_train)
    
    ### Backward and optimize
    # Zero the gradients of the model parameters
    optimizer.zero_grad()
    # Backpropagate the loss
    loss.backward()
    # Update the model parameters
    optimizer.step()
    
    # Print the loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')


# Make predictions and calculate accuracy
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == y_test).sum().item()
    accuracy = correct / batch_size
    print(f'Accuracy: {accuracy:.2f}%')
