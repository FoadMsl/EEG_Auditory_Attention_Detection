#=================================================
#   ML_Project__Auditory Attention Detection (on a part of KULeuven Dataset)
#   3_2_AE_Concat_Function
#   Foad Moslem (foad.moslem@gmail.com) - Researcher | Aerodynamics
#   Using Python 3.11.4 & Spyder IDE
#=================================================

#%%
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

#%%

#% Libraries
import torch.nn as nn

#% Function - (BiLSTM) Bidrectional LSTM and fully connected layers
input_size = 48*64 # The size of the concatenated feature map
hidden_size = 48 # The size of the hidden state of the BLSTM layer
num_layers = 1 # The number of layers of the BLSTM layer
direction_scale = 0.5
num_spkr = 2 # The number of output classes (speaker 1 or speaker 2)
dropout = 0.25 # The dropout probability

# Bidrectional LSTM and fully connected layers
class blstm(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        # Bidirectional LSTM layer
        self.blstm = nn.LSTM(input_size = input_size,
                        hidden_size = int(hidden_size * direction_scale),
                        num_layers = num_layers,
                        batch_first = True,
                        bidirectional = True)
        
        # Four fully connected layers
        self.fc1 = nn.Linear(in_features = hidden_size*2, out_features = 2304)
        self.fc2 = nn.Linear(in_features = 128, out_features = 128)
        self.fc3 = nn.Linear(in_features = 128, out_features = 32)
        self.fc4 = nn.Linear(in_features = 32, out_features = num_spkr)
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        # Pass the input through the BLSTM layer
        x, _ = self.blstm(x)
        # Take the last output of the BLSTM layer
        # x = x[:, -1, :]

        # Pass the output through the four FC layers with ReLU activation for the first three and softmax for the last one
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.softmax(self.fc4(x), dim=1) 
        
        return x