#=================================================
#   ML_Project__Auditory Attention Detection (on a part of KULeuven Dataset)
#   1_5_EEG_CNN_FeatureMaps
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


#% Libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


#%% (EEG_CNN) Convolutional Neural Network For EEG

# Defining the convolutional neural network class
class EEG_CNN(nn.Module):
    def __init__(self):
        
        ### call the parent class constructor
        super(EEG_CNN, self).__init__()
        
        # Define the convolutional layers
        # Define the maxpooling layers
        # Define the batch normalization layers
        # Define the dropout layer
        # Define the ReLU activation function
        # Define the adaptive pooling layer
        
        # Layer 1
        # Layer 2
        # Layer 3
        # Layer 4

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(24,1), stride=(1, 1), dilation=(1,1), padding=(12,0))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(7,1), stride=(1, 1), dilation=(2,1), padding=(6,0))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(7,5), stride=(1, 1), dilation=(1,1), padding=(3,2))
        self.conv4 = nn.Conv2d(32, 1, kernel_size=(7,1), stride=(1, 1), dilation=(1,1), padding=(3,0))
        
        # Define the maxpooling layers
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,2))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,5))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1,1))
        
        # Define the batch normalization layers
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.bn4 = nn.BatchNorm2d(num_features=1)

        # Define the dropout layer
        self.drop = nn.Dropout(p=0.25)

        # Define the ReLU activation function
        self.relu = nn.ReLU()

        # Define the adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((48,32))
        
        
    ### define the forward method that takes an input x and returns an output x
    def forward(self, x):
        # order: CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x = self.bn1(self.maxpool1(self.conv1(x)))
        x = self.bn2(self.maxpool2(self.conv2(x)))
        x = self.bn3(self.maxpool3(self.conv3(x)))
        x = self.bn4(self.maxpool4(self.conv4(x)))

        x = self.drop(self.relu(x))
        x = self.adaptive_pool(x)
        
        return x


# Create an instance of the EEG_CNN class
model = EEG_CNN()


#%%
# Read EEG files name
OurEEG_Names = pd.read_csv("./1_3_EEG_10Ch_csv_Files/OurEEG_Names.csv")

for j in range(0,OurEEG_Names.shape[0]):
# for j in range(0,1): #for_test
    
    EEG_feature_maps = []
    EEG_Name = OurEEG_Names.iloc[j][0]
    RawSignals = np.load(f"./1_4_EEG_10Ch_RawSignal/RawSignal_{EEG_Name}.npy")
    
    
    i=-1
    for E in RawSignals:
        i = i+1
        
        E_tensor = torch.from_numpy(E).unsqueeze(0).unsqueeze(0) # convert Spec to a torch tensor and add two extra dimensions
        
        output_tensor = model(E_tensor.to(torch.float32)) # pass Spec through the model and get the output tensor
        output = output_tensor.detach().numpy().squeeze() # convert the output tensor to a numpy array and remove the extra dimensions
        
        EEG_feature_maps.append(output) # append the output to the list
        
        np.save(f"./2_5_All_epochs_Dataset/{EEG_Name}_epoch{i}.npy", output)

    np.save(f"./1_5_EEG_Dataset_Feature_Maps/EEG_feature_maps_{EEG_Name}.npy", EEG_feature_maps)
    # feature_maps is a list of 2D arrays of shape (48, 32)

