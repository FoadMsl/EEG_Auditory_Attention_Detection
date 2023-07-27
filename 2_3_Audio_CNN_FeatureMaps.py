#=================================================
#   ML_Project__Auditory Attention Detection (on a part of KULeuven Dataset)
#   2_3_Audio_CNN_FeatureMaps
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


#%
# Libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


#%% (Audio_CNN) Convolutional Neural Network For Audios

# Defining the convolutional neural network class
class Audio_CNN(nn.Module):
    def __init__(self):
        
        ### call the parent class constructor
        super(Audio_CNN, self).__init__()
        
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
        # Layer 5

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1,7), dilation=(1,1), padding=(0,3))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(7,1), dilation=(1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,5), dilation=(8,8), padding=(0,16))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(16,16), padding=(0,16))
        self.conv5 = nn.Conv2d(32, 1, kernel_size=(1,1), dilation=(1,1), padding=(0,0))
        
        # Define the maxpooling layers
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,4))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1,2))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1,1))
        self.maxpool5 = nn.MaxPool2d(kernel_size=(2,2))
        
        # Define the batch normalization layers
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.bn5 = nn.BatchNorm2d(num_features=1)
        
        # Define the dropout layer
        self.drop = nn.Dropout(p=0.4)

        # Define the ReLU activation function
        self.relu = nn.ReLU()

        # define the adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((48,16))
        

    def forward(self, x):
        # order: CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x = self.bn1(self.maxpool1(self.conv1(x)))
        x = self.bn2(self.maxpool2(self.conv2(x)))
        x = self.bn3(self.maxpool3(self.conv3(x)))
        x = self.bn4(self.maxpool4(self.conv4(x)))
        x = self.bn5(self.maxpool5(self.conv5(x)))

        x = self.drop(self.relu(x))
        x = self.adaptive_pool(x)
        
        return x
   

# create an instance of the Audio_CNN class
model = Audio_CNN()


#%%
# Read Audio files name
OurAudio_Names = pd.read_csv("./2_2_Audio_dry_wav_Files/OurAudio_Names.csv")

for j in range(0,OurAudio_Names.shape[0]):
# for j in range(0,1): #for_test
    
    Audio_feature_maps = []
    audio = OurAudio_Names.iloc[j][0]
    Spec = np.load(f"./2_3_Audio_dry_Spectrogram/spectrogram_{audio}.npy")

    
    i=-1
    for S in Spec:
        i = i+1

        S_tensor = torch.from_numpy(S).unsqueeze(0).unsqueeze(0) # convert Spec to a torch tensor and add two extra dimensions

        output_tensor = model(S_tensor.to(torch.float32)) # pass Spec through the model and get the output tensor
        output = output_tensor.detach().numpy().squeeze() # convert the output tensor to a numpy array and remove the extra dimensions
        
        Audio_feature_maps.append(output) # append the output to the list
        
        np.save(f"./2_5_All_epochs_Dataset/{audio}_epoch{i}.npy", output)

    
    np.save(f"./2_4_Audio_Dataset_Feature_Maps/Audio_feature_maps_{audio}.npy", Audio_feature_maps)
    # feature_maps is a list of 2D arrays of shape (48, 16)
