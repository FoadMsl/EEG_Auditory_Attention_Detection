#=================================================
#   ML_Project__Auditory Attention Detection (on a part of KULeuven Dataset)
#   1_4_EEG_CNN_Function
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


#%% (EEG_CNN) Convolutional Neural Network For EEG
import torch.nn as nn # import the neural network module

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


#%%
""" Explanation:      
    #=================================================
        torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, 
                           return_indices=False, ceil_mode=False)
        
        Applies a 2D max pooling over an input signal composed of several input 
        planes.
        
        kernel_size:    the size of the window to take a max over
        stride:         the stride of the window. Default value is kernel_size
        padding:        Implicit negative infinity padding to be added on both 
                        sides
        dilation:       a parameter that controls the stride of elements in the 
                        window
        return_indices: if True, will return the max indices along with the 
                        outputs. Useful for torch.nn.MaxUnpool2d later
        ceil_mode:      when True, will use ceil instead of floor to compute 
                        the output shape
    
    #=================================================
        torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, 
                             track_running_stats=True, device=None, dtype=None)
        
        Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs 
        with additional channel dimension)
        
        num_features: C from an expected input of size (N,C,H,W)
        eps: a value added to the denominator for numerical stability. 
            Default: 1e-5
        momentum: the value used for the running_mean and running_var 
            computation. Can be set to None for cumulative moving average 
            (i.e. simple average). Default: 0.1
        affin: a boolean value that when set to True, this module has learnable 
            affine parameters. Default: True
        track_running_stats: a boolean value that when set to True, this module 
            tracks the running mean and variance, and when set to False, this 
            module does not track such statistics, and initializes statistics 
            buffers running_mean and running_var as None. When these buffers 
            are None, this module always uses batch statistics. in both 
            training and eval modes. Default: True    
    
    #=================================================
        torch.nn.Dropout(p=0.5, inplace=False)
        
        During training, randomly zeroes some of the elements of the input 
        tensor with probability p using samples from a Bernoulli distribution. 
        Each channel will be zeroed out independently on every forward call.
        This has proven to be an effective technique for regularization and 
        preventing the co-adaptation of neurons.
        
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to True, will do this operation in-place. Default: False
        
    #=================================================
        torch.nn.ReLU(inplace=False)
        Applies the rectified linear unit function element-wise
        inplace: can optionally do the operation in-place. Default: False

    #=================================================
"""