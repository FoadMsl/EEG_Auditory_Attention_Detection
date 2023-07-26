#=================================================
#   ML_Project__Auditory Attention Detection (on a part of KULeuven Dataset)
#   3_1_Dataset_Maker
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


#%% Read the original dataset info ile
Dataset_info = pd.read_csv("./Dataset_info.csv")

# extract what we need from dataset (dry files without repetition)
Dataset_info = Dataset_info.drop(Dataset_info[
    (Dataset_info['repetition'] == True) | 
    (Dataset_info['condition'] == "hrtf")].index)
Dataset_info = Dataset_info.reset_index(drop=True)


#%% Load Feature Maps of EEG and Audios of eahch row of Dataset
Dataset = []
features_array = []
target_array = []

for i in range(0,Dataset_info.shape[0]):
# for i in range(0,1): #for_test

    # Load EEG_feature_maps ======================================
    subject = Dataset_info["subject"][i]
    TrialID = Dataset_info["TrialID"][i]
    EEG = (f"{subject}_trials{TrialID}_10Ch")
    EEG_feature_map = np.load(
        f"./1_5_EEG_Dataset_Feature_Maps/EEG_feature_maps_{EEG}.npy")
    # Convert feature maps to tensor
    EEG_feature_map_Tensor = torch.from_numpy(EEG_feature_map)
    # ============================================================

    # Load Audio_feature_maps 1 ==================================
    Audio1 = Dataset_info["stimuli1"][i]
    Audio1_feature_map = np.load(
        f"./2_4_Audio_Dataset_Feature_Maps/Audio_feature_maps_{Audio1}.npy")
    # Convert feature maps to tensor
    Audio1_feature_map_Tensor = torch.from_numpy(Audio1_feature_map)
    Audio1_feature_map_Tensor = torch.narrow(Audio1_feature_map_Tensor, dim=0, start=0, length=len(EEG_feature_map_Tensor))
    # ============================================================

    # Load Audio_feature_maps 2 ==================================
    Audio2 = Dataset_info["stimuli2"][i]
    Audio2_feature_map = np.load(
        f"./2_4_Audio_Dataset_Feature_Maps/Audio_feature_maps_{Audio2}.npy")
    # Convert feature maps to tensor
    Audio2_feature_map_Tensor = torch.from_numpy(Audio2_feature_map)
    Audio2_feature_map_Tensor = torch.narrow(Audio2_feature_map_Tensor, dim=0, start=0, length=len(EEG_feature_map_Tensor))
    # ============================================================

    # Concatenate the audios feature maps (48, 32) ===============
    A_Concat = torch.cat((Audio1_feature_map_Tensor, Audio2_feature_map_Tensor), dim=2) 
    # ============================================================

    # Concatenate the Audios and EEG feature maps (48, 64) =======
    AE_Concat = torch.cat((EEG_feature_map_Tensor, A_Concat), dim=2)  
    # ============================================================


    # Load Target ================================================
    target = Dataset_info["attended ear"][i]
    if target == "L":
        # target = "stimuli1"
        target = 0
    else:
        # target = "stimuli2"
        target = 1
    # ============================================================
    
    #% Concatenate Dataset_info ==================================
    for j in range(0,EEG_feature_map.shape[0]):
    # for j in range(0,2): #for_test
        Dataset_info2 = [f"{i}_{j}",(EEG+f"_epoch{j}"),(Audio1+f"_epoch{j}"),(Audio2+f"_epoch{j}"),target]
        Dataset.append(Dataset_info2)
    # ============================================================
    
    
    # ============================================================
    features_array.append(AE_Concat)
    target_array.append(target)
    # ============================================================


    # torch.save(AE_Concat, f'./3_1_AE_Concats/feature_{i}.pt')
    # torch.save(target, f'./3_1_AE_Concats/target_{i}.pt')
    
Dataset = pd.DataFrame(Dataset)
Dataset.to_csv("Dataset.csv", index=False)

Dataset2 = [features_array, target_array]
torch.save(Dataset2, './Dataset.pt')

