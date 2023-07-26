#=================================================
#   ML_Project__Auditory Attention Detection (on a part of KULeuven Dataset)
#   1_2_Specified_EEG_Dataset_Extracter
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
import pandas as pd
import numpy as np


#%%
# Load the EEG_Dataset_info
Dataset_info = pd.read_csv("./Dataset_info.csv")
Dataset_info = Dataset_info.drop(Dataset_info[
    (Dataset_info['repetition'] == True) | 
    (Dataset_info['condition'] == "hrtf")].index)
Dataset_info = Dataset_info.reset_index(drop=True)

# Select the EEG datasets that pass our conditions
OurEEG_Names = []

# Build a for loop to do below steps one-by-one too all EEG datasets that pass our conditions
for i in range(0,Dataset_info.shape[0]):
    
    # Build a string that shows the name of our selected EEG datasets
    subject = Dataset_info["subject"][i]
    TrialID = Dataset_info["TrialID"][i]
    EEG = (f"{subject}_trials{TrialID}")
    
    # Load our selected EEG datasets
    EEG_dataset = pd.read_csv(f"./1_2_EEG_Original_csv_Files/{EEG}.csv")
    
    # Select specific columns that mentioned in reference papaer
    EEG_dataset_10Ch = EEG_dataset[['F3','F7','C3','T7','Pz','F4','F8','Cz','C4','T8']]
    
    # drop the first row of them witch show the column's number
    EEG_dataset_10Ch = EEG_dataset_10Ch.drop([0])
    
    # Build a specific name as string to save this new dataset and add them to a list
    EEG_dataset_10Ch_Name = (f"{EEG}_10Ch")
    OurEEG_Names.append(EEG_dataset_10Ch_Name)
    
    # Save specific coulumns of selected EEG datasets in csv files
    EEG_dataset_10Ch.to_csv(f"./1_3_EEG_10Ch_csv_Files/{EEG_dataset_10Ch_Name}.csv", sep = ',', index=False)

# Save a list of our selected EEG datasets' name
np.savetxt('./1_3_EEG_10Ch_csv_Files/OurEEG_Names.csv', OurEEG_Names, delimiter=',', fmt='%s')

