% =================================================
%       ML_Project
%       EEG_Extracter_From_Dataset
%       Foad Moslem - PhD Student - Aerospace Engineering_Aerodynamics
%       Using MATLAB R2022a
% =================================================
clc; clear; close all;

% =================================================
DatasetDetails = [];

for i = 1:16
    Subject = "S"+i;
    FileName = "./1_1_EEG_Original_mat_Files/"+Subject+".mat";
    File = load(FileName);

    for j = 1:20
        %==============================
        trials = File.trials{1,j};
        VariableNames = {'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'};
        RawData_Channels = array2table(trials.RawData.Channels, 'VariableNames', VariableNames);
        RawData_EegData = array2table(trials.RawData.EegData, 'VariableNames', VariableNames);
        T = [RawData_Channels; RawData_EegData];
        
        Subject_trials = Subject+"_"+"trials"+j;
        trialsName = "./1_2_EEG_Original_csv_Files/"+Subject_trials+".csv";
        writetable(T, trialsName);
        
        %==============================
        subject = cellstr(trials.subject);
        TrialID = trials.TrialID;
        condition = cellstr(trials.condition);
        repetition = trials.repetition;
        experiment = trials.experiment;
        part = trials.part;
        attended_track = trials.attended_track;
        attended_ear = cellstr(trials.attended_ear);
        stimuli = trials.stimuli;
        
        T2 = [subject; TrialID; condition; repetition; experiment; part; attended_track; attended_ear; stimuli];
        DatasetDetails = [DatasetDetails; T2'];
        %==============================
    end
    
end

Variables = {'subject', 'TrialID', 'condition', 'repetition', 'experiment', 'part', 'attended_track', 'attended_ear', 'stimuli1', 'stimuli2'};
Dataset_info = array2table(DatasetDetails, 'VariableNames', Variables);
writetable(Dataset_info, "Dataset_info.csv");


