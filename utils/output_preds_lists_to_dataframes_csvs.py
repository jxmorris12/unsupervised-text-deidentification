########
# This script takes true positive and false positive person_ids in the form of numpy arrays and returns dataframe containing full document and profiles for those person_ids
#######
import pandas as pd
import os
import datasets
import sys
import numpy as np

def save_preds():

#preds_df = pd.DataFrame(columns=["person_id", "note", "profile", "model's_prediction"])
    
    true_positives_path = sys.argv[1] # path of .npy true positives file
    false_positives_GT_path = sys.argv[2] # path of .npy false positives GT file
    false_positives_preds_path = sys.argv[3] # path of .npy false positives preds file
    dataset_path = sys.argv[4] # path of .parquet file containing the dataset of interest
    print("true_positives_path given : ", true_positives_path)
    print("false_positives_GT_path given : ", false_positives_GT_path)
    print("false_positives_preds_path given : ", false_positives_preds_path)
    print("dataset_path(parquet format) given : ", dataset_path)
    
    # Convert .npy list of person_ids to python list of person_ids    
    listIndsOfRowsCorrect = [int(i) for i in np.load(true_positives_path).tolist()] 
    listIndsOfRowsIncorrectGT = [int(i) for i in np.load(false_positives_GT_path).tolist()] 
    listIndsOfRowsIncorrectPreds = [int(i) for i in np.load(false_positives_preds_path).tolist()] 


    p_data_df = datasets.Dataset.from_parquet(dataset_path).to_pandas()
    p_data_df_correct = p_data_df[p_data_df['person_id'].isin(listIndsOfRowsCorrect)]
    p_data_df_incorrect_GT = p_data_df[p_data_df['person_id'].isin(listIndsOfRowsIncorrectGT)]
    p_data_df_incorrect_preds = p_data_df[p_data_df['person_id'].isin(listIndsOfRowsIncorrectPreds)]



    tpDirList = true_positives_path.split("/") # directories of true positives' path
    fpGTDirList = false_positives_GT_path.split("/") # directories of false positives' GT path
    fpPredsDirList = false_positives_preds_path.split("/") # directories of false positives' preds path
    tpNewPath = os.path.join("/".join(tpDirList[:-1]), tpDirList[-1].split(".")[-2] + ".csv") # loc of the new file
    fpGTNewPath = os.path.join("/".join(fpGTDirList[:-1]), fpGTDirList[-1].split(".")[-2] + ".csv") # loc of the new file
    fpPredsNewPath = os.path.join("/".join(fpPredsDirList[:-1]), fpPredsDirList[-1].split(".")[-2] + ".csv") # loc of the new file
 
    p_data_df_correct.reset_index(drop=True).to_csv(tpNewPath, index_label=False)
    p_data_df_incorrect_GT.reset_index(drop=True).to_csv(fpGTNewPath, index_label=False)
    p_data_df_incorrect_preds.reset_index(drop=True).to_csv(fpPredsNewPath, index_label=False)
 
    print("Correct preds file new path : ", tpNewPath)
    print("Incorrect preds GT file new path : ", fpGTNewPath)
    print("Incorrect preds preds file new path : ", fpPredsNewPath)

if __name__ == "__main__":
    save_preds()
