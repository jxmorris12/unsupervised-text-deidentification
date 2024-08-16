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
    false_positives_path = sys.argv[2] # path of .npy false positives file
    # false_positives_GT_path = sys.argv[3] # path of .npy false positives file
    dataset_path = sys.argv[3] # path of .parquet file containing the dataset of interest
    print("true_positives_path given : ", true_positives_path)
    print("false_positives_path given : ", false_positives_path)
    # print("false_positives_GT_path given : ", false_positives_GT_path)
    print("dataset_path given : ", dataset_path )
    
    listIndsOfRowsCorrect = [int(i) for i in np.load(true_positives_path).tolist()] 
    listIndsOfRowsIncorrect = [int(i) for i in np.load(false_positives_path).tolist()] 


    p_data_df = datasets.Dataset.from_parquet(dataset_path).to_pandas()
    p_data_df_correct = p_data_df[p_data_df['person_id'].isin(listIndsOfRowsCorrect)]
    p_data_df_incorrect = p_data_df[p_data_df['person_id'].isin(listIndsOfRowsIncorrect)]



    tpDirList = true_positives_path.split("/") # directories of true positives' path
    fpDirList = false_positives_path.split("/") # directories of false positives' path
    tpNewPath = os.path.join("/".join(tpDirList[:-1]), tpDirList[-1].split(".")[-2] + "_full.csv") # loc of the new file
    fpNewPath = os.path.join("/".join(fpDirList[:-1]), fpDirList[-1].split(".")[-2] + "_full.csv") # loc of the new file
 
    p_data_df_correct.reset_index(drop=True).to_csv(tpNewPath, index_label=False)
    p_data_df_incorrect.reset_index(drop=True).to_csv(fpNewPath, index_label=False)
 
    print("Correct preds file new path : ", tpNewPath)
    print("Incorrect preds file new path : ", fpNewPath)

if __name__ == "__main__":
    save_preds()
