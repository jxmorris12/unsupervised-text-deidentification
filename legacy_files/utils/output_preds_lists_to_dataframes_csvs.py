########
# This script takes true positive and false positive person_ids in the form of numpy arrays and returns dataframe containing full document and profiles for those person_ids
#######
import pandas as pd
import os
import datasets
import sys
import numpy as np
from update_or_create_relevancy_and_relevance_dict import process_data
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Get arguments for the program.')
    parser.add_argument('--tp_path', help="Give the path of the true positive file.", type=str, required=True)
    parser.add_argument('--fp_GT_path', help="Give the path of the false positive's GT file.", type=str, required=True)
    parser.add_argument('--fp_preds_path', help="Give the path of the false positive's preds file.", type=str, required=True)
    parser.add_argument('--dataset_path', help="Give the path of dataset(in parquet) file.", type=str, required=True)
    return parser.parse_args()

def save_preds(args):

#preds_df = pd.DataFrame(columns=["person_id", "note", "profile", "model's_prediction"])
    
    true_positives_path = args.tp_path # path of .npy true positives file
    false_positives_GT_path = args.fp_GT_path # path of .npy false positives GT file
    false_positives_preds_path = args.fp_preds_path # path of .npy false positives preds file
    dataset_path = args.dataset_path # path of .parquet file containing the dataset of interest
    print("true_positives_path given : ", true_positives_path)
    print("false_positives_path given : ", false_positives_path)
    # print("false_positives_GT_path given : ", false_positives_GT_path)
    print("dataset_path given : ", dataset_path )
    
    # Convert .npy list of person_ids to python list of person_ids    
    listIndsOfRowsCorrect = [int(i) for i in np.load(true_positives_path).tolist()]  # True postivies person_ids, in the form of python list
    listIndsOfRowsIncorrectGT = [int(i) for i in np.load(false_positives_GT_path).tolist()] # False positives GT person_ids, in the form of python list
    listIndsOfRowsIncorrectPreds = [int(i) for i in np.load(false_positives_preds_path).tolist()] # False positives Preds person_ids, in the form of python list
    
    false_positives_GT_to_preds_indices_dict = dict(zip(listIndsOfRowsIncorrectGT, listIndsOfRowsIncorrectPreds)) # Maps incorrect GT indexes to corresponding preds indexes

    p_data_df = datasets.Dataset.from_parquet(dataset_path).to_pandas()
    p_data_df_correct = p_data_df[p_data_df['person_id'].isin(listIndsOfRowsCorrect)] # those rows of p_data_df whose person ids are correctly mapped i.e true positive
    p_data_df_incorrect_GT = p_data_df[p_data_df['person_id'].isin(listIndsOfRowsIncorrectGT)] # those rows of p_data_df whose person_ids are incorrectly mapped i.e false positive GT
    p_data_df_indexes_for_values_of_false_positives_GT_to_preds_indices_dict = [p_data_df.index[p_data_df['person_id'] == false_positives_GT_to_preds_indices_dict[i]][0] for i in p_data_df_incorrect_GT['person_id']] # indices of those rows of p_data_df, whose person_ids are what incorrectly mapped person_ids are mapped to.
    p_data_df_incorrect_preds = p_data_df.loc[p_data_df_indexes_for_values_of_false_positives_GT_to_preds_indices_dict] # those rows of p_data_df, whose person_ids are what incorrectly mapped person_ids are mapped to.
    p_data_df_incorrect_preds_note_text_with_GT_demo = pd.DataFrame(p_data_df_incorrect_GT) # demo of incorrect_GT with notes of incorrect_preds. notes are put in below. This DataFrame is useful when analyzing false positives.
    p_data_df_incorrect_preds_note_text_with_GT_demo['note_text\n'] = list(p_data_df_incorrect_preds['note_text\n'])
    p_data_df_incorrect_preds_note_text_with_GT_demo['original_note_text\n'] = list(p_data_df_incorrect_preds['original_note_text\n'])

    tpDirList = true_positives_path.split("/") # directories of true positives' path
    fpGTDirList = false_positives_GT_path.split("/") # directories of false positives' GT path
    fpPredsDirList = false_positives_preds_path.split("/") # directories of false positives' preds path
    tpNewPath = os.path.join("/".join(tpDirList[:-1]), tpDirList[-1].split(".")[-2] + ".csv") # loc of the new file
    fpGTNewPath = os.path.join("/".join(fpGTDirList[:-1]), fpGTDirList[-1].split(".")[-2] + ".csv") # loc of the new file
    fpPredsNewPath = os.path.join("/".join(fpPredsDirList[:-1]), fpPredsDirList[-1].split(".")[-2] + ".csv") # loc of the new file
    fpPredsNotesWithGTDemoPath = os.path.join("/".join(fpPredsDirList[:-1]), fpPredsDirList[-1].split(".")[-2] + "_notes_with_demo_of_GT.csv") # loc of the new file

    # Updating relevancy and relvance dicts of the dataframes just created
    p_data_df_correct = process_data(p_data_df_correct)
    p_data_df_incorrect_GT = process_data(p_data_df_incorrect_GT)
    p_data_df_incorrect_preds = process_data(p_data_df_incorrect_preds)
    p_data_df_incorrect_preds_note_text_with_GT_demo = process_data(p_data_df_incorrect_preds_note_text_with_GT_demo)
 
    p_data_df_correct.reset_index(drop=True).to_csv(tpNewPath, index_label=False)
    p_data_df_incorrect_GT.reset_index(drop=True).to_csv(fpGTNewPath, index_label=False)
    p_data_df_incorrect_preds.reset_index(drop=True).to_csv(fpPredsNewPath, index_label=False)
    p_data_df_incorrect_preds_note_text_with_GT_demo.reset_index(drop=True).to_csv(fpPredsNotesWithGTDemoPath, index_label=False)
 
    print("Correct preds file's new path : ", tpNewPath)
    print("Incorrect preds GT file's new path : ", fpGTNewPath)
    print("Incorrect preds preds file's new path : ", fpPredsNewPath)
    print("Incorrect preds preds notes with demo of GT file's new path : ", fpPredsNotesWithGTDemoPath)

if __name__ == "__main__":
    args = get_arguments()
    save_preds(args)
