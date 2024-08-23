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
    parser.add_argument('--tp_mappings_path', help="Give the path of the true positive mappings file.", type=str, required=True)
    parser.add_argument('--fp_mappings_path', help="Give the path of the false positive mappings file.", type=str, required=True)
    parser.add_argument('--dataset_parquet_path', help="Give the path of dataset(in parquet) file.", type=str, required=True)
    return parser.parse_args()

def save_preds(args):

#preds_df = pd.DataFrame(columns=["person_id", "note", "profile", "model's_prediction"])
    
    true_positives_mappings_path = args.tp_mappings_path # path of .npy true positives mappings file
    false_positives_mappings_path = args.fp_mappings_path # path of .npy false positives mappings file
    dataset_parquet_path = args.dataset_parquet_path # path of .parquet file containing the dataset of interest
    print("true_positives_mappings_path given : ", true_positives_mappings_path)
    print("false_positives_mappings_path given : ", false_positives_mappings_path)
    print("dataset_parquet_path given : ", dataset_parquet_path)
    # Loadings the mappings
    true_positives_mappings = np.load(true_positives_mappings_path)
    false_positives_mappings = np.load(false_positives_mappings_path)
    
    # Convert .npy list of person_ids to python list of person_ids    
    correctPredsPersonIdsList = [int(i) for i in true_positives_mappings[:, 0].tolist()]  # True postivies person_ids, in the form of python list
    incorrectPredsGTPersonIdsList = [int(i) for i in false_positives_mappings[:, 0].tolist()] # False positives GT person_ids, in the form of python list
    incorrectPredsPredictedPersonIdsList = [int(i) for i in false_positives_mappings[:, 1].tolist()] # False positives Preds person_ids, in the form of python list
    
    false_positives_GT_person_ids_to_predicted_person_ids_dict = dict(zip(incorrectPredsGTPersonIdsList, incorrectPredsPredictedPersonIdsList)) # Maps incorrect GT notes' demographics' person ids to corresponding predicted demographics' person ids

    p_data_df = datasets.Dataset.from_parquet(dataset_parquet_path).to_pandas()
    p_data_df_correct = p_data_df[p_data_df['person_id'].isin(correctPredsPersonIdsList)] # those rows of p_data_df whose notes are correctly mapped i.e true positive
    p_data_df_incorrect_GT = p_data_df[p_data_df['person_id'].isin(incorrectPredsGTPersonIdsList)] # those rows of p_data_df whose notes are incorrectly mapped i.e false positive GT
    p_data_df_indexes_for_values_of_false_positives_GT_person_ids_to_predicted_person_ids_dict = [p_data_df.index[p_data_df['person_id'] == false_positives_GT_person_ids_to_predicted_person_ids_dict[i]][0] for i in p_data_df_incorrect_GT['person_id']] # indices of those rows of p_data_df, whose demographics are what incorrectly mapped notes are mapped to.
    p_data_df_incorrect_preds = p_data_df.loc[p_data_df_indexes_for_values_of_false_positives_GT_person_ids_to_predicted_person_ids_dict] # those rows of p_data_df, whose demographics are what incorrectly mapped notes are mapped to.
    p_data_df_incorrect_predicted_demo_with_note_that_is_predicted = pd.DataFrame(p_data_df_incorrect_preds) # predicted demographics of incorrect predictions with notes that are mapped. notes are put in below. This DataFrame is useful when analyzing false positives.
    p_data_df_incorrect_predicted_demo_with_note_that_is_predicted['note_text\n'] = list(p_data_df_incorrect_GT['note_text\n'])

    tpDirList = true_positives_mappings_path.split("/") # directories of true positives' path
    fpMappingsDirList = false_positives_mappings_path.split("/") # directories of false positives' mappings path
    tpNewPath = os.path.join("/".join(tpDirList[:-1]), tpDirList[-1].split(".")[-2] + ".csv") # loc of the new file
    fpGTNewPath = os.path.join("/".join(fpMappingsDirList[:-1]), fpMappingsDirList[-1].split(".")[-2] + "_GT.csv") # loc of the new file
    fpPredsPredictedDemoWithNoteThatIsPredictedNewPath = os.path.join("/".join(fpMappingsDirList[:-1]), fpMappingsDirList[-1].split(".")[-2] + "_predicted_demo_with_note_that_is_predicted.csv") # loc of the new file

    # Updating relevancy and relvance dicts of the dataframes just created
    p_data_df_correct = process_data(p_data_df_correct)
    p_data_df_incorrect_GT = process_data(p_data_df_incorrect_GT)
    p_data_df_incorrect_predicted_demo_with_note_that_is_predicted = process_data(p_data_df_incorrect_predicted_demo_with_note_that_is_predicted)
   
    # Adding mappings probabilites
    def get_probability(row, mappings, mappings_dimension_of_concern):
        person_id = str(row['person_id'])
        index = np.where(mappings[:, mappings_dimension_of_concern] == person_id)[0]
        probability = float(mappings[index, mappings_dimension_of_concern + 1][0])
        return probability
    p_data_df_correct['Mapping Probability'] = p_data_df_correct.apply(lambda row : get_probability(row, true_positives_mappings, 0), axis=1)
    p_data_df_incorrect_predicted_demo_with_note_that_is_predicted['Mapping Probability'] = p_data_df_incorrect_predicted_demo_with_note_that_is_predicted.apply(lambda row : get_probability(row, false_positives_mappings, 1), axis=1)
 
    # Saving dataframes
    p_data_df_correct.reset_index(drop=True).to_csv(tpNewPath, index_label=False)
    p_data_df_incorrect_GT.reset_index(drop=True).to_csv(fpGTNewPath, index_label=False)
    p_data_df_incorrect_predicted_demo_with_note_that_is_predicted.reset_index(drop=True).to_csv(fpPredsPredictedDemoWithNoteThatIsPredictedNewPath, index_label=False)
 
    print("Correct preds file's new path : ", tpNewPath)
    print("Incorrect preds GT file's new path : ", fpGTNewPath)
    print("Incorrect preds predicted demos with notes that are predicted file's new path : ", fpPredsPredictedDemoWithNoteThatIsPredictedNewPath)

if __name__ == "__main__":
    args = get_arguments()
    save_preds(args)
