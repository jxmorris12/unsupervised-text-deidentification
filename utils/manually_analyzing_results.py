import pandas as pd
import ast
import sys
import os
import argparse
from update_or_create_relevancy_and_relevance_dict import process_data

def get_arguments():
    parser = argparse.ArgumentParser(description='Get arguments for the program', add_help=False)
    parser.add_argument('--tp_or_fp', choices=['tp', 'fp', 'TP', 'FP'], help="Do you want to process the results for true positive mappings or false positive mappings?", type=str, required=True)
    parser.add_argument("--tp_path", type=str, help="Get path of the true positives results file in csv format.")
    parser.add_argument("--fp_gt_path", type=str, help="Get path of the false positives results file with ground truth notes, in csv format.")
    parser.add_argument("--fp_preds_demo_with_note_that_is_predicted_path", type=str, help="Get path of the false positives results file with predicted notes and ground truth demographics, in csv format.")
    
    # Parse the argument list
    args = parser.parse_args()
    
    return args
    
def convert_path(original_path):
    # Split the path into directory and filename
    dir_path, file_name = os.path.split(original_path)
    
    # Check if the file is a CSV file
    if file_name.endswith('.csv'):
        # Prepend 'review_results_' to the filename
        new_file_name = "review_results_" + file_name
        new_file_name = new_file_name.replace("gt", "")
        new_file_name = new_file_name.replace("Gt", "")
        new_file_name = new_file_name.replace("GT", "")
        new_file_name = new_file_name.replace("preds", "")
        new_file_name = new_file_name.replace("Preds", "")
        new_file_name = new_file_name.replace("PREDS", "")
        
        # Join the directory path with the new filename
        new_path = os.path.join(dir_path, new_file_name)
        
        return new_path
    else:
        raise ValueError("The specified file is not a CSV file")

def is_subdict(subdict, superdict):
    return set(subdict.items()).issubset(set(superdict.items()))

def get_conditional_random_prob_taking_superset_relevant_dicts_into_account(df, index):
    orig_df = df
    df = pd.DataFrame(orig_df)
    df['note_text\n'] = df.iloc[index]['note_text\n']
    df = process_data(df)
    relevance_dict_at_index = (df.loc[index]['Relevance_dict(handle None values properly)'])
    all_relevant_dicts = list(df['Relevance_dict(handle None values properly)'])
    is_subdict_bool_list = list(map(lambda relevant_dict : is_subdict(relevance_dict_at_index, relevant_dict), all_relevant_dicts))
    conditional_random_prob = 1 / is_subdict_bool_list.count(True)
    return conditional_random_prob

args = get_arguments()
tp_or_fp = args.tp_or_fp
if tp_or_fp not in ['tp', 'TP', 'fp', 'FP']:
   raise ValueError("Please specify the type of mapping you are dealing with as first argument. Possible answer : 'tp', 'TP', 'fp', 'FP'")
if tp_or_fp in ["tp", 'TP']:
    # path to csv having false positive GT notes with demos
    tp_or_fp_gt_path = args.tp_path
if tp_or_fp in ['fp', 'FP']:
    tp_or_fp_gt_path = args.fp_gt_path
    # path to csv having false positive preds demographics with corresponding notes that are predicted. 
    fp_preds_demo_with_note_that_is_predicted_path = args.fp_preds_demo_with_note_that_is_predicted_path
# Path where result/output of this program is stored
review_results_path = convert_path(tp_or_fp_gt_path)

print('review_results path generated : ', review_results_path)

tp_or_fp_gt = pd.read_csv(tp_or_fp_gt_path)
if tp_or_fp in ['fp', 'FP']:
  fp_preds_demo_with_note_that_is_predicted = pd.read_csv(fp_preds_demo_with_note_that_is_predicted_path)

# Update results file if exists, else create a new one
if os.path.exists(review_results_path):
    print("THIS WILL CHANGE THE ORIGINAL REVIEW RESULTS FILE TOO, IF IT EXISTS. IS IT OKAY? PRESS 'Y' or 'N'")
    response = input()
    if response not in ['Y', 'N']:
        raise ValueError("Response invalid")
    review_results = pd.read_csv(review_results_path)    
else:
    print("Creating new result review file, as it does not exist.")
    # 'note_id_preds' needed since it might give some insights why mapping was done
    review_results = pd.DataFrame(columns = ['id_of_note_that_is_predicted', 'predicted_demo_person_id', 'reason', 'conditional random probability', 'mapping probability']) if tp_or_fp in ["FP", "fp"] else pd.DataFrame(columns = ['note_id', 'reason', 'conditional random probability', 'mapping probability'])

# random sampling to prevent bias
sample_tp_or_fp_gt = tp_or_fp_gt.sample(n=100)

for index, row in sample_tp_or_fp_gt.iterrows():
          # If the current note has already been processed, move on!
          if row['note_id'] in list((review_results['id_of_note_that_is_predicted'] if tp_or_fp in ['fp', 'FP'] else review_results['note_id'])):
              print("note_id already done. Moving to the next one!")
              continue
          CRP = get_conditional_random_prob_taking_superset_relevant_dicts_into_account(tp_or_fp_gt, index)
          if tp_or_fp in ['tp', 'TP']:
            print("\n\nRow data :\n\n\n")
            print(row.drop(['note_text\n', "Relevance_dict(handle None values properly)"])) # Dropping values so that they won't be trimmed when they will be printed like below.
            print("\n\nRow note text :\n\n\n")
            print(str(row['note_text\n'])) # No trimming
            print("\n\nRelevance dict :\n\n\n")
            print(str(row["Relevance_dict(handle None values properly)"]), "\n\n\n") # No trimming
            print("\n\nConditional random prob taking superset relevant dicts into account : ", CRP)
          if tp_or_fp in ['fp', 'FP']:
            print("\n\nGT demographics : \n\n\n")
            print(row.drop(['note_text\n', "Relevance_dict(handle None values properly)"])) # Dropping rel dict to avoid trimming. Note_text should be printed only once, as will be done later.
            print("\n\nGT Relevance dict :\n\n\n")
            print(str(row["Relevance_dict(handle None values properly)"]), "\n\n\n") # No trimming
            print("\n\nPredicted Demographics : \n\n\n")
            print(fp_preds_demo_with_note_that_is_predicted.iloc[index].drop(['note_text\n', "Relevance_dict(handle None values properly)"]))
            print("\n\nPredicted demographics' Relevance dict :\n\n\n")
            print(str(fp_preds_demo_with_note_that_is_predicted.iloc[index]["Relevance_dict(handle None values properly)"]), "\n\n\n") # No trimming
            assert row['note_text\n'] == fp_preds_demo_with_note_that_is_predicted.iloc[index]['note_text\n']
            print("\n\nCommon note : \n\n\n")
            print(row['note_text\n'])
            print("\n\nCommon note(unmasked) : \n\n\n")
            print(row['original_note_text\n'])
            print("\n\nConditional random prob taking superset relevant dicts into account : \n\n\n", CRP)
            
          #if tp_or_fp in ['fp', 'FP']:
          #  print("\n\nGT demos : \n")
          ## Present relevant demos and note for investigation
          #print(row.drop(['note_text\n']))
          #if tp_or_fp in ['fp', 'FP']:
          #  print("\n\nPredicted demo : \n")
          #print(str(row[['note_text\n']].iloc[0]))
          #print("Relevance Dict : ", str(row[['Relevance_dict(handle None values properly)']].iloc[0]))
          #print("Person Id : ", str(row[['person_id']].iloc[0]))
          #print("Note Id : ", str(row[['note_id']].iloc[0]))
          # CRP = get_conditional_random_prob_taking_superset_relevant_dicts_into_account(tp_or_fp_gt, index)
          #print("Conditional random prob taking superset relevant dicts into account : ", CRP)
          ## breakpoint()
          #if tp_or_fp in ['fp', 'FP']:
          #  print("\n\nPreds's demos : \n")
          #  print(fp_preds.iloc[index].drop(['note_text\n', 'Relevance_dict(handle None values properly)']))
          #  print("\n\nPreds note : \n")
          #  print(fp_preds.iloc[index]['note_text\n'])
          #  print(fp_preds.iloc[index]['Relevance_dict(handle None values properly)'])
          #  print("Person Id : ", str(fp_preds_note.iloc[index][['person_id']].iloc[0]))
          #  print("Note Id : ", str(fp_preds_note.iloc[index][['note_id']].iloc[0]))
          #  print("\n\nPreds's GT demos : \n")
          #  print(fp_preds_note_with_demo_from_GT.iloc[index].drop(['note_text\n', 'Relevance_dict(handle None values properly)']))
          #  print("\n\nPreds note : \n")
          #  print(fp_preds_note_with_demo_from_GT.iloc[index]['note_text\n'])
          # print(fp_preds_note_with_demo_from_GT.iloc[index]['Relevance_dict(handle None values properly)'])
          #  print("Person Id : ", str(fp_preds_note_with_demo_from_GT.iloc[index][['person_id']].iloc[0]))
          #  print("Note Id : ", str(fp_preds_note_with_demo_from_GT.iloc[index][['note_id']].iloc[0]))
          print("\nfor investigation...\n\n")
          reason = input(f"Possible factors contributing to the{' incorrect' if tp_or_fp in ['fp', 'FP'] else ' correct'} mapping: ")
          review_results.loc[len(review_results)] = [row['note_id'], fp_preds_demo_with_note_that_is_predicted.iloc[index]['person_id'], reason, CRP, fp_preds_demo_with_note_that_is_predicted.iloc[index]['Mapping Probability']] if tp_or_fp in ["FP", "fp"] else [row['note_id'], reason, CRP, row['Mapping Probability']]
          review_results.to_csv(review_results_path, index=False)
