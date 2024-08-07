import pandas as pd
import ast
import sys
import os
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Get arguments for the program', add_help=False)
    parser.add_argument('--tp_or_fp', choices=['tp', 'fp', 'TP', 'FP'], help="Do you want to process the results for true positive mappings or false positive mappings?", type=str, required=True)
    parser.add_argument("--tp_path", type=str, help="Get path of the true positives results file in csv format.")
    parser.add_argument("--fp_gt_path", type=str, help="Get path of the false positives results file with ground truth notes, in csv format.")
    parser.add_argument("--fp_preds_path", type=str, help="Get path of the false positives preds results file, in csv format.")
    parser.add_argument("--fp_preds_with_gt_demo", type=str, help="Get path of the false positives results file with predicted notes and ground truth demographics, in csv format.")
    
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
    relevance_dict_at_index = ast.literal_eval(df.loc[index]['Relevance_dict(handle None values properly)'])
    all_relevant_dicts = list(df['Relevance_dict(handle None values properly)'].map(lambda x : ast.literal_eval(x)))
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
    # path to csv having false positive preds notes. 
    fp_preds_path = args.fp_preds_path
    # path to csv having false positive preds notes with demos of GT person_id. 
    fp_preds_note_with_demo_from_GT_path = args.fp_preds_with_gt_demo
# Path where result/output of this program is stored
review_results_path = convert_path(tp_or_fp_gt_path)

print('review_results path generated : ', review_results_path)

tp_or_fp_gt = pd.read_csv(tp_or_fp_gt_path)
if tp_or_fp in ['fp', 'FP']:
  raise error("Need to modify program for fp")
  fp_preds = pd.read_csv(fp_preds_path)
  fp_preds_note_with_demo_from_GT = pd.read_csv(fp_preds_note_with_demo_from_GT_path)

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
    review_results = pd.DataFrame(columns = ['note_id_gt', 'reason', 'conditional random probability']) if tp_or_fp in ["FP", "fp"] else pd.DataFrame(columns = ['note_id', 'reason', 'conditional random probability'])

# random sampling to prevent bias
sample_tp_or_fp_gt = tp_or_fp_gt.sample(n=100)

for index, row in sample_tp_or_fp_gt.iterrows():
          # If the current note has already been processed, move on!
          if row['note_id'] in list((review_results['note_id_gt'] if tp_or_fp in ['fp', 'FP'] else review_results['note_id'])):
              print("note_id already done. Moving to the next one!")
              continue
          if tp_or_fp in ['fp', 'FP']:
            print("\n\nGT demos : \n")
          # Prevent relevant demos and note for investigation
          print(row.drop(['note_text\n', 'Relevance_dict(handle None values properly)']))
          if tp_or_fp in ['fp', 'FP']:
            print("\n\nGT note : \n")
          print(str(row[['note_text\n']].iloc[0]))
          print("Relevance Dict : ", str(row[['Relevance_dict(handle None values properly)']].iloc[0]))
          print("Person Id : ", str(row[['person_id']].iloc[0]))
          print("Note Id : ", str(row[['note_id']].iloc[0]))
          CRP = get_conditional_random_prob_taking_superset_relevant_dicts_into_account(tp_or_fp_gt, index)
          print("Conditional random prob taking superset relevant dicts into account : ", CRP)
          # breakpoint()
          if tp_or_fp in ['fp', 'FP']:
            print("\n\nPreds's demos : \n")
            print(fp_preds.iloc[index].drop(['note_text\n', 'Relevance_dict(handle None values properly)']))
            print("\n\nPreds note : \n")
            print(fp_preds.iloc[index]['note_text\n'])
            print(fp_preds.iloc[index]['Relevance_dict(handle None values properly)'])
            print("Person Id : ", str(fp_preds_note.iloc[index][['person_id']].iloc[0]))
            print("Note Id : ", str(fp_preds_note.iloc[index][['note_id']].iloc[0]))
            print("\n\nPreds's GT demos : \n")
            print(fp_preds_note_with_demo_from_GT.iloc[index].drop(['note_text\n', 'Relevance_dict(handle None values properly)']))
            print("\n\nPreds note : \n")
            print(fp_preds_note_with_demo_from_GT.iloc[index]['note_text\n'])
            print(fp_preds_note_with_demo_from_GT.iloc[index]['Relevance_dict(handle None values properly)'])
            print("Person Id : ", str(fp_preds_note_with_demo_from_GT.iloc[index][['person_id']].iloc[0]))
            print("Note Id : ", str(fp_preds_note_with_demo_from_GT.iloc[index][['note_id']].iloc[0]))
          print("\nfor investigation...\n\n")
          reason = input(f"Possible factors contributing to the{' incorrect' if tp_or_fp in ['fp', 'FP'] else ' correct'} mapping: ")
          review_results.loc[len(review_results)] = [row['note_id'], reason, CRP]
          review_results.to_csv(review_results_path, index=False)
