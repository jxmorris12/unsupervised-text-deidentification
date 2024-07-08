import pandas as pd
import sys
import os

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

# dealing with true positives or false positives?
tp_or_fp = sys.argv[1]
if tp_or_fp not in ['tp', 'TP', 'fp', 'FP']:
   raise ValueError("Please specify the type of mapping you are dealine with as first argument. Possible answer : 'tp', 'TP', 'fp', 'FP'")
# path to csv having false positive GT notes with demos
tp_or_fp_gt_path = sys.argv[2]
if tp_or_fp in ['fp', 'FP']:
   # path to csv having false positive preds notes with demos. 
   fp_preds_path = sys.argv[3]
# Path where result/output of this program is stored
review_results_path = convert_path(tp_or_fp_gt_path)

print('review_results path generated : ', review_results_path)

tp_or_fp_gt = pd.read_csv(tp_or_fp_gt_path)
if tp_or_fp in ['fp', 'FP']:
  fp_preds = pd.read_csv(fp_preds_path)

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
    review_results = pd.DataFrame(columns = ['note_id_gt', 'note_id_preds', 'reason']) if tp_or_fp in ["FP", "fp"] else pd.DataFrame(columns = ['note_id', 'reason'])

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
          print(str(row[['Relevance_dict(handle None values properly)']].iloc[0]))
          if tp_or_fp in ['fp', 'FP']:
            breakpoint()
            print("\n\nPreds note : \n")
            print(fp_preds['note_text\n'])
          print("for investigation...")
          reason = input("Possible factors contributing to the mapping: ")
          review_results.loc[len(review_results)] = [row['note_id'], reason] if tp_or_fp not in ['fp', 'FP'] else [row['note_id'], fp_preds[fp_preds['note_id'] == row['note_id']]['note_id'], reason]
          review_results.to_csv(review_results_path, index=False)
