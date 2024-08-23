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
        
        # Join the directory path with the new filename
        new_path = os.path.join(dir_path, new_file_name)
        
        return new_path
    else:
        raise ValueError("The specified file is not a CSV file")

# path to csv having false positive GT notes with demos
fp_gt_path = sys.argv[1]
# path to csv having false positive preds notes with demos. 
fp_preds_path = sys.argv[2]
# Path where result/output of this program is stored
review_results_fp_path = convert_path(fp_gt_path)

print('review_results_fp_gt_path generated : ', review_results_fp_path)

fp_gt = pd.read_csv(fp_gt_path)
fp_preds = pd.read_csv(fp_preds_path)

# Update results file if exists, else create a new one
if os.path.exists(review_results_fp_path):
    print("THIS WILL CHANGE THE ORIGINAL REVIEW RESULTS FILE TOO, IF IT EXISTS. IS IT OKAY? PRESS 'Y' or 'N'")
    response = input()
    if response not in ['Y', 'N']:
        raise ValueError("Response invalid")
    review_results_fp = pd.read_csv(review_results_fp_path)    
else:
    print("Creating new result review file, as it does not exist.")
    # 'note_id_preds' needed since it might give some insights why mapping was done
    review_results_fp = pd.DataFrame(columns = ['note_id_gt', 'note_id_preds', 'reason'])

# random sampling to prevent bias
sampled_fp_gt = fp_gt.sample(n=100)

for index, row in sampled_fp_gt.iterrows():
          # If the current note has already been processed, move on!
          if row['note_id'] in list(review_results_fp['note_id_gt']):
              print("note_id already done. Moving to the next one!")
              continue
          # Prevent relevant demos and note for investigation
          print(row.drop(['note_text\n', 'Relevance_dict(handle None values properly)']))
          print(str(row[['note_text\n']].iloc[0]), str(row[['note_text\n']].iloc[0]))
          print(str(row[['Relevance_dict(handle None values properly)']].iloc[0]))
          print("for investigation...")
          breakpoint()
          reason = input("Possible factors contributing to the mapping: ")
          review_results_fp.loc[len(review_results_fp)] = [row['note_id'], reason]
          review_results_fp.to_csv(review_results_fp_path, index=False)
