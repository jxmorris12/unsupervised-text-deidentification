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

tp_path = sys.argv[1]
review_results_tp_path = convert_path(tp_path)

print('review_results_tp_path generated : ', review_results_tp_path)

tp = pd.read_csv(tp_path)

if os.path.exists(review_results_tp_path):
    print("THIS WILL CHANGE THE ORIGINAL REVIEW RESULTS FILE TOO, IF IT EXISTS. IS IT OKAY? PRESS 'Y' or 'N'")
    response = input()
    if response not in ['Y', 'N']:
        raise ValueError("Response invalid")
    review_results_tp = pd.read_csv(review_results_tp_path)    
else:
    print("Creating new result review file, as it does not exist.")
    review_results_tp = pd.DataFrame(columns = ['note_id', 'reason'])


sampled_tp = tp.sample(n=100)

for index, row in sampled_tp.iterrows():
          if row['note_id'] in list(review_results_tp['note_id']):
              print("note_id already done. Moving to the next one!")
              continue
          print(row.drop(['note_text\n', 'Relevance_dict(handle None values properly)']))
          print(str(row[['note_text\n']].iloc[0]))
          print(str(row[['Relevance_dict(handle None values properly)']].iloc[0]))
          print("for investigation...")
          breakpoint()
          reason = input("Reason for decision: ")
          review_results_tp.loc[len(review_results_tp)] = [row['note_id'], reason]
