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

# Path of the true positive file
tp_path = sys.argv[1]
# Path where results are stored
review_results_tp_path = convert_path(tp_path)

print('review_results_tp_path generated : ', review_results_tp_path)

# Reads true positive file
tp = pd.read_csv(tp_path)

# Update if results file exists, else create a new one.
if os.path.exists(review_results_tp_path):
    print("THIS WILL CHANGE THE ORIGINAL REVIEW RESULTS FILE TOO, IF IT EXISTS. IS IT OKAY? PRESS 'Y' or 'N'")
    response = input()
    if response not in ['Y', 'N']:
        raise ValueError("Response invalid")
    # Read existing results file
    review_results_tp = pd.read_csv(review_results_tp_path)    
else:
    print("Creating new result review file, as it does not exist.")
    review_results_tp = pd.DataFrame(columns = ['note_id', 'reason'])


# Random sampling needed to prevent biases.
sampled_tp = tp.sample(n=100)

for index, row in sampled_tp.iterrows():
          # Has the node already been considered? If yes, move on.
          if row['note_id'] in list(review_results_tp['note_id']):
              print("note_id already done. Moving to the next one!")
              continue
          # Printing relevant demos and note
          print(row.drop(['note_text\n', 'Relevance_dict(handle None values properly)']))
          print(str(row[['note_text\n']].iloc[0]))
          print(str(row[['Relevance_dict(handle None values properly)']].iloc[0]))
          # print("for investigation...")
          # breakpoint()
          # Inputting posible reasons of why it turned out to be true positive
          reason = input("Possible factors contributing to the mapping: ")
          review_results_tp.loc[len(review_results_tp)] = [row['note_id'], reason]
          review_results_tp.to_csv(review_results_tp_path, index=False)
