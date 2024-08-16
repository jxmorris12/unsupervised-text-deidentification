### Used to process two files : one which has data about correct predictions and other which has the data about incorrect predictions. 
### For both the file, we create two separate columns stating the key, value pairs found in those files, one contains 'nan' values and the other one doesn't.
### We use this file further down for better understanding of the results.


import pandas as pd 
import sys
import os

def main():
    
    correct_preds_path = sys.argv[1] # Contains the path of the correct preds file
    incorrect_preds_GT_path = sys.argv[2] # Contains the path of the incorrect preds' GT file
    incorrect_preds_preds_path = sys.argv[3] # Contains the path of the incorrect preds' preds file


    c = pd.read_csv(correct_preds_path)
    i_GT = pd.read_csv(incorrect_preds_GT_path)
    i_preds = pd.read_csv(incorrect_preds_preds_path)
   
    relevant_keys = ["person_id", "note_id", "note_date", "note_datetime", "note_type", "note_class", "empi_id", "mrn", "gender", "year_of_birth", 'month_of_birth', 'day_of_birth', 'race', 'ethnicity', "death_date", "death_datetime", 'address_1', 'address_2', 'city', 'state', 'zip'] # keys whose value we find in note_text  
 
    ### RELEVANT DICT IS CREATED AT THE TIME OF DATA CREATION. SO NO NEED TO CREATE AGAIN!

    # def get_key_value_pairs_found_dict(row): # Same as previous function, but excludes the pairs when values are 'nan'
    #    return dict([(key, row[key])  for key in relevant_keys if str(row[key]) != "nan" and str(row[key]) in str(row['note_text\n'])])
        
    # c['found_key_value_pairs'] = c.apply(lambda x : get_key_value_pairs_found_dict(x), axis=1) 
    # i_GT['found_key_value_pairs'] = i_GT.apply(lambda x : get_key_value_pairs_found_dict(x), axis=1) 
    # i_preds['found_key_value_pairs'] = i_preds.apply(lambda x : get_key_value_pairs_found_dict(x), axis=1) 

    cDirList = correct_preds_path.split("/")
    iGTDirList = incorrect_preds_GT_path.split("/")
    iPredsDirList = incorrect_preds_preds_path.split("/")
    cNewPath = os.path.join("/".join(cDirList[:-1]), cDirList[-1].split(".")[-2] + "_with_found_key_value_pairs.csv") # Loc of new file
    iGTNewPath = os.path.join("/".join(iGTDirList[:-1]), iGTDirList[-1].split(".")[-2] + "_with_found_key_value_pairs.csv") # Loc of new file
    iPredsNewPath = os.path.join("/".join(iPredsDirList[:-1]), iPredsDirList[-1].split(".")[-2] + "_with_found_key_value_pairs.csv") # Loc of new file

    c.to_csv(cNewPath, index_label=False)
    i_GT.to_csv(iGTNewPath, index_label=False)
    i_preds.to_csv(iPredsNewPath, index_label=False)

    print("Correct preds file new path : ", cNewPath)
    print("Incorrect preds' GT file new path : ", iGTNewPath)
    print("Incorrect preds' preds file new path : ", iPredsNewPath)
    
if __name__ == '__main__':
    main()
