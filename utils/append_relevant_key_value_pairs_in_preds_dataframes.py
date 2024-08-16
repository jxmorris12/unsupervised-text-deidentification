### Used to process two files : one which has data about correct predictions and other which has the data about incorrect predictions. 
### For both the file, we create two separate columns stating the key, value pairs found in those files, one contains 'nan' values and the other one doesn't.
### We use this file further down for better understanding of the results.


import pandas as pd 
import sys
import os

def main():
    
    correct_preds_path = sys.argv[1] # Contains the path of the correct preds file
    incorrect_preds_path = sys.argv[2] # Contains the path of the incorrect preds file


    c = pd.read_csv(correct_preds_path)
    i = pd.read_csv(incorrect_preds_path)
   
    relevant_keys = ["person_id", "note_id", "note_date", "note_datetime", "note_type", "empi_id", "mrn", "year_of_birth", "death_date", "death_datetime", 'address_1', 'address_2', 'city', 'state', 'zip'] # keys whose value we find in note_text  
 
    def get_with_nan_key_value_found_dict(row): # Returns key, values pairs(in form of a dict) such that the values are found in the document(or note_text)
        return dict([(key, row[key])  for key in relevant_keys if str(row[key]) in str(row['note_text\n'])]) 
    
    def get_without_nan_key_value_found_dict(row): # Same as previous function, but excludes the pairs when values are 'nan'
        return dict([(key, row[key])  for key in relevant_keys if str(row[key]) != "nan" and str(row[key]) in str(row['note_text\n'])])
    
    relevant_keys = ["person_id", "note_id", "note_date", "note_datetime", "note_type", "empi_id", "mrn", "year_of_birth", "death_date", "death_datetime", 'address_1', 'address_2', 'city', 'state', 'zip']
        
    c['with_nan_found_key_value_pairs'] = c.apply(lambda x : get_with_nan_key_value_found_dict(x), axis=1) 
    i['with_nan_found_key_value_pairs'] = i.apply(lambda x : get_with_nan_key_value_found_dict(x), axis=1) 
    
    c['without_nan_found_key_value_pairs'] = c.apply(lambda x : get_without_nan_key_value_found_dict(x), axis=1)      
    i['without_nan_found_key_value_pairs'] = i.apply(lambda x : get_without_nan_key_value_found_dict(x), axis=1)      

    cDirList = correct_preds_path.split("/")
    iDirList = incorrect_preds_path.split("/")
    cNewPath = os.path.join("/".join(cDirList[:-1]), cDirList[-1].split(".")[-2] + "_with_found_key_value_pairs.csv") # Loc of new file
    iNewPath = os.path.join("/".join(iDirList[:-1]), iDirList[-1].split(".")[-2] + "_with_found_key_value_pairs.csv") # Loc of new file

    c.to_csv(cNewPath, index_label=False)
    i.to_csv(iNewPath, index_label=False)

    print("Correct preds file new path : ", cNewPath)
    print("Incorrect preds file new path : ", iNewPath)
    
if __name__ == '__main__':
    main()
