from datetime import datetime, date, time, timezone
import os
import pyarrow as pa
import pandas as pd
import datasets
import transformers
import sys

print("THIS FILE OVERWRITES THE GIVEN ORIGINAL FILE, MAKE SURE YOU ARE OKAY WITH IT!")

# Configuration variables
input_filepath = sys.argv[1]
print("input_filepath : ", input_filepath)

# Function to process your data
# @ profile
def process_data(df):
    # Perform your data processing here
    # Example: df['new_column'] = df['old_column'] * 2
    # DATA CLEANING
    
    ### We want to process different kinds of keys differently, hence this dict. Reason being that the structure of profile value in profile may differ with the structure of that profile value in the note. Ex : date may change from dd-mm-yyyy to dd/mm/yyyy 
    relevant_keys = {'dates' : ['death_date', ['year_of_birth', 'month_of_birth', 'day_of_birth']],
                     'date_times' : ['note_datetime', 'death_datetime'],
                     'texts' : ["note_type", "note_class", "gender", "race", "ethnicity", 'address_1', 'address_2', 'city', 'state'],
                     'numbers' : ["person_id", "note_id", "empi_id", "mrn", "zip"]
}
    print("CONVERTING NOTES COLUMN TO A LIST OF STRINGS")
    notes_list = df['note_text\n'].astype(str).to_list() # astype is required, as some value(s) in was found to be nan
    print("CONVERTING NOTES COLUMN TO A LIST OF STRINGS DONE")
    notes_list_lower = [note.lower() for note in notes_list]
    global row_ind 
    row_ind = 0
    # @ profile
    def get_row_relevancy_data(row): # Whether any value is found in note_text
        global row_ind
        relevant_dict = {}
        sum_ = 0
        note = notes_list[row_ind]
        nulls = ["nan", "None"]
        for key_type in relevant_keys.keys():
            if key_type == "dates":
                date_format = '%Y-%m-%d' # Assert to check if this is correct
                for key in relevant_keys[key_type]: 
                    if type(key) == list: # y, m, d of dates separated
                        y = row[key[0]]
                        m = row[key[1]]
                        d = row[key[2]]
                        if y in nulls or m in nulls or d in nulls:
                            continue
                        date_var = date(y, m, d)
                    else:
                        if str(row[key]) in nulls:
                            continue
                        assert datetime.strptime(str(row[key]), date_format)
                        date_var = datetime.strptime(str(row[key]), date_format)
                    date_formats = ['%m/%d/%Y', '%m-%d-%Y', '%-m/%-d/%Y', '%-m-%-d-%Y']
                    for fmt in date_formats:
                        formatted_date = date_var.strftime(fmt)
                        if formatted_date in str(note):
                            relevant_dict['Birth or Death date'] = str(formatted_date) 
                            sum_ += 1
                            break

            elif key_type == "date_times":
                for key in relevant_keys[key_type]:
                    if str(row[key]) in nulls:
                        continue
                    # The datetime string to be converted
                    date_time_original = str(row[key])
                    datetime_string = date_time_original.split('.')[0] # It is found that some time contain fractional seconds, which we can safely ignore
                    assert datetime_string in date_time_original # To make sure that if the note contains date_time_original, it will be matched with datetime_string
                    # if date_time_original != datetime_string: # To manually check what kind of fractional time there is.
                    #    print(date_time_original)
                    # The format of the datetime string
                    datetime_format = '%Y-%m-%d %H:%M:%S'
                    assert datetime.strptime(datetime_string, datetime_format) 
                    datetime_var = datetime.strptime(datetime_string, datetime_format)
                    date_var = datetime_var.date()
                    time = datetime_var.time()
                    formatted_time_without_leading_zeros = time.strftime('%-H:%-M:%-S')
                    formatted_time_with_leading_zeros = time.strftime('%H:%M:%S')
                    if formatted_time_without_leading_zeros in str(note):
                        relevant_dict[key] = str(formatted_time_without_leading_zeros)
                        sum_ += 1
                        continue
                    elif formatted_time_with_leading_zeros in str(note):
                        relevant_dict[key] = str(formatted_time_with_leading_zeros)
                        sum_ += 1
                        continue
                    date_formats = ['%m/%d/%Y', '%m-%d-%Y', '%-m/%-d/%Y', '%-m-%-d-%Y']
                    for fmt in date_formats:
                        formatted_date = date_var.strftime(fmt)
                        if formatted_date in str(note):
                            relevant_dict[key] = str(formatted_date)
                            sum_ += 1
                            break 
            else:
                for key in relevant_keys[key_type]:
                    if str(row[key]) in nulls:
                        continue
                    if str(row[key]).lower() in notes_list_lower[row_ind]:
                        relevant_dict[key] = str(row[key]).lower()
                        sum_ += 1
                     
        ret = [True if sum_ > 0 else False, sum_, relevant_dict]
        row_ind += 1
        return ret
    
    
    print("GETTING RELEVANT ROWS INDICES(TAKING TIME)")
    relevancy_data = df.apply(lambda row : get_row_relevancy_data(row), axis=1)
    
    print("GETTING INDIVIDUAL DATA FROM RELEVANCY DATA")
    
    relevant_rows_bools = relevancy_data.apply(lambda x : x[0]) # Not required as of now as some relevant rows may have zero values of keys found in it
    relevance_values = relevancy_data.apply(lambda x : x[1])
    relevance_dicts = relevancy_data.apply(lambda x : x[2])
    
    
    print("APPENDING INDIVIDUAL DATA OBTAINED TO THE DATAFRAME")
    
    df["Relevance"] = relevance_values
    df["Relevance_dict(handle None values properly)"] = relevance_dicts
    
    return df

df = pd.read_csv(input_filepath)
processed_df = process_data(df)
processed_df.to_csv(input_filepath)
