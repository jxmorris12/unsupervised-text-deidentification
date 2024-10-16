from datetime import datetime, date, time, timezone
import os
import pyarrow as pa
import pandas as pd
import datasets
import transformers
import sys

# Function to process your data
# @ profile
def mask_identifier_in_the_data(df, identifier, identifier_type):
    # Perform your data processing here
    # Example: df['new_column'] = df['old_column'] * 2
    # DATA CLEANING
    
    ### We want to process different kinds of identifers differently, hence this dict. Reason being that the structure of profile value in profile may differ with the structure of that profile value in the note. Ex : date may change from dd-mm-yyyy to dd/mm/yyyy 
    print("CONVERTING NOTES COLUMN TO A LIST OF STRINGS")
    notes_list = df['note_text\n'].astype(str).to_list() # astype is required, as some value(s) in was found to be nan
    print("CONVERTING NOTES COLUMN TO A LIST OF STRINGS DONE")
    notes_list_lower = [note.lower() for note in notes_list]
    global row_ind 
    row_ind = 0
    # @ profile
    def get_row_relevancy_data(row, identifer, identifier_type): # Whether any value is found in note_text
        global row_ind
        sum_ = 0
        note = notes_list[row_ind]
        row_ind += 1
        nulls = ["nan", "None"]
        if True:
            if identifier_type == "dates":
                date_format = '%Y-%m-%d' # Assert to check if this is correct
                if True:
                    if type(identifer) == list: # y, m, d of dates separated
                        y = row[identifer[0]]
                        m = row[identifer[1]]
                        d = row[identifer[2]]
                        if y in nulls or m in nulls or d in nulls:
                            return str(note)
                        date_var = date(y, m, d)
                    else:
                        if str(row[identifer]) in nulls:
                            return str(note)
                        assert datetime.strptime(str(row[identifer]), date_format)
                        date_var = datetime.strptime(str(row[identifer]), date_format)
                    date_formats = ['%m/%d/%Y', '%m-%d-%Y', '%-m/%-d/%Y', '%-m-%-d-%Y']
                    for fmt in date_formats:
                        formatted_date = date_var.strftime(fmt)
                        if formatted_date in str(note):
                            new_note = str(note).replace(formatted_date, '*' * len(formatted_date))
                            return new_note
            elif identifier_type == "date_times":
                if True:
                    if str(row[identifer]) in nulls:
                        return str(note)
                    # The datetime string to be converted
                    date_time_original = str(row[identifer])
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
                        new_note = str(note).replace(formatted_time_without_leading_zeros, '*' * len(formatted_time_without_leading_zeros))
                        return new_note
                    elif formatted_time_with_leading_zeros in str(note):
                        # relevant_dict[identifer] = str(formatted_time_with_leading_zeros)
                        new_note = str(note).replace(formatted_time_with_leading_zeros, '*' * len(formatted_time_with_leading_zeros))
                        return new_note
                    date_formats = ['%m/%d/%Y', '%m-%d-%Y', '%-m/%-d/%Y', '%-m-%-d-%Y']
                    for fmt in date_formats:
                        formatted_date = date_var.strftime(fmt)
                        if formatted_date in str(note):
                            new_note = str(note).replace(formatted_date, '*' * len(formatted_date))
                            return new_note
            else:
                if True:
                    if str(row[identifer]) in nulls:
                        return str(note)
                    if str(row[identifer]).lower() in notes_list_lower[row_ind - 1]:
                        start_index = notes_list_lower[row_ind - 1].index(str(row[identifer]).lower()) # Index from where the identifier starts
                        substr_to_replace = notes_list[row_ind - 1][start_index : start_index + len(str(row[identifer]).lower())]
                        new_note = notes_list[row_ind - 1].replace(substr_to_replace, '*' * len(substr_to_replace))
                        return new_note
            return str(note)
    
    
    print("GETTING NEW NOTES(TAKING TIME)")
    new_notes = df.apply(lambda row : get_row_relevancy_data(row, identifier, identifier_type), axis=1)
    
    print("SUBSTITUTING INDIVIDUAL NOTES OBTAINED TO THE DATAFRAME")
    
    df["note_text\n"] = new_notes
    
    return df

if __name__ == "__main__":
    # Change for your cutom usecase 
    relevant_identifers = {'dates' : ['death_date', ['year_of_birth', 'month_of_birth', 'day_of_birth']],
                     'date_times' : ['note_datetime', 'death_datetime'],
                     'texts' : ["note_type", "note_class", "gender", "race", "ethnicity", 'address_1', 'address_2', 'city', 'state'],
                     'numbers' : ["person_id", "note_id", "empi_id", "mrn", "zip"]
    }
    # Path of the dataset file, that needs to be masked
    input_filepath = sys.argv[1]
    print("input_filepath : ", input_filepath)
    df = pd.read_csv(input_filepath)
    for identifier_type in relevant_identifers.keys():
        for identifier in relevant_identifers[identifier_type]:
            df_copy = pd.DataFrame(df)
            df_with_identifier_masked = mask_identifier_in_the_data(df_copy, identifier, identifier_type)
            output_filepath_csv = input_filepath.split(".csv")[-2] + f"_{identifier if type(identifier) != list else '_'.join(identifier)}_masked.csv"
            output_filepath_parquet = input_filepath.split(".csv")[-2] + f"_{identifier if type(identifier) != list else '_'.join(identifier)}_masked.parquet"
            df_with_identifier_masked.to_csv(output_filepath_csv, index=False)
            df_with_identifier_masked.to_parquet(output_filepath_parquet, index=False)
