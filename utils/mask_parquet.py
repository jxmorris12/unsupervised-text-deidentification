import pandas as pd
from pathlib import Path  
import datasets
import os


def main():
    parquet_data_path = Path("../../full_csv_most_relevant_note_per_person.parquet")
    masked_data_parent_path = Path("/prj0124_gpu/sln4001/philter/philter-ucsf/philter_results_new")     
    p_data = datasets.Dataset.from_parquet(str(parquet_data_path))
    
    p_data_df = pd.DataFrame(p_data)
    
    def processRow(row):                                                                                                                                                                               
              print("processing row num : ", row.name)
              rowIndex = row.name                                                                                                                                                                           
              f = open(os.path.join(str(masked_data_parent_path), "output_" + str(rowIndex) + ".txt"))                                                                                                           
              row["note_text\n"] = f.read().strip('"')                                                                                                                                                      
              f.close()  
              return row
    
    # 18000 as only that many rows are processed by philter till now
    index_till = 18000
    p_data_df = p_data_df[:index_till].apply(lambda x : processRow(x), axis=1) 
    
    curr_file_name, curr_file_ext = os.path.splitext(os.path.basename(str(parquet_data_path)))
    new_file_name = curr_file_name + "_masked_till_" + str(index_till)

    p_data_df.to_parquet(os.path.join(parquet_data_path.parent.absolute(), new_file_name + curr_file_ext))


if __name__ == "__main__":
    main()
