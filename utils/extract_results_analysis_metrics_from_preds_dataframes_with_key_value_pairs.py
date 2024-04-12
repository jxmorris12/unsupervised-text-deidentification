import pandas as pd
import ast
import sys
from collections import Counter
import os

def for_key_without_value(correct_preds_with_found_key_value_pairs_path, incorrect_preds_with_found_key_value_pairs_path, result_analysis_folder_path):
    correct_preds_with_found_key_value_pairs = pd.read_csv(correct_preds_with_found_key_value_pairs_path)
    correct_preds_with_found_key_value_pairs_dicts = list(map(ast.literal_eval, correct_preds_with_found_key_value_pairs["without_nan_found_key_value_pairs"].to_list()))
    correct_preds_with_found_key_value_pairs_keys = []
    for dic in correct_preds_with_found_key_value_pairs_dicts:                                                                                                                                         
        for key in dic.keys():                                                                                                                                
            correct_preds_with_found_key_value_pairs_keys.append(key)
    
    
    incorrect_preds_with_found_key_value_pairs = pd.read_csv(incorrect_preds_with_found_key_value_pairs_path)
    incorrect_preds_with_found_key_value_pairs_dicts = list(map(ast.literal_eval, incorrect_preds_with_found_key_value_pairs["without_nan_found_key_value_pairs"].to_list()))
    incorrect_preds_with_found_key_value_pairs_keys = [] 
    for dic in incorrect_preds_with_found_key_value_pairs_dicts:
        for key in dic.keys():                                                                                                                                
            incorrect_preds_with_found_key_value_pairs_keys.append(key)

    final_df = pd.DataFrame(set(correct_preds_with_found_key_value_pairs_keys).union(set(incorrect_preds_with_found_key_value_pairs_keys)), columns=["keys"])
    final_df["correct_count"] = [Counter(correct_preds_with_found_key_value_pairs_keys)[key] for key in final_df["keys"]]
    final_df["correct_count_percentage"] = final_df["correct_count"] / final_df["correct_count"].sum() * 100
    final_df["incorrect_count"] = [Counter(incorrect_preds_with_found_key_value_pairs_keys)[key] for key in final_df["keys"]]
    final_df["incorrect_count_percentage"] = final_df["incorrect_count"] / final_df["incorrect_count"].sum() * 100
    final_df["ratio_correct_count_to_incorrect_count"] = final_df["correct_count"] / final_df["incorrect_count"]
    final_df = final_df.sort_values('ratio_correct_count_to_incorrect_count', ascending=False)
    final_df.to_csv(os.path.join(result_analysis_folder_path, "total_without_nan_key_distribution.csv"), index=False)

def for_key_with_value(correct_preds_with_found_key_value_pairs_path, incorrect_preds_with_found_key_value_pairs_path, result_analysis_folder_path):
    correct_preds_with_found_key_value_pairs = pd.read_csv(correct_preds_with_found_key_value_pairs_path)
    correct_preds_with_found_key_value_pairs_dicts = list(map(ast.literal_eval, correct_preds_with_found_key_value_pairs["without_nan_found_key_value_pairs"].to_list()))
    correct_preds_with_found_key_value_pairs_counter = Counter()
    for dic in correct_preds_with_found_key_value_pairs_dicts:                                                                                                                                   
        correct_preds_with_found_key_value_pairs_counter.update(dic.items())
    
    incorrect_preds_with_found_key_value_pairs = pd.read_csv(incorrect_preds_with_found_key_value_pairs_path)
    incorrect_preds_with_found_key_value_pairs_dicts = list(map(ast.literal_eval, incorrect_preds_with_found_key_value_pairs["without_nan_found_key_value_pairs"].to_list()))
    incorrect_preds_with_found_key_value_pairs_counter = Counter()
    for dic in incorrect_preds_with_found_key_value_pairs_dicts:                                                                                                                                   
        incorrect_preds_with_found_key_value_pairs_counter.update(dic.items())

    final_df = pd.DataFrame(set(correct_preds_with_found_key_value_pairs_counter).union(set(incorrect_preds_with_found_key_value_pairs_counter)), columns=["keys", "values"])
    final_df["correct_count"] = 0 
    for ind in range(len(final_df)):                                                                                                                                                                   
        final_df.loc[ind, "correct_count"] = correct_preds_with_found_key_value_pairs_counter[(final_df.iloc[ind]["keys"], final_df.iloc[ind]["values"])]
    final_df["correct_count_percentage"] = final_df["correct_count"] / final_df["correct_count"].sum() * 100


    final_df["incorrect_count"] = 0 
    for ind in range(len(final_df)):                                                                                                                                                                   
        final_df.loc[ind, "incorrect_count"] = incorrect_preds_with_found_key_value_pairs_counter[(final_df.iloc[ind]["keys"], final_df.iloc[ind]["values"])]
    final_df["incorrect_count_percentage"] = final_df["incorrect_count"] / final_df["incorrect_count"].sum() * 100
    final_df["ratio_correct_count_to_incorrect_count"] = final_df["correct_count"] / final_df["incorrect_count"]  
    final_df = final_df.sort_values('ratio_correct_count_to_incorrect_count', ascending=False)
    final_df.to_csv(os.path.join(result_analysis_folder_path, "total_without_nan_key_with_value_distribution.csv"), index=False)



def main():

    correct_preds_with_found_key_value_pairs_path =  sys.argv[1]
    incorrect_preds_with_found_key_value_pairs_path =  sys.argv[2]
    result_analysis_folder_path = os.path.join("/".join(correct_preds_with_found_key_value_pairs_path.split("/")[:-1]), "results_analysis")
    if not os.path.exists(result_analysis_folder_path):
        os.mkdir(result_analysis_folder_path)
    
    for_key_without_value(correct_preds_with_found_key_value_pairs_path, incorrect_preds_with_found_key_value_pairs_path, result_analysis_folder_path)
    for_key_with_value(correct_preds_with_found_key_value_pairs_path, incorrect_preds_with_found_key_value_pairs_path, result_analysis_folder_path)

if __name__ == "__main__":
    main()
