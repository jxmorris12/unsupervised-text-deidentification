from typing import List

import csv
import pickle

import pandas as pd
import textattack
from textattack.loggers import CSVLogger
from textattack.shared import AttackedText


class CustomCSVLogger(CSVLogger):
    """Logs attack results to a CSV."""
    def __init__(self, filename="results.csv", color_method="file"):
        textattack.shared.logger.info(f"Logging to CSV at path {filename}")
        assert ".csv" in filename
        self.filename = filename
        self.pickle_filename = filename.replace(".csv", "_examples.p")
        self.color_method = color_method
        self.row_list = []
        self.example_strings_list = [] # for each example, *all the strings* 
        self._flushed = True
    
    def _get_example_strings(self, pt: AttackedText) -> List[str]:
        """The list of all texts that an AttackedText has been. Used to get the text
        at all levels of masking, from no masks to all masks.
        """
        strings = [pt.text]
        while 'prev_attacked_text' in pt.attack_attrs:
            strings.append(pt.attack_attrs['prev_attacked_text'].text)
            pt = pt.attack_attrs['prev_attacked_text']
        return strings[::-1]


    def log_attack_result(self, result: textattack.goal_function_results.ClassificationGoalFunctionResult):
        original_text, perturbed_text = result.diff_color(self.color_method)
        original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
        perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
        result_type = result.__class__.__name__.replace("AttackResult", "")
        row = {
            "original_person": result.original_result._processed_output[0],
            "original_text": original_text,
            "perturbed_person": result.perturbed_result._processed_output[0],
            "perturbed_text": perturbed_text,
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "original_output": result.original_result.output,
            "perturbed_output": result.perturbed_result.output,
            "ground_truth_output": result.original_result.ground_truth_output,
            "num_queries": result.num_queries,
            "result_type": result_type,
        }
        self.row_list.append(row)
        self.example_strings_list.append(self._get_example_strings(result.perturbed_result.attacked_text))
        self._flushed = False

    def flush(self):
        self.df = pd.DataFrame.from_records(self.row_list)
        self.df.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
        pickle.dump(self.example_strings_list, open(self.pickle_filename, 'wb'))
        textattack.shared.logger.info(f"Wrote examples to file at {self.pickle_filename}")
        self._flushed = True
