from typing import Dict, List
import textattack

from utils import get_profile_embeddings


class RobertaRobertaReidMetric(textattack.metrics.Metric):
    model_key: str
    num_examples_offset: int
    print_identified_results: bool
    def __init__(self, num_examples_offset: int, print_identified_results: bool = True):
        self.model_key = "model_3_3"
        self.num_examples_offset = num_examples_offset
        # TODO: enhance this class to support shuffled indices from the attack.
        #   Right now this assumes things are sequential.
    
    def _document_from_attack_result(self, result: textattack.attack_results.AttackResult):
        document = result.perturbed_result.attacked_text.text
        return document.replace("[MASK]", "<mask>")
    
    def calculate(self, attack_results: List[textattack.attack_results.AttackResult]) -> Dict[str, float]:
        # TODO move to logging statement
        print("Computing reidentification score...")
        # get profile embeddings
        all_profile_embeddings = get_profile_embeddings(model_key=self.model_key, use_train_profiles=True)

        # initialize model
        model = CoordinateAscentModel.load_from_checkpoint(
            model_paths_dict[self.model_key]
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
        model_wrapper = MyModelWrapper(
            model=model,
            document_tokenizer=tokenizer,
            max_seq_length=128,
            profile_embeddings=all_profile_embeddings,
        ).to('cuda')

        # get documents
        documents = list(
            map(self._document_from_attack_result, attack_results)
        )

        # check accuracy
        predictions = model_wrapper(documents)
        true_labels = (
            torch.arange(len(attack_results)) + self.num_examples_offset
        ).cuda()
        correct_preds = (predictions.argmax(dim=1) == true_labels)
        accuracy = correct_preds.float().mean()

        for i, pred in enumerate(correct_preds.tolist()):
            if pred:
                print(f'Identified example {i}:', attack_results[i])

        model_wrapper.model.cpu()
        del model_wrapper
        return f'{accuracy.item()*100.0:.2f}'