from typing import Dict, Optional

import textattack
import torch
import transformers

from model import AbstractModel, ContrastiveCrossAttentionModel, CoordinateAscentModel

class MainModelWrapper(textattack.models.wrappers.ModelWrapper):
    model: CoordinateAscentModel
    most_recent_datapoint: Optional[Dict]
    most_recent_datapoint_idx: Optional[int]
    document_tokenizer: transformers.AutoTokenizer
    profile_embeddings: torch.Tensor
    max_seq_length: int
    fake_response: bool
    
    def __init__(self,
            model: CoordinateAscentModel,
            document_tokenizer: transformers.AutoTokenizer,
            profile_embeddings: torch.Tensor,
            max_seq_length: int = 128,
            fake_response: bool = False
        ):
        self.model = model
        self.model.eval()
        self.document_tokenizer = document_tokenizer
        self.profile_embeddings = profile_embeddings.clone().detach()
        self.max_seq_length = max_seq_length
        self.fake_response = fake_response
        self.most_recent_datapoint = None
        self.most_recent_datapoint_idx = None
                 
    def to(self, device):
        self.model.to(device)
        self.profile_embeddings = self.profile_embeddings.to(device)
        return self # so semantics `model = MyModelWrapper().to('cuda')` works properly

    def __call__(self, text_input_list):
        if self.fake_response:
            return torch.ones((len(text_input_list), 72_831 * 2))
        model_device = next(self.model.parameters()).device

        tokenized_documents = self.document_tokenizer.batch_encode_plus(
            text_input_list,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        batch = {f"document__{k}": v for k,v in tokenized_documents.items()}

        with torch.no_grad():
            document_embeddings = self.model.forward_document(batch=batch, document_type='document')
            document_to_profile_logits = document_embeddings @ (self.profile_embeddings.T)
        assert document_to_profile_logits.shape == (len(text_input_list), len(self.profile_embeddings))
        return document_to_profile_logits
    

class CrossEncoderModelWrapper(textattack.models.wrappers.ModelWrapper):
    model: ContrastiveCrossAttentionModel
    document_tokenizer: transformers.AutoTokenizer
    most_recent_datapoint: Optional[Dict]
    most_recent_datapoint_idx: Optional[int]
    max_seq_length: int
    fake_response: bool
    
    def __init__(self,
            model: ContrastiveCrossAttentionModel,
            document_tokenizer: transformers.AutoTokenizer,
            max_seq_length: int = 128,
            fake_response: bool = False
        ):
        self.model = model
        self.model.eval()
        self.document_tokenizer = document_tokenizer
        self.max_seq_length = max_seq_length
        self.fake_response = fake_response
        self.most_recent_datapoint = None
        self.most_recent_datapoint_idx = None
                 
    def to(self, device):
        self.model.to(device)
        return self # so semantics `model = CrossEncoderModelWrapper().to('cuda')` works properly

    def __call__(self, text_input_list):
        num_profiles = 72_831 * 2
        if self.fake_response:
            return torch.ones((len(text_input_list)))
        
        assert self.most_recent_datapoint is not None, "need to set self.most_recent_datapoint so we have a profile"
        model_device = next(self.model.parameters()).device
        batch_size = len(text_input_list)

        tokenized_documents = self.document_tokenizer.batch_encode_plus(
            text_input_list,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        batch = {f"document__{k}": v for k,v in tokenized_documents.items()}

        input_ids = torch.cat(
            (
                tokenized_documents['input_ids'], 
                torch.tensor(self.most_recent_datapoint['profile__input_ids'])[None].repeat((batch_size,1))
            ), dim=1
        ).to(model_device)
        attention_mask = torch.cat(
            (
                tokenized_documents['attention_mask'], 
                torch.tensor(self.most_recent_datapoint['profile__attention_mask'])[None].repeat((batch_size,1))
            ), dim=1
        ).to(model_device)
        all_inputs = {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
        }
        with torch.no_grad():
            document_scores = self.model.forward_document_and_profile_inputs(inputs=all_inputs).squeeze(dim=1)
        assert document_scores.shape == (batch_size,)
        document_to_profile_logits = torch.where(
            (torch.arange(num_profiles,) == self.most_recent_datapoint_idx)[None].to(model_device),
            document_scores[:, None], torch.tensor(-(10**5), dtype=torch.float32).to(model_device),
        )
        return document_to_profile_logits