from datamodule import WikipediaDataModule
from model import CoordinateAscentModel
import os


num_cpus = len(os.sched_getaffinity(0))

checkpoint_path = "models_weights/wikibio_roberta_roberta/model.chkpt"
model = CoordinateAscentModel.load_from_checkpoint(checkpoint_path)
dm = WikipediaDataModule(
    document_model_name_or_path=model.document_model_name_or_path,
    profile_model_name_or_path=model.profile_model_name_or_path,
    dataset_name='wiki_bio',
    dataset_train_split='train[:10%]',
    dataset_val_split='val[:20%]',
    dataset_version='1.2.0',
    num_workers=1,
    train_batch_size=64,
    eval_batch_size=64,
    max_seq_length=128,
    sample_spans=False,
)
