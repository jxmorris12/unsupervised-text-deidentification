import argparse
import os
import shutil

from huggingface_hub import Repository


def main(args: argparse.ArgumentParser):
    assert os.path.exists(args.checkpoint_name)
    assert args.checkpoint_name.endswith(".ckpt"), "is the source file a model checkpoint? if so, just add ckpt extension"
    assert not os.path.exists(args.tmp_model_save_folder), "error, folder already exists"
    os.makedirs(args.tmp_model_save_folder, exist_ok=False)
    model_tmp_path = os.path.join(args.tmp_model_save_folder, 'model.ckpt')
    repo = Repository(
        local_dir = args.tmp_model_save_folder, 
        clone_from = args.model_hub_name,
    )
    shutil.copyfile(args.checkpoint_name, model_tmp_path)
    repo.push_to_hub(
        commit_message = args.commit_message
    ) 
    shutil.rmtree(args.tmp_model_save_folder)
    print('uploaded model and deleted tmp folder')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_name', type=str,
                        help='path to model checkpoint, like x/y/last.ckpt')
    parser.add_argument('model_hub_name', type=str,
                        help='model hub model name including group/user. ex: jxm/wikibio_vanilla')
    parser.add_argument('--tmp_model_save_folder', type=str, default='wikibio_vanilla',
                        help='temporary folder used to save model to during upload')
    parser.add_argument('--commit_message', type=str, default='Wikibio document-profile matching model',
                        help='commit message for pushing to model hub')
    args = parser.parse_args()
    main(args)