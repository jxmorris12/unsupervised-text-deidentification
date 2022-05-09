from huggingface_hub import Repository

def main():
    # make sure model file and anything else you need is in local_dir
    repo = Repository(
        local_dir = './wikibio_vanilla', 
        clone_from='jxm/wikibio_vanilla'
    )
    repo.push_to_hub(
        commit_message = "Wikibio document-profile matching model"
    )
    
    
if __name__ == '__main__':
    main()