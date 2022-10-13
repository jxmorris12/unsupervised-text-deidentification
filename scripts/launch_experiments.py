from deidentify import main as deidentify_main

class ExperimentArgs:
    n = 1000
    num_examples_offset = 0
    model = "model_3_4"
    beam_width = 1
    min_percent_words = 1.0
    table_score = 100.0
    out_folder_path = "experiments"
    ###########################################
    # Everything below here is an experimental
    # variable that we didn't end up using.
    ###########################################
    k = None
    eps = None
    do_reid = False
    use_train_profiles = False
    ignore_stopwords = False
    use_type_swap = True
    max_idf_goal = None
    fuzzy_ratio = 1.0


def launch_experiment(**kwargs):
    args = ExperimentArgs()
    print(f'Launching experiment with kwargs: {kwargs}')
    for k,v in kwargs.items():
        setattr(args, k, v)

    ###########################################

    deidentify_main(
        k=args.k,
        eps=args.eps,
        min_percent_words=args.min_percent_words,
        n=args.n,
        num_examples_offset=args.num_examples_offset,
        beam_width=args.beam_width,
        model_key=args.model,
        use_type_swap=args.use_type_swap,
        use_train_profiles=args.use_train_profiles,
        out_folder_path=args.out_folder_path,
        out_file_path=args.out_file_path,
        ignore_stopwords=args.ignore_stopwords,
        no_model=args.no_model,
        fuzzy_ratio=args.fuzzy_ratio,
        max_idf_goal=args.max_idf_goal,
        table_score=args.table_score,
        do_reid=args.do_reid,
    )


def main():
    ##################################################################
    # NN DeID (Biencoder)
    launch_experiment(max_idf_goal=None, model="model_3_4", table_score=100.0, no_model=False, out_file_path='nn_deid_biencoder_table')

    # NN DeID (Cross-encoder)
    launch_experiment(max_idf_goal=None, model="cross_encoder", table_score=100.0, no_model=False, out_file_path='nn_deid_crossencoder_table')
    launch_experiment(max_idf_goal=None, model="cross_encoder_10", table_score=100.0, no_model=False, out_file_path='nn_deid_crossencoder10_table')

    # IDF
    launch_experiment(max_idf_goal=1e-10, table_score=100.0, no_model=True, out_file_path='idf_table')

    ##################################################################
    # NN DeID (Biencoder)
    launch_experiment(max_idf_goal=None, model="model_3_4", table_score=0.0, no_model=False, out_file_path='nn_deid_biencoder')

    # NN DeID (Cross-encoder)
    launch_experiment(max_idf_goal=None, model="cross_encoder", table_score=0.0, no_model=False, out_file_path='nn_deid_crossencoder')
    launch_experiment(max_idf_goal=None, model="cross_encoder_10", table_score=0.0, no_model=False, out_file_path='nn_deid_crossencoder10')

    # IDF
    launch_experiment(max_idf_goal=1e-10, table_score=0.0, no_model=True, out_file_path='idf')

if __name__ == '__main__': main()