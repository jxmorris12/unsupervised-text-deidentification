import glob

model_paths_dict = {
    # model_1 (deprecated)

    # model_2 (deprecated) was roberta-distilbert, from here:
    #   wandb.ai/jack-morris/deid-wikibio-2/runs/xjybn01j/logs?workspace=user-jxmorris12

    # model_3 is roberta-tapas trained with adv masking m=8, from here:
    #   wandb.ai/jack-morris/deid-wikibio-2/runs/2kyfzwx7/logs?workspace=user-jxmorris12
    "model_3": "/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__adv/deid-wikibio-2_default/2kyfzwx7_433/checkpoints/epoch=21-step=100165.ckpt",
}

 # model_4 is trained with word dropout/prof dropout 0.5/0.5/0.5 for about two days,
#   while I manually lowered the learning rate.
#   wandb.ai/jack-morris/deid-wikibio-2/runs/28ierlgd/logs?workspace=user-jxmorris12
model_paths_dict["model_4"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__dropout_0.5_0.5_0.5/deid-wikibio-2_default/28ierlgd_437/checkpoints/epoch=21-step=100165.ckpt'


# model_5 is roberta-tapas trained with adv masking m=16, from here:
#   wandb.ai/jack-morris/deid-wikibio-2/runs/236desyb?workspace=user-jxmorris12
model_paths_dict["model_5"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__adv/deid-wikibio-2_default/236desyb_444/checkpoints/epoch=22-step=104718.ckpt'

# model_6 is roberta-tapas vanilla (no masking during training). training stopped early
# as performance on masked data started to get worse.
#   wandb.ai/jack-morris/deid-wikibio-2/runs/3ge40631
model_paths_dict["model_6"] = "/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas/deid-wikibio-2_default/3ge40631_445/checkpoints/epoch=3-step=18211.ckpt"