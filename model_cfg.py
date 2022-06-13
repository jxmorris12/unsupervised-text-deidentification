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

# model_7 is roberta-tapas with word dropout (no prof dropout) UNIFORM SAMPLING MASKING RATE for
# about two days. trained here.   wandb.ai/jack-morris/deid-wikibio-2/runs/26w4n18i
model_paths_dict["model_7"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__dropout_-1.0_1.0_0.0/deid-wikibio-2_default/26w4n18i_461/checkpoints/epoch=8-step=18974.ckpt'

# model_8 is roberta-tapas-idf with uniform sampling and linear lr decay 
#       and label smoothing {0.1, 0.01, 0.05}.
#       wandb.ai/jack-morris/deid-wikibio-3/runs/{1jjn2o39,3u0271pj,3f738g9c}
model_paths_dict["model_8_ls0.01"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__idf__dropout_-1.0_1.0_0.0__e3072__ls0.01/deid-wikibio-3_default/3u0271pj_704/checkpoints/epoch=69-step=159389-adv100_acc.ckpt'
model_paths_dict["model_8_ls0.05"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__idf__dropout_-1.0_1.0_0.0__e3072__ls0.05/deid-wikibio-3_default/3f738g9c_700/checkpoints/epoch=66-step=152558-adv100_acc.ckpt'
model_paths_dict["model_8_ls0.1"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__idf__dropout_-1.0_1.0_0.0__e3072__ls0.1/deid-wikibio-3_default/1jjn2o39_703/checkpoints/epoch=67-step=154835-adv100_acc.ckpt'


# model_9 is pmlm-tapas-idf with uniform sampling and linear lr decay
#       and label smoothing {0.1, 0.01, 0.05}.
#       wandb.ai/jack-morris/deid-wikibio-3/runs/{1vfvk4uq,3fd1ha2g,1jpaswow}
model_paths_dict["model_9_ls0.01"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__pmlm-a__tapas__idf__dropout_-1.0_1.0_0.0__e3072__ls0.01/deid-wikibio-3_default/1jpaswow_695/checkpoints/epoch=93-step=214037-adv100_acc.ckpt'
model_paths_dict["model_9_ls0.05"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__pmlm-a__tapas__idf__dropout_-1.0_1.0_0.0__e3072__ls0.05/deid-wikibio-3_default/1vfvk4uq_702/checkpoints/epoch=79-step=182159-adv100_acc.ckpt'
model_paths_dict["model_9_ls0.1"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__pmlm-a__tapas__idf__dropout_-1.0_1.0_0.0__e3072__ls0.1/deid-wikibio-3_default/3fd1ha2g_699/checkpoints/epoch=96-step=220868-adv100_loss.ckpt'