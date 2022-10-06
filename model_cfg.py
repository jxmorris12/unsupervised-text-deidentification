import glob

model_paths_dict = {}

#
#    bi-encoders
#

# model_1 (deprecated)

# model_2 (deprecated) was roberta-distilbert, from here:
#   wandb.ai/jack-morris/deid-wikibio-2/runs/xjybn01j/logs?workspace=user-jxmorris12

# model_3 is roberta-tapas trained with adv masking m=8, from here:
#   wandb.ai/jack-morris/deid-wikibio-2/runs/2kyfzwx7/logs?workspace=user-jxmorris12
model_paths_dict["model_3"] = "/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__adv/deid-wikibio-2_default/2kyfzwx7_433/checkpoints/epoch=21-step=100165.ckpt"

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

# model_2_1 is roberta-tapas, no masking
#   wandb.ai/jack-morris/deid-wikibio-4/runs/1tu650oe
model_paths_dict["model_2_1"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__e3072__ls0.1/deid-wikibio-4_default/1tu650oe_747/checkpoints/epoch=63-step=145727-idf_total.ckpt'

# model_2_2 is pmlm-a-tapas, uniformly sampled idf masking
#   wandb.ai/jack-morris/deid-wikibio-4/runs/3fyovpn7
model_paths_dict["model_2_2"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__pmlm-a__tapas__idf__dropout_-1.0_1.0_0.0__e3072__ls0.1/deid-wikibio-4_default/3fyovpn7_753/checkpoints/epoch=30-step=70586-idf_total.ckpt'

# model_2_3 is roberta-tapas, uniformly sampled idf masking
#   wandb.ai/jack-morris/deid-wikibio-4/runs/6soyixf9
model_paths_dict["model_2_3"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__idf__dropout_-1.0_1.0_0.0__e3072__ls0.1/deid-wikibio-4_default/6soyixf9_752/checkpoints/epoch=30-step=70586-idf_total.ckpt'

# model_2_4 is roberta-tapas, 0.5 sampled random masking
#   wandb.ai/jack-morris/deid-wikibio-4/runs/20zw3yj3
model_paths_dict["model_2_4"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__dropout_0.5_1.0_0.0__e3072__ls0.1/deid-wikibio-4_default/20zw3yj3_748/checkpoints/epoch=63-step=145727-idf_total.ckpt'

# model_2_5 is roberta-roberta, 0.5 sampled random masking
#   wandb.ai/jack-morris/deid-wikibio-4/runs/1c9464tp
model_paths_dict["model_2_5"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__dropout_0.5_1.0_0.0__e3072__ls0.1/deid-wikibio-4_default/1c9464tp_750/checkpoints/epoch=58-step=134342-idf_total.ckpt'

# model_2_6 is roberta-roberta, 0.5 sampled idf masking
#   wandb.ai/jack-morris/deid-wikibio-4/runs/1c9464tp
model_paths_dict["model_2_6"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__idf__dropout_0.5_1.0_0.0__e3072__ls0.1/deid-wikibio-4_default/f18dh3hl_751/checkpoints/epoch=30-step=70586-idf_total.ckpt'


# model_3_1 is roberta-tapas, no masking
#   wandb.ai/jack-morris/deid-wikibio-4/runs/1tu650oe
model_paths_dict["model_3_1"] = model_paths_dict["model_2_1"]

# model_3_2 is roberta-tapas, uniformly sampled random masking
#   wandb.ai/jack-morris/deid-wikibio-4/runs/ojgxa1tf?workspace=user-jxmorris12
model_paths_dict["model_3_2"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__dropout_-1.0_1.0_0.0__e3072__ls0.1/deid-wikibio-4_lightning_logs/ojgxa1tf_6/checkpoints/last.ckpt'
model_paths_dict["model_3_2__idf"] = model_paths_dict["model_8_ls0.01"]

# model_3_3 is roberta-roberta, uniformly sampled random masking
model_paths_dict["model_3_3"] = "/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__dropout_-1.0_1.0_0.0__e3072__ls0.1/deid-wikibio-4_lightning_logs/2cr1gp87_28/checkpoints/epoch=68-step=157113.ckpt"
# model_paths_dict["model_3_3"] = '??'
model_paths_dict["model_3_3__placeholder"] = model_paths_dict["model_2_5"]
#  wandb.ai/jack-morris/deid-wikibio-4/runs/f18dh3hl/logs?workspace=
model_paths_dict["model_3_3__idf"] = "/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__idf__dropout_0.5_1.0_0.0__e3072__ls0.1/deid-wikibio-4_default/f18dh3hl_751/checkpoints/last.ckpt"

# model_3_4 is pmlm-a-tapas, uniformly sampled random masknig
#   wandb.ai/jack-morris/deid-wikibio-4/runs/1g8o1iw3
model_paths_dict["model_3_4"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__pmlm-a__tapas__dropout_-1.0_1.0_0.0__e3072__ls0.1/deid-wikibio-4_default/1g8o1iw3_749/checkpoints/last.ckpt'
model_paths_dict["model_3_4__idf"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__pmlm-a__tapas__idf__dropout_-1.0_1.0_0.0__e3072__ls0.1/deid-wikibio-4_default/3fyovpn7_753/checkpoints/epoch=60-step=138896-idf_total.ckpt'

# model_3_5 is pmlm-a-tapas, uniformly sampled random masking,
#   **trained on lexically redacted data**
model_paths_dict["model_3_5__epoch47"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__pmlm-a__tapas__dropout_-1.0_1.0_0.0__e3072__ls0.1/deid-wikibio-4_lightning_logs/2lslhb53_41/checkpoints/epoch=47-step=27360.ckpt'


#
# cross-encoders
#
# trained on 10% of data for 16 epochs
model_paths_dict["cross_encoder_10%"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/cca__roberta__n_7__dropout_-1.0_1.0_0.0__e3072__ls0.01/deid-wikibio-5-cross-encoder_default/vyrd6owo_804/checkpoints/last.ckpt'
# trained on 100% of data for 8 epochs (harder neighbors)
model_paths_dict["cross_encoder"] = '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/cca__roberta__n_7__dropout_-1.0_1.0_0.0__e3072__ls0.01/deid-wikibio-5-cross-encoder_lightning_logs/185h9j9n_72/checkpoints/epoch=8-step=163881.ckpt'
