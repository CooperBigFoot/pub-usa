  GPU 1 - EALSTM folds 1-6:
  tl-train experiments/small_models_275k_params_12fold.yaml --models
  ealstm_275k_fold_1,ealstm_275k_fold_2,ealstm_275k_fold_3,ealstm_275k_fold_4,ealstm_275k_fold_5,ealstm_275k_fold_6

  GPU 2 - EALSTM folds 7-12:
  tl-train experiments/small_models_275k_params_12fold.yaml --models
  ealstm_275k_fold_7,ealstm_275k_fold_8,ealstm_275k_fold_9,ealstm_275k_fold_10,ealstm_275k_fold_11,ealstm_275k_fold_12

  GPU 3 - Mamba folds 1-6:
  tl-train experiments/small_models_275k_params_12fold.yaml --models
  mamba_275k_fold_1,mamba_275k_fold_2,mamba_275k_fold_3,mamba_275k_fold_4,mamba_275k_fold_5,mamba_275k_fold_6

  GPU 4 - Mamba folds 7-12:
  tl-train experiments/small_models_275k_params_12fold.yaml --models
  mamba_275k_fold_7,mamba_275k_fold_8,mamba_275k_fold_9,mamba_275k_fold_10,mamba_275k_fold_11,mamba_275k_fold_12
