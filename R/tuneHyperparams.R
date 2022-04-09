## NTD:
# 1) take out all "run_"
# 2) make sigma, hs, bss, lrs, dim_zs, niws, n_epochs customizable? (kinda is, but create defaults and take out hard coding --> put in Paper repo)

#' Hyperparameter tuning used by NIMIWAE() function
#'
#' @param FUN run_<> function for other methods (not NIMIWAE)
#' @param method String specifying the method to tune hyperparameters. Can be "NIMIWAE" (default), "MIWAE", "VAEAC", "HIVAE", "MEAN", or "MF". Hyperparameters are not tuned for "MEAN" or "MF"
#' @param data Data matrix (N x P)
#' @param data_types vector of length=ncol(data). Valid values: "real", "count", "cat" or "pos"
#' @param Missing Missingness mask matrix (N x P)
#' @param g Training-validation-test split partitioning
#' @param rdeponz TRUE/FALSE: Whether to allow missingness (r) to depend on the latent variable (z). Default is FALSE
#' @param learn_r TRUE/FALSE: Whether to learn missingness model via appended NN (TRUE, default), or fit a known logistic regression model (FALSE). If FALSE, `phi0` and `phi` must be specified
#' @param phi0 (optional) Intercept of logistic regression model, if learn_r = FALSE.
#' @param phi (optional) Vector of coefficients of logistic regression model for each input covariates `covars_r`, if learn_r = FALSE. `phi` must be the same length as the number of input covariates, or `sum(covars_r)`.
#' @param Cs (optional) # factors for categorical variables. Must be of length = # of categorical variables.
#' @param ignorable TRUE/FALSE: Whether missingness is ignorable (MCAR/MAR) or nonignorable (MNAR, default). If missingness is known to be ignorable, "ignorable=T" omits missingness model.
#' @param covars_r Vector of 1's and 0's of whether each feature is included as covariates in the missingness model. Need not be specified if `ignorable = T`. Default is using all features as covariates in missingness model. Must be length P (or `ncol(data)`)
#' @param arch Architecture of NIMIWAE. Can be "IWAE" or "VAE". "VAE" is specific case of the "IWAE" where only one sample is drawn from the joint posterior of (z, xm).
#' @param sigma activation function ("relu" or "elu")
#' @param h integer, number of nodes per hidden layer
#' @param n_hidden_layers integer, #hidden layers (except missingness model Decoder_r)
#' @param n_hidden_layers_r integer, #hidden layers for Decoder_r (default: 0)
#' @param bs integer, batch size (training)
#' @param lr float, learning rate
#' @param dim_z integer, dimensionality of latent z. Default: 1/4 and 1/2 of the columns of data
#' @param niw integer, number of importance weights (samples drawn from each latent space). Default: 5
#' @param n_epoch integer, maximum number of epochs (without early stop). Default: 2002
#' @param data_types_HIVAE Specify for HIVAE only.
#' @param one_hot_max_sizes Specify for VAEAC only.
#' @param ohms Specify for VAEAC only.
#' @param MissingDatas Specify for VAEAC only.
#' @return res object: method's fit on test set, after training on training set and validating best set of hyperparameter values using the validation set.
#'
#' @author David K. Lim, \email{deelim@live.unc.edu}
#' @references \url{https://github.com/DavidKLim/NIMIWAE}
#'
#' @importFrom reticulate source_python import
#' @importFrom mice mice complete
#'
#' @export
tuneHyperparams = function(FUN=NULL,method="NIMIWAE",dataset="",data,data_types,data_types_0, Missing, g,
                           rdeponz=F, learn_r=T, phi0=NULL, phi=NULL, Cs, ignorable=F, covars_r=rep(1,ncol(data)),
                           arch="IWAE", draw_xmiss=T,  # for NIMIWAE: whether each NN is optimized separately, architecture: VAE or IWAE
                           sigma="elu", h=c(128L,64L), h_r=c(128L,64L), n_hidden_layers=c(1L,2L), n_hidden_layers_r0=NULL, bs=1000L, lr=c(0.001,0.01),
                           dim_z=as.integer(c(floor(ncol(data)/2),floor(ncol(data)/4))), niws=5L, n_imputations=5L, n_epochs=2002L,
                           data_types_HIVAE=NULL, one_hot_max_sizes=NULL, ohms=NULL,
                           MissingDatas = NULL, save_imps=F, dir_name=".",normalize=T, early_stop = T
){
  h = as.integer(h); h_r = as.integer(h_r); n_hidden_layers = as.integer(n_hidden_layers)
  if(!is.null(n_hidden_layers_r0)[1]){n_hidden_layers_r0 = as.integer(n_hidden_layers_r0)}
  bs = as.integer(bs); niws = as.integer(niws); n_imputations = as.integer(n_imputations); n_epochs = as.integer(n_epochs)
  if(any(dim_z<=0)){dim_z[dim_z<=0]=1L}
  if(all(dim_z==dim_z[1])){dim_z = dim_z[1]}
  path <- paste(system.file(package="NIMIWAE"), "NIMIWAE.py", sep="/")
  # print(path)
  # print(ls())
  reticulate::source_python(path)  # can we do this in an R package?
  # FUN = eval(paste("run_",parse(text=method),sep="")) # change run_NIMIWAE_N16 to just run_NIMIWAE
  # FUN = match.fun(paste("run_",parse(text=method),sep=""))      # this fx not found because not imported from NIMIWAE.py upon library(NIMIWAE)?
  if(is.null(FUN)){FUN = run_NIMIWAE}      # this fx not found because not imported from NIMIWAE.py upon library(NIMIWAE)?

  p = ncol(data)

  # datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  # Missings = split(data.frame(Missing), g)

  if(all(is.null(g))){
    # if g is not specified, makes most sense to impute entire dataset
    # randomly assign 80-20 ratio train-valid, then entire data is set as test set
    ratios=c(train = .8, valid = .2)
    set.seed(333)
    g = sample(cut(
      seq(nrow(data)),
      nrow(data)*cumsum(c(0,ratios)),
      labels = names(ratios)
    ))
    datas = split(data.frame(data), g)        # split data into train-valid sets
    Missings = split(data.frame(Missing), g)

    # datas$test = data   # test set is entire data
    # Missings$test = Missing

    datas$test = datas$train   # test set is training set
    Missings$test = Missings$train

    print("test dataset")
    print(dim(datas$test))
    print(dim(Missings$test))
  }else{
    datas = split(data.frame(data), g)        # split by $train, $test, and $valid (custom)
    Missings = split(data.frame(Missing), g)
    if(is.null(datas$test)){
      datas$test = datas$train; Missings$test = Missings$train
    }
  }

  # probs_Missing = split(data.frame(prob_Missing),g)
  if(normalize){ norm_means=apply(datas$train,2,function(x) mean(x, na.rm=T)); norm_sds=apply(datas$train,2,function(x) sd(x, na.rm=T))
  }else{ norm_means=rep(0,ncol(datas$train)); norm_sds=rep(1,ncol(datas$train))}

  norm_means[data_types=="cat"] = 0; norm_sds[data_types=="cat"] = 1

  test_epochs=2L

  torch = reticulate::import("torch")
  np = reticulate::import("numpy")

  if(grepl("run_NIMIWAE",as.character(FUN))){
    if(learn_r | ignorable){phi0=NULL; phi=NULL}else{phi0=np$array(phi0); phi=np$array(phi)}
    list_train = list()

    #partial_opt=FALSE; nits=1L; nGibbs=0L; input_r="r"
    pre_impute_value=0L
    # add_miss_term = F; draw_xobs=F; draw_xmiss=T    # deprecated

    # betaVAE=F # default: no beta-VAE (need to remove from python, or can keep this option?)
    # if(betaVAE){ beta=0; beta_anneal_rate=1/500
    # }else{ beta=1; beta_anneal_rate=0 }

    if(arch=="VAE"){ niws=1L }
    #M=1L

    sparse="none"; dropout_pct=0; L1_weights=0; L2_weights=0
    # sparse="L1"; L1_weights=c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8); L2_weights=0; dropout_pct=0  # Exp 1
    # sparse="L2"; L1_weights=0; L2_weights=c(0, 0.004, 0.04, 0.1, 0.4, 0.8); dropout_pct=0  # Exp 6
    # sparse="dropout"; L1_weights=0; L2_weights=0; dropout_pct=0.9 # Exp 3

    # sparse="dropout"; dropout_pct=50; L1_weights=0; L2_weight=0
    # sparse="L1"; L1_weights=c(0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5); L2_weight=0; dropout_pct=0  # D3
    # sparse="L1"; L1_weights=c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5); L2_weight=0; dropout_pct=0  # D4
    # sparse="L1"; L1_weights=c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5); L2_weight=4e-3; dropout_pct=0  # D5: Weight decay of 4e-3: https://dejanbatanjac.github.io/2019/07/02/Impact-of-WD.html, https://arxiv.org/pdf/1803.09820.pdf
    # dim_zs = c(64L,128L)    # D6: way overparametrized Z

    if(ignorable){n_hidden_layers_r0=0L}  # no need to tune this parameter if ignorable model..

    if(is.null(n_hidden_layers_r0)){ n_hidden_layers_r = 999L # dummy set it to 999. replace in for loop with value of n_hidden_layers
    } else{n_hidden_layers_r = n_hidden_layers_r0}   # if not null, set to specified value(s)

    # n_combs_params=length(h)*length(h_r)*length(bs)*length(lr)*length(dim_z)*length(niws)*length(n_epochs)*length(n_hidden_layers)*length(n_hidden_layers_r)*length(L1_weights)*length(L2_weights)
    length_nodes_HLs = (length(n_hidden_layers)-1)*length(h)+1
    length_nodes_HLs_r = (length(n_hidden_layers_r)-1)*length(h_r)+1    # don't have to tune h when nhl = 0
    n_combs_params = length(bs)*length(lr)*length(dim_z)*length(niws)*length(n_epochs)* (length_nodes_HLs*length_nodes_HLs_r) *length(L1_weights)*length(L2_weights)
    LBs_trainVal = matrix(NA,nrow=n_combs_params,ncol=8+3+2+1)   # contain params, trainMSE,valMSE,trainLB,valLB
    colnames(LBs_trainVal) = c("bs","lr","dim_z","niw","n_epoch","nhls","nhls_r","h","h_r","L1_weights",
                               "LB_train","L1_train","LB_valid","L1_valid")
    index=1
    for(j in 1:length(bs)){for(k in 1:length(lr)){for(l in 1:length(dim_z)){
      for(m in 1:length(niws)){for(mm in 1:length(n_epochs)){for(nn in 1:length(n_hidden_layers)){
        for(nr in 1:length(n_hidden_layers_r)){

          ### if #HL = 0, no need to tune h ###
          if(n_hidden_layers[nn]==0){h0=0L} else{h0=h}
          if(n_hidden_layers_r[nr]==0){h_r0=0L} else{h_r0=h_r}

          for(i in 1:length(h0)){for(ii in 1:length(h_r0)){
            for(o1 in 1:length(L1_weights)){for(o2 in 1:length(L2_weights)){        # h1: encoder     q(z|xo (,r))
        # h2: decoder_x   p(x|z)
        # h3: decoder_r   p(r|x (,z))
        # h4: decoder_xr  p(x|z,r)
        impute_bs = bs[j] # batch_size in imputation same as batch_size in training

        print(paste("h:",h0[i],", bs:",bs[j],", lr:",lr[k],", dim_z:",dim_z[l],", niw:",niws[m],", n_epochs: ",n_epochs[mm],", n_hls: ",n_hidden_layers[nn],", n_hls_r: ",n_hidden_layers_r[nr],", L1_weight: ", L1_weights[o1], ", L2_weight: ", L2_weights[o2], sep=""))
        print(paste("M:", n_imputations))

        warm_started_model=NULL; warm_start=F
        # if(oo==1){warm_started_model = NULL; warm_start=F}else{warm_started_model = res_train$'saved_model'; warm_start=T}   # warm starts

        # fix h3=0: Logistic Regression for p(r|x)
        if(is.null(n_hidden_layers_r0)){ n_hidden_layers_r = n_hidden_layers[nn] }   # n_hidden_layers for R is same as other NNs
        # n_hidden_layers_r = 0L

        niw = 5L
        # niw = 25L
        # niw = 1L   # trying this test
        print(paste("Training niw:",niw))
        res_train = FUN(rdeponz=rdeponz, data=np$array(datas$train),data_types=np$array(data_types),data_types_0=np$array(data_types_0),data_val=np$array(datas$valid),Missing=np$array(Missings$train),Missing_val=np$array(Missings$valid),#probMissing=np$array(probs_Missing$train),
                        covars_r=np$array(covars_r), norm_means=np$array(norm_means), norm_sds=np$array(norm_sds), learn_r=learn_r, Cs=Cs,
                        ignorable=ignorable,n_hidden_layers=n_hidden_layers[nn], n_hidden_layers_r=n_hidden_layers_r[nr],
                        L1_weight=L1_weights[o1],L2_weight=L2_weights[o2],sparse=sparse,dropout_pct=dropout_pct,prune_pct=NULL,
                        covars_miss=NULL,covars_miss_val=NULL,impute_bs=impute_bs,
                        arch=arch, draw_xmiss=draw_xmiss,
                        #add_miss_term=add_miss_term,#draw_xobs=draw_xobs,draw_xmiss=draw_xmiss,
                        pre_impute_value=pre_impute_value,h1=h0[i],h2=h0[i],h3=h_r0[ii],h4=h0[i], #beta=beta,beta_anneal_rate=beta_anneal_rate,
                        phi0=phi0, phi=phi, warm_start=warm_start, saved_model=warm_started_model, early_stop = early_stop, train=1L,
                        # sigma=sigma, bs = bs[j], n_epochs = n_epochs[mm], lr=lr[k], niw=niws[m], dim_z=dim_z[l], L=niws[m], M=n_imputations, dir_name=dir_name, save_imps=F) #M=100L)
                        sigma=sigma, bs = bs[j], n_epochs = n_epochs[mm], lr=lr[k], niw=niw, dim_z=dim_z[l], L=niw, M=n_imputations, dir_name=dir_name, save_imps=F) #M=100L)
                        # sigma=sigma, bs = bs[j], n_epochs = n_epochs[mm], lr=lr[k], niw=5L, dim_z=dim_z[l], L=5L, M=5L, dir_name=dir_name, save_imps=F) #M=100L)

        res_valid = FUN(rdeponz=rdeponz, data=np$array(datas$valid),data_types=np$array(data_types),data_types_0=np$array(data_types_0),data_val=np$array(datas$valid),Missing=np$array(Missings$valid),Missing_val=np$array(Missings$valid),#probMissing=np$array(probs_Missing$valid),
                        covars_r=np$array(covars_r), norm_means=np$array(norm_means), norm_sds=np$array(norm_sds), learn_r=learn_r, Cs=Cs,
                        ignorable=ignorable,n_hidden_layers=n_hidden_layers[nn], n_hidden_layers_r=n_hidden_layers_r[nr],
                        L1_weight=L1_weights[o1],L2_weight=L2_weights[o2],sparse=sparse,dropout_pct=dropout_pct,prune_pct=NULL,
                        covars_miss=NULL,covars_miss_val=NULL,impute_bs=impute_bs,
                        arch=arch, draw_xmiss=draw_xmiss,
                        #add_miss_term=add_miss_term,draw_xobs=draw_xobs,draw_xmiss=draw_xmiss,
                        pre_impute_value=pre_impute_value,h1=h0[i],h2=h0[i],h3=h_r0[ii],h4=h0[i], #beta=1,beta_anneal_rate=0,
                        phi0=phi0, phi=phi, warm_start=F, saved_model=res_train$'saved_model', early_stop = F, train=0L,
                        # sigma=sigma, bs = bs[j], n_epochs=test_epochs, lr=lr[k], niw=niws[m], dim_z=dim_z[l], L=niws[m], M=n_imputations, dir_name=dir_name, save_imps=F) #M=100L)
                        sigma=sigma, bs = bs[j], n_epochs=test_epochs, lr=lr[k], niw=niw, dim_z=dim_z[l], L=niw, M=n_imputations, dir_name=dir_name, save_imps=F) #M=100L)
                        # sigma=sigma, bs = bs[j], n_epochs=test_epochs, lr=lr[k], niw=5L, dim_z=dim_z[l], L=5L, M=5L, dir_name=dir_name, save_imps=F) #M=100L)

        # print("all_params")
        # print(res_valid$all_params)

        print(c(bs[j],lr[k],dim_z[l],niws[m],res_train$train_params$early_stop_epochs,
                n_hidden_layers[nn],n_hidden_layers_r[nr],h0[i],h_r0[ii],L1_weights[o1],L2_weights[o2],
                res_train$'LB',res_train$'L1s'$'miss',res_valid$'LB',res_valid$'L1s'$'miss'))
        LBs_trainVal[index,]=c(bs[j],lr[k],dim_z[l],niws[m],res_train$train_params$early_stop_epochs,
                               n_hidden_layers[nn],n_hidden_layers_r[nr],h0[i],h_r0[ii],L1_weights[o1],
                               res_train$'LB',res_train$'L1s'$'miss',res_valid$'LB',res_valid$'L1s'$'miss')
        save(LBs_trainVal, file=sprintf("%s/LBs_trainVal",dir_name))

        print(LBs_trainVal[1:index,])
        print(paste0("Search grid #",index," of ",n_combs_params))
        if(is.na(res_valid$'LB')){res_valid$'LB'=-Inf}

        # save only the best result currently (not all results) --> save memory
        if(index==1){opt_train = res_train; opt_LB = res_valid$'LB'; save(opt_train,file=sprintf("%s/temp_opt_train.out",dir_name)); torch$save(opt_train$'saved_model',sprintf("%s/temp_opt_train_saved_model.pth",dir_name))  #; save(opt_train, file="temp_opt_train.out")
        }else if(res_valid$'LB' > opt_LB){opt_train = res_train; opt_LB = res_valid$'LB'; save(opt_train,file=sprintf("%s/temp_opt_train.out",dir_name)); torch$save(opt_train$'saved_model',sprintf("%s/temp_opt_train_saved_model.pth",dir_name))} #; save(opt_train, file="temp_opt_train.out")

        # if(length(L1_weights)>1){warm_started_model = res_train$'saved_model'; warm_start=T; early_stop=T}  # warm starts

        rm(opt_train)
        rm(res_train)
        rm(res_valid)
        index=index+1
      }}}}}}}}}}}
    #opt_id=which.min(LBs)
    #opt_params=list_train[[opt_id]]$'train_params'
    print("Hyperparameter tuning complete.")

    load(sprintf("%s/temp_opt_train.out",dir_name))
    saved_model = torch$load(sprintf("%s/temp_opt_train_saved_model.pth",dir_name))

    opt_params = opt_train$'train_params' #; saved_model = opt_train$'saved_model'

    # batch_size = opt_params$'bs'   # runs out of memory: taking more samples --> need smaller batch size
    # test_bs = 500L
    test_bs = as.integer(opt_params$'bs'/10)
    opt_params$'test_bs' = test_bs

    # assuming niws is not tuned. use user-custom L and M (niw in training/validation time set to 5 for computational efficiency)
    #opt_params$'niw'=niws[1]
    opt_params$'L'= niws[1]; opt_params$'M' = n_imputations
    res_test = FUN(rdeponz=rdeponz, data=np$array(datas$test),data_types=np$array(data_types),data_types_0=np$array(data_types_0),data_val=np$array(datas$valid),Missing=np$array(Missings$test),Missing_val=np$array(Missings$valid),#probMissing=np$array(probs_Missing$test),
                   covars_r=np$array(covars_r), norm_means=np$array(norm_means), norm_sds=np$array(norm_sds), learn_r=learn_r, Cs=Cs,
                   L1_weight=opt_params$'L1_weight',L2_weight=opt_params$'L2_weight', ignorable=ignorable, n_hidden_layers=opt_params$'n_hidden_layers', n_hidden_layers_r=opt_params$'n_hidden_layers_r',
                   sparse=sparse, dropout_pct=dropout_pct,prune_pct=NULL,covars_miss=NULL,covars_miss_val=NULL,impute_bs=test_bs,
                   arch=arch, draw_xmiss=draw_xmiss,
                   #add_miss_term=add_miss_term,draw_xobs=draw_xobs,draw_xmiss=draw_xmiss,
                   pre_impute_value=pre_impute_value,h1=opt_params$'h1',h2=opt_params$'h2',h3=opt_params$'h3',h4=opt_params$'h4',#beta=1,beta_anneal_rate=0,
                   phi0=phi0, phi=phi, warm_start=F, saved_model=saved_model, early_stop=F, train=0L,
                   sigma=opt_params$'sigma',bs = test_bs, n_epochs = test_epochs,lr=opt_params$'lr',niw=opt_params$'niw',dim_z=opt_params$'dim_z',L=opt_params$'L', M=opt_params$'M', dir_name=dir_name, save_imps=save_imps)

    print(c(opt_params$'h1', opt_params$'bs', opt_params$'lr', opt_params$'dim_z', opt_params$'niw',
            opt_params$'n_hidden_layers', opt_params$'n_hidden_layers_r', opt_params$'L1_weight', opt_params$'L2_weight',
            res_test$'LB'))

    print("dim(datas$test):")
    print(dim(datas$test))
    print("dim(res_test$xhat)")
    print(dim(res_test$xhat))

    res_test$opt_params=opt_params

    res_test$train_LB_epoch = opt_train$NIMIWAE_LB_epoch
    res_test$val_LB_epoch = opt_train$NIMIWAE_val_LB_epoch
    # res_test$xhat_rev = reverse_norm_MIWAE(res_test$xhat,norm_means,norm_sds)
    res_test$xhat_rev = res_test$xhat   # reverse normalization done inside Python
    res_test$LBs_trainVal = LBs_trainVal

    res_test$g = g

    return(res_test)
  } else if(grepl("run_MIWAE",as.character(FUN))){
    dec_distrib="Normal"
    n_combs_params=length(h)*length(bs)*length(lr)*length(dim_z)*length(niws)*length(n_epochs)*length(n_hidden_layers)
    list_train = list()
    LBs = rep(NA,n_combs_params)
    index=1
    # loop train --> valid
    for(i in 1:length(h)){for(j in 1:length(bs)){for(k in 1:length(lr)){for(l in 1:length(dim_z)){
      for(m in 1:length(niws)){for(mm in 1:length(n_epochs)){for(nn in 1:length(n_hidden_layers)){

        print(paste("h:",h[i],", bs:",bs[j],", lr:",lr[k],", dim_z:",dim_z[l],", niw:",niws[m],", n_epochs: ",n_epochs[mm],", n_hls: ",n_hidden_layers[nn],sep=""))
        res_train = FUN(data=np$array(datas$train),Missing=np$array(Missings$train),
                        norm_means=np$array(norm_means),norm_sds=np$array(norm_sds),
                        n_hidden_layers=n_hidden_layers[nn],
                        dec_distrib=dec_distrib, train=1L, h=h[i],
                        sigma=sigma,bs = bs[j],
                        n_epochs = n_epochs[mm],lr=lr[k],niw=niws[m],dim_z=dim_z[l],L=niws[m])

        res_valid = FUN(data=np$array(datas$valid),Missing=np$array(Missings$valid),
                        norm_means=np$array(norm_means),norm_sds=np$array(norm_sds),
                        n_hidden_layers=n_hidden_layers[nn],
                        dec_distrib=dec_distrib, train=0L, saved_model=res_train$'saved_model', h=h[i],
                        sigma=sigma,bs = bs[j],
                        n_epochs = test_epochs,lr=lr[k],niw=niws[m],dim_z=dim_z[l],L=niws[m])

        #list_train[[index]] = res_train
        #LBs[index]=res_valid$'LB'

        if(is.na(res_valid$'LB')){res_valid$'LB'=-Inf}

        # save only the best result currently (not all results) --> save memory
        if(index==1){opt_train = res_train; opt_LB = res_valid$'LB'
        }else if(res_valid$'LB' > opt_LB){opt_train=res_train; opt_LB = res_valid$'LB'
        }

        index=index+1
      }}}}}}}
    #opt_id=which.min(LBs)
    #opt_params=list_train[[opt_id]]$'train_params'
    print("Hyperparameter tuning complete.")

    opt_params = opt_train$'train_params'


    res_test = FUN(data=np$array(datas$test),Missing=np$array(Missings$test),
                   norm_means=np$array(norm_means),norm_sds=np$array(norm_sds),
                   n_hidden_layers=opt_params$'n_hidden_layers',
                   dec_distrib=dec_distrib, train=0L, saved_model=opt_train$'saved_model', h=opt_params$'h',
                   sigma=opt_params$'sigma',bs = opt_params$'bs',
                   n_epochs = test_epochs,lr=opt_params$'lr',niw=opt_params$'niw',dim_z=opt_params$'dim_z',L=opt_params$'L')

    res_test$opt_params=opt_params

    res_test$xhat_rev = reverse_norm_MIWAE(res_test$xhat,norm_means,norm_sds) # reverse normalization for proper comparison

    res_test$g = g
    return(res_test)
  } else if(grepl("run_HIVAE",as.character(FUN))){
    #bss=min(nrow(datas$train),200000L)
    if(dataset=="HEPMASS"){
      bs = 150000L   # need to put in FSCseqPaper instead
    }else if(dataset%in%c("GAS","POWER","MINIBOONE")){
      bs = 500000L   # need to put in FSCseqPaper instead
    }
    ############################ HIVAE
    dim_latent_y = c(5L,10L)   # need to put in FSCseqPaper instead
    n_combs_params=length(bs)*length(lr)*length(dim_z)*length(n_epochs)*length(dim_latent_y)

    list_train = list()
    LBs = rep(NA,n_combs_params)
    index=1
    dim_latent_s = 10L #; dim_latent_y=10L # default
    model_name = "model_HIVAE_inputDropout"

    for(i in 1:length(h)){for(j in 1:length(bs)){for(k in 1:length(lr)){for(l in 1:length(dim_z)){
      for(mm in 1:length(n_epochs)){for(yy in 1:length(dim_latent_y)){
        print(paste("h:",h[i],", bs:",bs[j],", lr:",lr[k],", dim_z:",dim_z[l],", n_epochs: ",n_epochs[mm], ", dim_latent_y: ", dim_latent_y[yy],sep=""))
        save_file = sprintf("HIVAE_interm_%f_%d_%d_%d_%d_%d_%s",lr[k],bs[j],n_epochs[mm],dim_latent_s,dim_z[l],dim_latent_y[yy],model_name)
        res_train = FUN(data=np$array(datas$train),Missing=np$array(Missings$train),data_types_HIVAE=data_types_HIVAE,
                        lr=lr[k],bs=bs[j],n_epochs=n_epochs[mm],train=1L,
                        display=100L, n_save=1000L, restore=0L, dim_latent_s=dim_latent_s, dim_latent_z=dim_z[l],
                        dim_latent_y=dim_latent_y[yy], model_name=model_name,save_file=save_file)
        res_valid = FUN(data=np$array(datas$valid),Missing=np$array(Missings$valid),data_types_HIVAE=data_types_HIVAE,
                        lr=lr[k],bs=10000000L,n_epochs=1L,train=0L,
                        display=100L, n_save=1000L, restore=1L, dim_latent_s=dim_latent_s, dim_latent_z=dim_z[l],
                        dim_latent_y=dim_latent_y[yy], model_name=model_name,save_file=save_file)
        #list_train[[index]] = res_train
        #LBs[index]=res_valid$'mean_loss'

        if(is.na(res_valid$'mean_loss')){res_valid$'mean_loss'=-Inf}

        if(index==1){opt_train = res_train; opt_LB = res_valid$'mean_loss'
        }else if(res_valid$'mean_loss' > opt_LB){opt_train=res_train; opt_LB = res_valid$'mean_loss'
        }
        index=index+1
      }}}}}}
    print("Hyperparameter tuning complete.")

    opt_params=opt_train$'train_params'
    print("opt params:")
    print(opt_params)
    opt_save_file = sprintf("HIVAE_interm_%f_%d_%d_%d_%d_%d_%s",
                            opt_params$'lr',opt_params$'bs',opt_params$'n_epochs',
                            opt_params$'dim_latent_s',opt_params$'dim_latent_z',
                            opt_params$'dim_latent_y',opt_params$'model_name')
    res_test = FUN(data=np$array(datas$test),Missing=np$array(Missings$test),data_types=data_types,
                   lr=opt_params$'lr',bs=10000000L,n_epochs=1L,train=0L,
                   display=100L, n_save=1000L, restore=1L, dim_latent_s=opt_params$'dim_latent_s', dim_latent_z=opt_params$'dim_latent_z',
                   dim_latent_y=opt_params$'dim_latent_y', model_name="model_HIVAE_inputDropout",save_file=opt_save_file)

    res_test$opt_params=opt_params

    res_test$g = g
    return(res_test)
  } else if(grepl("run_VAEAC",as.character(FUN))){
    ############################# VAEAC
    # default: h=256, n_hidden_layers=10, dim_z=64, bs=64

    norm_means[one_hot_max_sizes>=2] = 0; norm_sds[one_hot_max_sizes>=2] = 1
    n_epochs = c(200L)
    n_combs_params=length(h)*length(bs)*length(lr)*length(dim_z)*length(n_epochs)*length(n_hidden_layers)

    list_train = list()
    LBs = rep(NA,n_combs_params)
    n_imputations=5L; validation_ratio=0.2; validations_per_epoch=1L; validation_iwae_n_samples=25L  # ; n_hidden_layers=10L   # put this in the tune_hyperparameters() function
    index=1

    for(i in 1:length(h)){for(j in 1:length(bs)){for(k in 1:length(lr)){for(l in 1:length(dim_z)){for(nn in 1:length(n_hidden_layers)){
      for(mm in 1:length(n_epochs)){for(nn in 1:length(n_hidden_layers)){
        print(paste("h:",h[i],", bs:",bs[j],", lr:",lr[k],", dim_z:",dim_z[l],", n_epochs: ",n_epochs[mm], ", n_hls: ", n_hidden_layers[nn],sep=""))
        save_file = sprintf("VAEAC_interm_%d_%d_%f_%d_%d_%d_%d_%d_%f_%d",
                            h[i],n_hidden_layers[nn],lr[k],bs[j],n_epochs[mm],dim_z[l],
                            n_imputations,validations_per_epoch,validation_ratio,validation_iwae_n_samples)
        res_train = FUN(data=np$array(rbind(MissingDatas$train,MissingDatas$valid)),one_hot_max_sizes=one_hot_max_sizes,
                        norm_mean=np$array(norm_means),norm_std=np$array(norm_sds),
                        h=h[i], n_hidden_layers=n_hidden_layers[nn], dim_z=dim_z[l],bs=bs[j],lr=lr[k],output_file=save_file,
                        train=1L,saved_model=NULL,saved_networks=NULL,
                        n_epochs=n_epochs[mm],n_imputations=n_imputations,validation_ratio=validation_ratio,
                        validations_per_epoch=validations_per_epoch,
                        validation_iwae_n_samples=validation_iwae_n_samples,restore=FALSE)
        # VAEAC does the validation inside their own function

        #list_train[[index]] = res_train
        #LBs[index]=res_train$'LB'

        if(is.na(res_train$'LB')){res_train$'LB'=-Inf}

        if(index==1){opt_train = res_train; opt_LB = res_train$'LB'
        }else if(res_train$'LB' > opt_LB){opt_train=res_train; opt_LB = res_train$'LB'
        }

        index=index+1
      }}}}}}}
    print("Hyperparameter tuning complete.")

    #opt_id=which.min(LBs)
    opt_params=opt_train$'train_params'
    opt_save_file = sprintf("VAEAC_interm_%d_%d_%f_%d_%d_%d_%d_%d_%f_%d",
                            opt_params$'h',opt_params$'n_hidden_layers',opt_params$'lr',opt_params$'bs',opt_params$'n_epochs',
                            opt_params$'dim_z',opt_params$'n_imputations',opt_params$'validations_per_epoch',
                            opt_params$'validation_ratio',opt_params$'validation_iwae_n_samples')
    res_test = FUN(data=np$array(MissingDatas$test),one_hot_max_sizes=one_hot_max_sizes,
                   norm_mean=np$array(norm_means),norm_std=np$array(norm_sds),
                   h=opt_params$'h', n_hidden_layers=opt_params$'n_hidden_layers', dim_z=opt_params$'dim_z',
                   bs=opt_params$'bs',lr=opt_params$'lr',output_file=opt_save_file, train=0L, saved_model=opt_train$'saved_model', saved_networks=opt_train$'saved_networks',
                   n_epochs=opt_params$'n_epochs',n_imputations=opt_params$'n_imputations',validation_ratio=opt_params$'validation_ratio',
                   validations_per_epoch=opt_params$'validations_per_epoch',
                   validation_iwae_n_samples=opt_params$'validation_iwae_n_samples',restore=TRUE)
    res_test$opt_params=opt_params

    res_test$g = g
    return(res_test)
  } else if(grepl("run_missForest",as.character(FUN)) | grepl("run_meanImputation",as.character(FUN))){
    res = FUN(data=datas$test, Missing=Missings$test)
    xhat = if(grepl("run_missForest",as.character(FUN))){res$xhat_mf}else{res$xhat_mean}
    res$xhat_rev = reverse_norm_MIWAE(xhat,norm_means,norm_sds)
    res$g = g
    return(res)
  }
}
