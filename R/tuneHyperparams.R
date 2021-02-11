## NTD:
# 1) take out all "run_"
# 2) make sigma, hs, bss, lrs, dim_zs, niws, n_epochs customizable? (kinda is, but create defaults and take out hard coding --> put in Paper repo)

#' Hyperparameter tuning used by NIMIWAE() function
#'
#' @param FUN run_<> function for other methods (not NIMIWAE)
#' @param method String specifying the method to tune hyperparameters. Can be "NIMIWAE" (default), "MIWAE", "VAEAC", "HIVAE", "MEAN", or "MF". Hyperparameters are not tuned for "MEAN" or "MF"
#' @param data Data matrix (N x P)
#' @param Missing Missingness mask matrix (N x P)
#' @param g Training-validation-test split partitioning
#' @param rdeponz TRUE/FALSE: Whether to allow missingness (r) to depend on the latent variable (z). Default is FALSE
#' @param learn_r TRUE/FALSE: Whether to learn missingness model via appended NN (TRUE, default), or fit a known logistic regression model (FALSE). If FALSE, `phi0` and `phi` must be specified
#' @param phi0 (optional) Intercept of logistic regression model, if learn_r = FALSE.
#' @param phi (optional) Vector of coefficients of logistic regression model for each input covariates `covars_r`, if learn_r = FALSE. `phi` must be the same length as the number of input covariates, or `sum(covars_r)`.
#' @param ignorable TRUE/FALSE: Whether missingness is ignorable (MCAR/MAR) or nonignorable (MNAR, default). If missingness is known to be ignorable, "ignorable=T" omits missingness model.
#' @param covars_r Vector of 1's and 0's of whether each feature is included as covariates in the missingness model. Need not be specified if `ignorable = T`. Default is using all features as covariates in missingness model. Must be length P (or `ncol(data)`)
#' @param arch Architecture of NIMIWAE. Can be "IWAE" or "VAE". "VAE" is specific case of the "IWAE" where only one sample is drawn from the joint posterior of (z, xm).
#' @param sigma activation function ("relu" or "elu")
#' @param h integer, number of nodes per hidden layer
#' @param n_hidden_layers integer, #hidden layers (except missingness model Decoder_r)
#' @param n_hidden_layers_r integer, #hidden layers for Decoder_r (default: 0)
#' @param bs integer, batch size (training)
#' @param lr float, learning rate
#' @param dim_z integer, dimensionality of latent z. Default: 5
#' @param niw integer, number of importance weights (samples drawn from each latent space). Default: 5
#' @param n_epoch integer, maximum number of epochs (without early stop). Default: 2002
#' @param data_types Specify for HIVAE only.
#' @param one_hot_max_sizes Specify for VAEAC only.
#' @param ohms Specify for VAEAC only.
#' @param MissingDatas Specify for VAEAC only.
#' @return res object: method's fit on test set, after training on training set and validating best set of hyperparameter values using the validation set.
#'
#' @author David K. Lim, \email{deelim@live.unc.edu}
#' @references \url{https://github.com/DavidKLim/NIMIWAE}
#'
#' @importFrom reticulate source_python import
#'
#' @export
tuneHyperparams = function(FUN=NULL,method="NIMIWAE",dataset,data, Missing, g,
                           rdeponz=F, learn_r=T, phi0=NULL, phi=NULL, ignorable=F, covars_r=rep(1,ncol(data)),
                           arch="IWAE",   # for NIMIWAE: whether each NN is optimized separately, architecture: VAE or IWAE
                           sigma="elu", h=c(128L,64L), n_hidden_layers=c(1L,2L), n_hidden_layers_r=0L, bs=1000L, lr=c(0.001,0.01),
                           dim_z=as.integer(c(floor(ncol(data)/2),floor(ncol(data)/4))), niws=5L, n_epochs=2002L,
                           data_types=NULL, one_hot_max_sizes=NULL, ohms=NULL,
                           MissingDatas = NULL # just for vaeac
){

  path <- paste(system.file(package="NIMIWAE"), "NIMIWAE.py", sep="/")
  # print(path)
  # print(ls())
  reticulate::source_python(path)  # can we do this in an R package?
  # FUN = eval(paste("run_",parse(text=method),sep="")) # change run_NIMIWAE_N16 to just run_NIMIWAE
  # FUN = match.fun(paste("run_",parse(text=method),sep=""))      # this fx not found because not imported from NIMIWAE.py upon library(NIMIWAE)?
  if(is.null(FUN)){FUN = run_NIMIWAE}      # this fx not found because not imported from NIMIWAE.py upon library(NIMIWAE)?

  p = ncol(data)

  datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  Missings = split(data.frame(Missing), g)
  # probs_Missing = split(data.frame(prob_Missing),g)
  norm_means=colMeans(datas$train); norm_sds=apply(datas$train,2,sd)
  test_epochs=2L

  torch = reticulate::import("torch")
  np = reticulate::import("numpy")

  if(grepl("run_NIMIWAE",as.character(FUN))){
    if(learn_r){phi0=NULL; phi=NULL}else{phi=np$array(phi)}
    list_train = list()

    include_xo=TRUE; betaVAE = F; dec_distrib="Normal"    # code this out of run_NIMIWAE python function?
    #partial_opt=FALSE; nits=1L; nGibbs=0L; input_r="r"
    add_miss_term = F; draw_xobs=F; draw_xmiss=T; pre_impute_value=0L

    betaVAE=F # default: no beta-VAE (need to remove from python, or can keep this option?)
    if(betaVAE){ beta=0; beta_anneal_rate=1/500
    }else{ beta=1; beta_anneal_rate=0 }

    if(arch=="VAE"){ niws=1L }
    #M=1L

    sparse="none"; dropout_pct=NULL; L1_weights=0; L2_weight=0
    #sparse="L1"; L1_weights=c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8); L2_weight=0; dropout_pct=NULL  # D2

    # sparse="dropout"; dropout_pct=50; L1_weights=0; L2_weight=0
    # sparse="L1"; L1_weights=c(0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5); L2_weight=0; dropout_pct=NULL  # D3
    # sparse="L1"; L1_weights=c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5); L2_weight=0; dropout_pct=NULL  # D4
    # sparse="L1"; L1_weights=c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5); L2_weight=4e-3; dropout_pct=NULL  # D5: Weight decay of 4e-3: https://dejanbatanjac.github.io/2019/07/02/Impact-of-WD.html, https://arxiv.org/pdf/1803.09820.pdf
    # dim_zs = c(64L,128L)    # D6: way overparametrized Z

    n_combs_params=length(h)*length(bs)*length(lr)*length(dim_z)*length(niws)*length(n_epochs)*length(n_hidden_layers)*length(n_hidden_layers_r)*length(L1_weights)
    LBs_trainVal = matrix(NA,nrow=n_combs_params,ncol=8+4)   # contain params, trainMSE,valMSE,trainLB,valLB
    colnames(LBs_trainVal) = c("h","bs","lr","dim_z","niw","n_epoch","n_hidden_layers","L1_weights",
                               "LB_train","MSE_train","LB_valid","MSE_valid")
    index=1
    for(i in 1:length(h)){for(j in 1:length(bs)){for(k in 1:length(lr)){for(l in 1:length(dim_z)){
      for(m in 1:length(niws)){for(mm in 1:length(n_epochs)){for(nn in 1:length(n_hidden_layers)){for(oo in 1:length(L1_weights)){        # h1: encoder     q(z|xo (,r))
        # h2: decoder_x   p(x|z)
        # h3: decoder_r   p(r|x (,z))
        # h4: decoder_xr  p(x|z,r)
        impute_bs = bs[j] # batch_size in imputation same as batch_size in training

        print(paste("h:",h[i],", bs:",bs[j],", lr:",lr[k],", dim_z:",dim_z[l],", niw:",niws[m],", n_epochs: ",n_epochs[mm],", n_hls: ",n_hidden_layers[nn],", L1_weight: ", L1_weights[oo], sep=""))

        if(oo==1){warm_started_model = NULL; warm_start=F}else{warm_started_model = res_train$'saved_model'; warm_start=T}
        # fix h3=0: Logistic Regression for p(r|x)
        res_train = FUN(rdeponz=rdeponz, data=np$array(datas$train),data_val=np$array(datas$valid),Missing=np$array(Missings$train),Missing_val=np$array(Missings$valid),#probMissing=np$array(probs_Missing$train),
                        covars_r=np$array(covars_r), norm_means=np$array(norm_means), norm_sds=np$array(norm_sds), learn_r=learn_r,
                        ignorable=ignorable,n_hidden_layers=n_hidden_layers[nn], n_hidden_layers_r=0L,
                        L1_weight=L1_weights[oo],L2_weight=L2_weight,unnorm=F,sparse=sparse,dropout_pct=dropout_pct,prune_pct=NULL,covars_miss=NULL,covars_miss_val=NULL,impute_bs=impute_bs,include_xo=include_xo,
                        arch=arch,
                        add_miss_term=add_miss_term,draw_xobs=draw_xobs,draw_xmiss=draw_xmiss,
                        pre_impute_value=pre_impute_value,h1=h[i],h2=h[i],h3=0,h4=h[i],beta=beta,beta_anneal_rate=beta_anneal_rate,
                        phi0=phi0, phi=phi, warm_start=warm_start, saved_model=warm_started_model, dec_distrib=dec_distrib, train=1L,
                        sigma=sigma, bs = bs[j], n_epochs = n_epochs[mm], lr=lr[k], niw=niws[m], dim_z=dim_z[l], L=niws[m], M=niws[m])

        res_valid = FUN(rdeponz=rdeponz, data=np$array(datas$valid),data_val=np$array(datas$valid),Missing=np$array(Missings$valid),Missing_val=np$array(Missings$valid),#probMissing=np$array(probs_Missing$valid),
                        covars_r=np$array(covars_r), norm_means=np$array(norm_means), norm_sds=np$array(norm_sds), learn_r=learn_r,
                        ignorable=ignorable,n_hidden_layers=n_hidden_layers[nn], n_hidden_layers_r=0L,
                        L1_weight=L1_weights[oo],L2_weight=L2_weight,unnorm=F,sparse=sparse,dropout_pct=dropout_pct,prune_pct=NULL,covars_miss=NULL,covars_miss_val=NULL,impute_bs=impute_bs,include_xo=include_xo,
                        arch=arch,
                        add_miss_term=add_miss_term,draw_xobs=draw_xobs,draw_xmiss=draw_xmiss,
                        pre_impute_value=pre_impute_value,h1=h[i],h2=h[i],h3=0,h4=h[i],beta=1,beta_anneal_rate=0,
                        phi0=phi0, phi=phi, warm_start=F, saved_model=res_train$'saved_model', dec_distrib=dec_distrib, train=0L,
                        sigma=sigma, bs = bs[j], n_epochs=test_epochs, lr=lr[k], niw=niws[m], dim_z=dim_z[l], L=niws[m], M=niws[m])
        #list_train[[index]] = res_train
        #LBs[index]=res_valid$'LB'
        print(c(h[i],bs[j],lr[k],dim_z[l],niws[m],res_train$train_params$early_stop_epochs,
                n_hidden_layers[nn],L1_weights[oo],
                res_train$'LB',res_train$'MSE'$miss[length(res_train$'MSE'$miss)],res_valid$'LB',res_valid$'MSE'$miss[length(res_valid$'MSE'$miss)]))
        LBs_trainVal[index,]=c(h[i],bs[j],lr[k],dim_z[l],niws[m],res_train$train_params$early_stop_epochs,
                               n_hidden_layers[nn],L1_weights[oo],
                               res_train$'LB',res_train$'MSE'$miss[length(res_train$'MSE'$miss)],res_valid$'LB',res_valid$'MSE'$miss[length(res_valid$'MSE'$miss)])

        print(LBs_trainVal)
        if(is.na(res_valid$'LB')){res_valid$'LB'=-Inf}

        # save only the best result currently (not all results) --> save memory
        if(index==1){opt_train = res_train; opt_LB = res_valid$'LB'; save(opt_train,file="temp_opt_train.out"); torch$save(opt_train$'saved_model',"temp_opt_train_saved_model.pth")  #; save(opt_train, file="temp_opt_train.out")
        }else if(res_valid$'LB' > opt_LB){opt_train=res_train; opt_LB = res_valid$'LB'; save(opt_train,file="temp_opt_train.out"); torch$save(opt_train$'saved_model',"temp_opt_train_saved_model.pth")} #; save(opt_train, file="temp_opt_train.out")

        if(length(L1_weights)>1){warm_started_model = res_train$'saved_model'; warm_start=T; early_stop=T}

        rm(opt_train)
        rm(res_train)
        rm(res_valid)
        index=index+1
      }}}}}}}}
    #opt_id=which.min(LBs)
    #opt_params=list_train[[opt_id]]$'train_params'
    print("Hyperparameter tuning complete.")

    load("temp_opt_train.out")
    saved_model = torch$load("temp_opt_train_saved_model.pth")

    opt_params = opt_train$'train_params' #; saved_model = opt_train$'saved_model'
    res_test = FUN(rdeponz=rdeponz, data=np$array(datas$test),data_val=np$array(datas$valid),Missing=np$array(Missings$test),Missing_val=np$array(Missings$valid),#probMissing=np$array(probs_Missing$test),
                   covars_r=np$array(covars_r), norm_means=np$array(norm_means), norm_sds=np$array(norm_sds), learn_r=learn_r,
                   L1_weight=opt_train$'L1_weight',L2_weight=L2_weight, ignorable=ignorable, n_hidden_layers=opt_params$'n_hidden_layers', n_hidden_layers_r=opt_params$'n_hidden_layers_r',
                   unnorm=F,sparse=sparse, dropout_pct=dropout_pct,prune_pct=NULL,covars_miss=NULL,covars_miss_val=NULL,impute_bs=opt_params$'bs',include_xo=include_xo,
                   arch=arch,
                   add_miss_term=add_miss_term,draw_xobs=draw_xobs,draw_xmiss=draw_xmiss,
                   pre_impute_value=pre_impute_value,h1=opt_params$'h1',h2=opt_params$'h2',h3=opt_params$'h3',h4=opt_params$'h4',beta=1,beta_anneal_rate=0,
                   phi0=phi0, phi=phi, warm_start=F, saved_model=saved_model, dec_distrib=dec_distrib, train=0L,
                   sigma=opt_params$'sigma',bs = opt_params$'bs', n_epochs = test_epochs,lr=opt_params$'lr',niw=opt_params$'niw',dim_z=opt_params$'dim_z',L=opt_params$'L', M=opt_params$'M')

    res_test$opt_params=opt_params

    res_test$train_LB_epoch = opt_train$NIMIWAE_LB_epoch
    res_test$val_LB_epoch = opt_train$NIMIWAE_val_LB_epoch
    res_test$xhat_rev = reverse_norm_MIWAE(res_test$xhat,norm_means,norm_sds) # reverse normalization for proper comparison
    res_test$LBs_trainVal = LBs_trainVal

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
        res_train = FUN(data=np$array(datas$train),Missing=np$array(Missings$train),data_types=data_types,
                        lr=lr[k],bs=bs[j],n_epochs=n_epochs[mm],train=1L,
                        display=100L, n_save=1000L, restore=0L, dim_latent_s=dim_latent_s, dim_latent_z=dim_z[l],
                        dim_latent_y=dim_latent_y[yy], model_name=model_name,save_file=save_file)
        res_valid = FUN(data=np$array(datas$valid),Missing=np$array(Missings$valid),data_types=data_types,
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

    return(res_test)
  } else if(grepl("run_VAEAC",as.character(FUN))){
    ############################# VAEAC
    # default: h=256, n_hidden_layers=10, dim_z=64, bs=64

    norm_means[one_hot_max_sizes>=2] = 0; norm_sds[one_hot_max_sizes>=2] = 1
    n_epochs = c(200L)   # need to put in FSCseqPaper instead
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

    return(res_test)
  } else if(grepl("run_missForest",as.character(FUN)) | grepl("run_meanImputation",as.character(FUN))){
    res = FUN(data=datas$test, Missing=Missings$test)
    xhat = if(grepl("run_missForest",as.character(FUN))){res$xhat_mf}else{res$xhat_mean}
    res$xhat_rev = reverse_norm_MIWAE(xhat,norm_means,norm_sds)
    return(res)
  }
}
