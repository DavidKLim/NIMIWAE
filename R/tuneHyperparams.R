## NTD:
# 1) take out all "run_"
# 2) make sigma, hs, bss, lrs, dim_zs, niws, n_epochs customizable? (kinda is, but create defaults and take out hard coding --> put in Paper repo)


tuneHyperparams = function(FUN=c(run_NIMIWAE_toy_N6, run_MIWAE, run_HIVAE, ),
                            data,Missing,g,
                            rdeponz=F,
                            learn_r=T,phi0=NULL,phi=NULL,
                            covars_r=rep(1,ncol(data)), dec_distrib=c("Normal","StudentT"),
                            sigma="elu", hs=c(16L,8L), bss=c(10000L,5000L), lrs=c(0.001,0.01), dim_zs=c(2L,4L), niws=5L,n_epochss=2002L,
                            arch="IWAE", betaVAE=T,   # for NIMIWAE: whether each NN is optimized separately, architecture: VAE or IWAE
                            test_bs=10000000L, data_types=NULL, one_hot_max_sizes=NULL, ohms=NULL, n_hidden_layers=c(1L,2L),
                            MissingDatas = NULL, ignorable=F # just for vaeac
){
  p = ncol(data)
  if(dataset%in%c("TOYZ","TOYZa")){sigma="elu"; hs=c(4L,8L); bss=c(5000L); lrs=c(0.001,0.01); dim_zs=c(1L,2L); niws=5L; n_epochss=2002L}
  if(dataset%in%c("TOYZ2","TOYZ2a")){sigma="elu"; hs=c(16L,8L); bss=c(10000L); lrs=c(0.001,0.01); dim_zs=c(4L,2L); niws=5L; n_epochss=2002L}
  if(dataset=="TOYZ50"){sigma="elu"; hs=c(128L,64L); bss=c(10000L); lrs=c(0.001,0.01); dim_zs=c(8L,4L); niws=5L; n_epochss=2002L}
  if(dataset=="TOYZ_CLUSTER"){sigma="elu"; hs=c(128L,64L); bss=c(10000L); lrs=c(0.001,0.01); dim_zs=c(8L,4L); niws=5L; n_epochss=2002L}
  if(dataset%in%c("BANKNOTE","WINE","BREAST","YEAST","CONCRETE","SPAM","ADULT","RED")){
    sigma="elu"; hs=c(128L,64L); bss=c(200L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  if(dataset %in% c("SPAM","ADULT","WHITE")){
    sigma="elu"; hs=c(128L,64L); bss=c(1000L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  if(dataset%in% c("GAS","POWER","HEPMASS","MINIBOONE")){
    sigma="elu"; hs=c(128L,64L); bss=c(20000L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  if(dataset=="IRIS"){
    sigma="elu"; hs=c(128L,64L); bss=c(20L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  if(dataset=="Physionet_mean"){
    sigma="elu"; hs=c(128L,64L); bss=c(300L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  if(dataset=="Physionet_all"){
    sigma="elu"; hs=c(128L,64L); bss=c(5000L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  input_prob_Missing=F
  datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  Missings = split(data.frame(Missing), g)
  probs_Missing = split(data.frame(prob_Missing),g)
  norm_means=colMeans(datas$train); norm_sds=apply(datas$train,2,sd)
  test_epochs=2L

  torch = import("torch")

  if(grepl("run_NIMIWAE",as.character(FUN))){
    if(learn_r){phi0=NULL; phi=NULL}else{phi=np$array(phi)}
    list_train = list()

    include_xo=TRUE
    #partial_opt=FALSE; nits=1L; nGibbs=0L; input_r="r"
    add_miss_term = F; draw_xobs=F; draw_xmiss=T; pre_impute_value=0L
    if(betaVAE){ beta=0; beta_anneal_rate=1/500
    }else{ beta=1; beta_anneal_rate=0 }
    if(arch=="VAE"){ niws=1L }
    #M=1L

    # sparse="dropout"; dropout_pct=50; L1_weights=0; L2_weight=0
    # sparse="L1"; L1_weights=c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8); L2_weight=0; dropout_pct=NULL  # D2
    # sparse="L1"; L1_weights=c(0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5); L2_weight=0; dropout_pct=NULL  # D3
    # sparse="L1"; L1_weights=c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5); L2_weight=0; dropout_pct=NULL  # D4
    sparse="L1"; L1_weights=c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5); L2_weight=4e-3; dropout_pct=NULL  # D5: Weight decay of 4e-3: https://dejanbatanjac.github.io/2019/07/02/Impact-of-WD.html, https://arxiv.org/pdf/1803.09820.pdf
    dim_zs = c(64L,128L)    # D6: way overparametrized Z

    # sparse="none"; dropout_pct=NULL; L1_weights=0; L2_weight=0
    n_combs_params=length(hs)*length(bss)*length(lrs)*length(dim_zs)*length(niws)*length(n_epochss)*length(n_epochss)*length(n_hidden_layers)*length(L1_weights)
    LBs_trainVal = matrix(NA,nrow=n_combs_params,ncol=8+4)   # contain params, trainMSE,valMSE,trainLB,valLB
    colnames(LBs_trainVal) = c("h","bs","lr","dim_z","niw","n_epoch","n_hidden_layers","L1_weights",
                               "LB_train","MSE_train","LB_valid","MSE_valid")
    index=1
    for(i in 1:length(hs)){for(j in 1:length(bss)){for(k in 1:length(lrs)){for(l in 1:length(dim_zs)){
      for(m in 1:length(niws)){for(mm in 1:length(n_epochss)){for(nn in 1:length(n_hidden_layers)){for(oo in 1:length(L1_weights)){        # h1: encoder     q(z|xo (,r))
        # h2: decoder_x   p(x|z)
        # h3: decoder_r   p(r|x (,z))
        # h4: decoder_xr  p(x|z,r)

        #rdeponz,input_r,data,Missing,probMissing,
        #covars_r,norm_means,norm_sds,learn_r,
        #L1_weight=0,L2_weight=0,unnorm=False,sparse="none",dropout_pct=None,prune_pct=None,covars_miss=None,impute_bs=None, (defaults: no sparsity, and don't unnormalize for Decoder_R)
        #include_xo=False,partial_opt=False,arch="IWAE",nits=1,nGibbs=5,add_miss_term=False,draw_xobs=True,draw_xmiss=True,pre_impute_value=0,h1=64,h2=None,h3=None,h4=None,beta=0,beta_anneal_rate=1/500,phi0=None,phi=None,dec_distrib="Normal",train=1,saved_model=None,sigma="relu",bs = 64,n_epochs = 2002,lr=0.001,niw=20,dim_z=5,L=20,M=20,trace=False):


        print(paste("h:",hs[i],", bs:",bss[j],", lr:",lrs[k],", dim_z:",dim_zs[l],", niw:",niws[m],", n_epochs: ",n_epochss[mm],", n_hls: ",n_hidden_layers[nn],", L1_weight: ", L1_weights[oo], sep=""))

        impute_bs = bss[j] # batch_size in imputation same as batch_size in training

        if(oo==1){warm_started_model = NULL; warm_start=F}else{warm_started_model = res_train$'saved_model'; warm_start=T}
        # fix h3=0: Logistic Regression for p(r|x)
        res_train = FUN(rdeponz=rdeponz, data=np$array(datas$train),data_val=np$array(datas$valid),Missing=np$array(Missings$train),Missing_val=np$array(Missings$valid),probMissing=np$array(probs_Missing$train),
                        covars_r=np$array(covars_r), norm_means=np$array(norm_means), norm_sds=np$array(norm_sds), learn_r=learn_r,
                        ignorable=ignorable,n_hidden_layers=n_hidden_layers[nn], n_hidden_layers_r=0L,
                        L1_weight=L1_weights[oo],L2_weight=L2_weight,unnorm=F,sparse=sparse,dropout_pct=dropout_pct,prune_pct=NULL,covars_miss=NULL,covars_miss_val=NULL,impute_bs=impute_bs,include_xo=include_xo,
                        arch=arch,
                        add_miss_term=add_miss_term,draw_xobs=draw_xobs,draw_xmiss=draw_xmiss,
                        pre_impute_value=pre_impute_value,h1=hs[i],h2=hs[i],h3=0,h4=hs[i],beta=beta,beta_anneal_rate=beta_anneal_rate,
                        phi0=phi0, phi=phi, warm_start=warm_start, saved_model=warm_started_model, dec_distrib=dec_distrib, train=1L,
                        sigma=sigma, bs = bss[j], n_epochs = n_epochss[mm], lr=lrs[k], niw=niws[m], dim_z=dim_zs[l], L=niws[m], M=niws[m])

        res_valid = FUN(rdeponz=rdeponz, data=np$array(datas$valid),data_val=np$array(datas$valid),Missing=np$array(Missings$valid),Missing_val=np$array(Missings$valid),probMissing=np$array(probs_Missing$valid),
                        covars_r=np$array(covars_r), norm_means=np$array(norm_means), norm_sds=np$array(norm_sds), learn_r=learn_r,
                        ignorable=ignorable,n_hidden_layers=n_hidden_layers[nn], n_hidden_layers_r=0L,
                        L1_weight=L1_weights[oo],L2_weight=L2_weight,unnorm=F,sparse=sparse,dropout_pct=dropout_pct,prune_pct=NULL,covars_miss=NULL,covars_miss_val=NULL,impute_bs=impute_bs,include_xo=include_xo,
                        arch=arch,
                        add_miss_term=add_miss_term,draw_xobs=draw_xobs,draw_xmiss=draw_xmiss,
                        pre_impute_value=pre_impute_value,h1=hs[i],h2=hs[i],h3=0,h4=hs[i],beta=1,beta_anneal_rate=0,
                        phi0=phi0, phi=phi, warm_start=F, saved_model=res_train$'saved_model', dec_distrib=dec_distrib, train=0L,
                        sigma=sigma, bs = bss[j], n_epochs=test_epochs, lr=lrs[k], niw=niws[m], dim_z=dim_zs[l], L=niws[m], M=niws[m])
        ## batching testing too: bs != test_bs
        #list_train[[index]] = res_train
        #LBs[index]=res_valid$'LB'
        print(c(hs[i],bss[j],lrs[k],dim_zs[l],niws[m],res_train$train_params$early_stop_epochs,
                n_hidden_layers[nn],L1_weights[oo],
                res_train$'LB',res_train$'MSE'$miss[length(res_train$'MSE'$miss)],res_valid$'LB',res_valid$'MSE'$miss[length(res_valid$'MSE'$miss)]))
        LBs_trainVal[index,]=c(hs[i],bss[j],lrs[k],dim_zs[l],niws[m],res_train$train_params$early_stop_epochs,
                               n_hidden_layers[nn],L1_weights[oo],
                               res_train$'LB',res_train$'MSE'$miss[length(res_train$'MSE'$miss)],res_valid$'LB',res_valid$'MSE'$miss[length(res_valid$'MSE'$miss)])

        print(LBs_trainVal)
        if(is.na(res_valid$'LB')){res_valid$'LB'=-Inf}

        # save only the best result currently (not all results) --> save memory
        if(index==1){opt_train = res_train; opt_LB = res_valid$'LB'; save(opt_train,file="temp_opt_train.out"); torch$save(opt_train$'saved_model',"temp_opt_train_saved_model.pth")  #; save(opt_train, file="temp_opt_train.out")
        }else if(res_valid$'LB' > opt_LB){opt_train=res_train; opt_LB = res_valid$'LB'; save(opt_train,file="temp_opt_train.out"); torch$save(opt_train$'saved_model',"temp_opt_train_saved_model.pth")} #; save(opt_train, file="temp_opt_train.out")

        rm(opt_train)
        #rm(res_train); rm(res_valid)
        index=index+1
      }}}}}}}}
    #opt_id=which.min(LBs)
    #opt_params=list_train[[opt_id]]$'train_params'
    print("Hyperparameter tuning complete.")

    load("temp_opt_train.out")
    saved_model = torch$load("temp_opt_train_saved_model.pth")

    opt_params = opt_train$'train_params' #; saved_model = opt_train$'saved_model'
    res_test = FUN(rdeponz=rdeponz, data=np$array(datas$test),data_val=np$array(datas$valid),Missing=np$array(Missings$test),Missing_val=np$array(Missings$valid),probMissing=np$array(probs_Missing$test),
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

    n_combs_params=length(hs)*length(bss)*length(lrs)*length(dim_zs)*length(niws)*length(n_epochss)*length(n_hidden_layers)
    list_train = list()
    LBs = rep(NA,n_combs_params)
    index=1
    # loop train --> valid
    for(i in 1:length(hs)){for(j in 1:length(bss)){for(k in 1:length(lrs)){for(l in 1:length(dim_zs)){
      for(m in 1:length(niws)){for(mm in 1:length(n_epochss)){for(nn in 1:length(n_hidden_layers)){

        print(paste("h:",hs[i],", bs:",bss[j],", lr:",lrs[k],", dim_z:",dim_zs[l],", niw:",niws[m],", n_epochs: ",n_epochss[mm],", n_hls: ",n_hidden_layers[nn],sep=""))
        res_train = FUN(data=np$array(datas$train),Missing=np$array(Missings$train),
                        norm_means=np$array(norm_means),norm_sds=np$array(norm_sds),
                        n_hidden_layers=n_hidden_layers[nn],
                        dec_distrib=dec_distrib, train=1L, h=hs[i],
                        sigma=sigma,bs = bss[j],
                        n_epochs = n_epochss[mm],lr=lrs[k],niw=niws[m],dim_z=dim_zs[l],L=niws[m])

        res_valid = FUN(data=np$array(datas$valid),Missing=np$array(Missings$valid),
                        norm_means=np$array(norm_means),norm_sds=np$array(norm_sds),
                        n_hidden_layers=n_hidden_layers[nn],
                        dec_distrib=dec_distrib, train=0L, saved_model=res_train$'saved_model', h=hs[i],
                        sigma=sigma,bs = bss[j],
                        n_epochs = test_epochs,lr=lrs[k],niw=niws[m],dim_z=dim_zs[l],L=niws[m])

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
      bss = 150000L
    }else if(dataset%in%c("GAS","POWER","MINIBOONE")){
      bss = 500000L
    }
    ############################ HIVAE
    dim_latent_ys = c(5L,10L)
    n_combs_params=length(bss)*length(lrs)*length(dim_zs)*length(n_epochss)*length(dim_latent_ys)

    list_train = list()
    LBs = rep(NA,n_combs_params)
    index=1
    dim_latent_s = 10L #; dim_latent_y=10L # default
    model_name = "model_HIVAE_inputDropout"

    for(i in 1:length(hs)){for(j in 1:length(bss)){for(k in 1:length(lrs)){for(l in 1:length(dim_zs)){
      for(mm in 1:length(n_epochss)){for(yy in 1:length(dim_latent_ys)){
        print(paste("h:",hs[i],", bs:",bss[j],", lr:",lrs[k],", dim_z:",dim_zs[l],", n_epochs: ",n_epochss[mm], ", dim_latent_y: ", dim_latent_ys[yy],sep=""))
        save_file = sprintf("HIVAE_interm_%f_%d_%d_%d_%d_%d_%s",lrs[k],bss[j],n_epochss[mm],dim_latent_s,dim_zs[l],dim_latent_ys[yy],model_name)
        res_train = FUN(data=np$array(datas$train),Missing=np$array(Missings$train),data_types=data_types,
                        lr=lrs[k],bs=bss[j],n_epochs=n_epochss[mm],train=1L,
                        display=100L, n_save=1000L, restore=0L, dim_latent_s=dim_latent_s, dim_latent_z=dim_zs[l],
                        dim_latent_y=dim_latent_ys[yy], model_name=model_name,save_file=save_file)
        res_valid = FUN(data=np$array(datas$valid),Missing=np$array(Missings$valid),data_types=data_types,
                        lr=lrs[k],bs=10000000L,n_epochs=1L,train=0L,
                        display=100L, n_save=1000L, restore=1L, dim_latent_s=dim_latent_s, dim_latent_z=dim_zs[l],
                        dim_latent_y=dim_latent_ys[yy], model_name=model_name,save_file=save_file)
        #list_train[[index]] = res_train
        #LBs[index]=res_valid$'mean_loss'

        if(is.na(res_valid$'mean_loss')){res_valid$'mean_loss'=-Inf}

        if(index==1){opt_train = res_train; opt_LB = res_valid$'mean_loss'
        }else if(res_valid$'mean_loss' > opt_LB){opt_train=res_train; opt_LB = res_valid$'mean_loss'
        }
        index=index+1
      }}}}}}
    print("Hyperparameter tuning complete.")


    ############# LOADING WRONG MODEL HOW TO LOAD CORRECT MODEL. CHANGE SAVE FILE.
    #opt_id=which.min(LBs)
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
    # n_epochss = c(100L,200L)
    n_epochss = c(200L)
    n_combs_params=length(hs)*length(bss)*length(lrs)*length(dim_zs)*length(n_epochss)*length(n_hidden_layers)

    list_train = list()
    LBs = rep(NA,n_combs_params)
    n_imputations=5L; validation_ratio=0.2; validations_per_epoch=1L; validation_iwae_n_samples=25L  # ; n_hidden_layers=10L   # put this in the tune_hyperparameters() function
    index=1

    for(i in 1:length(hs)){for(j in 1:length(bss)){for(k in 1:length(lrs)){for(l in 1:length(dim_zs)){for(nn in 1:length(n_hidden_layers)){
      for(mm in 1:length(n_epochss)){for(nn in 1:length(n_hidden_layers)){
        print(paste("h:",hs[i],", bs:",bss[j],", lr:",lrs[k],", dim_z:",dim_zs[l],", n_epochs: ",n_epochss[mm], ", n_hls: ", n_hidden_layers[nn],sep=""))
        save_file = sprintf("VAEAC_interm_%d_%d_%f_%d_%d_%d_%d_%d_%f_%d",
                            hs[i],n_hidden_layers[nn],lrs[k],bss[j],n_epochss[mm],dim_zs[l],
                            n_imputations,validations_per_epoch,validation_ratio,validation_iwae_n_samples)
        res_train = FUN(data=np$array(rbind(MissingDatas$train,MissingDatas$valid)),one_hot_max_sizes=one_hot_max_sizes,
                        norm_mean=np$array(norm_means),norm_std=np$array(norm_sds),
                        h=hs[i], n_hidden_layers=n_hidden_layers[nn], dim_z=dim_zs[l],bs=bss[j],lr=lrs[k],output_file=save_file,
                        train=1L,saved_model=NULL,saved_networks=NULL,
                        n_epochs=n_epochss[mm],n_imputations=n_imputations,validation_ratio=validation_ratio,
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
