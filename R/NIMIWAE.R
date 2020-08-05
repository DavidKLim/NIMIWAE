# call this NIMIWAE.R
# toy_sim.R procedure: simulate (or read) data --> simulate missingness --> 60-20-20 split --> save --> tune_hyperparams() --> output results

split_data = function(data, ratio=c(6,2,2), seed=333){
  # splits the data into training-validation-test sets. Default = 6-2-2
  # output g is the partition that can be used to split data and Missing (and probMissing optionally).
  ratio = ratio/sum(ratio) # if ratio doesn't sum to 1
  set.seed(333)
  g = sample(cut(
    seq(nrow(data)),
    nrow(data)*cumsum(c(0,ratios)),
    labels = names(ratios)
  ))
  return(g)
}

NIMIWAE = function(data, Missing, rdeponz=F, learn_r=T, phi0=NULL, phi=NULL, ignorable=T, covars_r=rep(1,ncol(data)), arch="IWAE",
                   hyperparameters=list(sigma="elu", h=c(128L,64L), bs=c(10000L,5000L), lr=c(0.001,0.01),
                                        as.integer(c(floor(ncol(data)/2),floor(ncol(data)/4))), niw=5L, n_epochs=2002L)
                   ){
  # 1) get rid of betaVAE
  # 2) set up g= ... splits in this function
  g = split_data(data,seed=333)
  datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  Missings = split(data.frame(Missing), g)
  probs_Missing = split(data.frame(prob_Missing),g)
  phi0=fit_missing$params[[1]]$phi0; phi=fit_missing$params[[1]]$phi

  norm_means=colMeans(datas$train); norm_sds=apply(datas$train,2,sd)    # calculate normalization mean/sd on training set --> use for all

  library(reticulate)
  source_python("toy_networks.py")

  res = tuneHyperparams(FUN=run_NIMIWAE_toy_N16,data=data,Missing=Missing,g=g,
                                 rdeponz=rdeponzs[rr], learn_r=learn_r,
                                 phi0=phi0,phi=phi,
                                 covars_r=covars_r, dec_distrib=dec_distrib,
                                 arch=archs[aa], betaVAE=betaVAEs[bb], ignorable=ignorable)
}





















#### I THINK THIS SHOULD GO INTO THE Paper REPO####
toy_run=function(mechanism=c("MCAR","MAR","MNAR","MNARZ"),miss_pct=30, sim_params=NULL,
                 dataset=c("TOY","TOY2","TOY_CLUSTER"),save.folder=dataset, save.dir=".",
                 run_methods=c("MIWAE","HIVAE","VAEAC","MEAN","MF"),phi_0=25,sim_index=1
){
  library(reticulate)
  source_python("toy_networks.py")
  np = import('numpy',convert=FALSE)

  source("comparisons_missing.R")   # get read_data() and simulate_missing(), and reverse_norm_MIWAE functions

  scheme="UV"; miss_cols=NULL;ref_cols=NULL;phis=NULL;phi_z=NULL

  ## Simulate data ##
  if(!is.null(sim_params)){save.folder=sprintf("%s_n%d_p%d_K%d",save.folder,sim_params$n,sim_params$p,sim_params$K)}

  dir_name1=sprintf("%s/Results",save.dir)
  dir_name=sprintf("%s/Results/%s/phi%d/sim%d",save.dir,save.folder,phi_0,sim_index)      # this directory is where everything will be saved

  ifelse(!dir.exists(dir_name),dir.create(dir_name,recursive=T),FALSE)
  fname_data=sprintf("%s/data_%s_%d",dir_name,mechanism,miss_pct)
  if(!file.exists(sprintf("%s.RData",fname_data))){
    print("Simulating data")
    fit_data=read_data(dataset=dataset,sim_params=sim_params,sim_index=sim_index); data=fit_data$data; classes=fit_data$classes
    n=nrow(data); p=ncol(data)

    # default phi=5
    if(is.null(phis)){set.seed(222); phis=rlnorm(p,log(phi_0),0.2)}else if(length(phis)==1){phis=rep(phis,p)}
    if(is.null(phi_z)){phi_z=phis[1]/length(unique(classes))} # set dependence on class as phi/#classes

    ## Simulate Missing ##
    if(!grepl("Physionet",dataset)){
      print(sprintf("Simulating %s missingness",mechanism))
      Missing=matrix(1L,nrow=nrow(data),ncol=ncol(data))    # all observed unless otherwise specified
      if(is.null(miss_cols)){
        set.seed(111)   # random selection of anchors/missing features
        ref_cols=sample(c(1:ncol(data)),ceiling(ncol(data)/2),replace=F)    # more anchors than missing --> true miss_pct always < miss_pct
        miss_cols=(1:ncol(data))[-ref_cols]
      }
      print(paste("ref_cols:",paste(ref_cols,collapse=",")))
      print(paste("miss_cols:",paste(miss_cols,collapse=",")))

      # weight miss_pct to simulate appropriate amount of missing
      weight=p/length(miss_cols)
      miss_pct2=miss_pct*weight
      pi=1-miss_pct2/100
      fit_missing=simulate_missing(data.matrix(data),miss_cols,ref_cols,pi,
                                   phis,phi_z,
                                   scheme,mechanism=mechanism)
      Missing=fit_missing$Missing; prob_Missing=fit_missing$probs      # missing mask, probability of each observation being missing
    } else{
      library(reticulate)
      np <- import("numpy")
      npz1 <- np$load("data/PhysioNet2012/physionet.npz")

      if(strsplit(dataset,"_")[[1]][2] == "mean"){
        # dataset=="Physionet_mean"
        Missing = 1 - floor(rbind(apply(npz1$f$m_train_miss, c(1,3), mean),
                                  apply(npz1$f$m_val_miss, c(1,3), mean),
                                  apply(npz1$f$m_test_miss, c(1,3), mean)))
      } else if(strsplit(dataset,"_")[[1]][2] == "all"){
        # dataset=="Physionet_all"
        library(abind)
        M3D = aperm(abind(npz1$f$m_train_miss,
                          npz1$f$m_val_miss,
                          npz1$f$m_test_miss,
                          along=1),
                    c(2,1,3))                                             # switch dims so 48 time points is first dim
        Missing = 1 - matrix(M3D, nrow=dim(M3D)[1]*dim(M3D)[2], ncol=dim(M3D)[3])   # stack time series data: 1st subject is 1st - 48th observations, 2nd subj is 49th - 96th, ...
      }
      fit_missing=NULL; prob_Missing=Missing; ref_cols=NULL; miss_cols=NULL
    }

    ## Create Validation/Training/Test splits here. default: 60%-20%-20% ##
    ratios=c(train = .6, test = .2, valid = .2)

    set.seed(333)
    g = sample(cut(
      seq(nrow(data)),
      nrow(data)*cumsum(c(0,ratios)),
      labels = names(ratios)
    ))
    save(list=c("data","Missing","fit_data","fit_missing","prob_Missing","g","ref_cols","miss_cols"),file=sprintf("%s.RData",fname_data))
  }else{
    print("Loading previously simulated data")
    load(sprintf("%s.RData",fname_data))
  }
  library(ggplot2)
  overlap_hists=function(x1,x2,x3=NULL,lab1="Truth",lab2="Imputed",lab3="...",
                         title="MNAR Missing Values, Truth vs Imputed, Missing column"){
    library(ggplot2)
    x1=data.frame(value=x1); x1$status=lab1
    x2=data.frame(value=x2); x2$status=lab2
    if(!is.null(x3)){x3=data.frame(value=x3); x3$status=lab3; df=rbind(x1,x2,x3)
    }else{df = rbind(x1,x2)}
    p = ggplot(df,aes(value,fill=status)) + geom_density(alpha=0.2, adjust=1/5) + ggtitle(title) + xlim(quantile(df$value,c(0.01,0.99),na.rm=T))
    print(p)
  }

  diag_dir_name = sprintf("%s/Diagnostics/miss%d",dir_name,miss_pct)
  ifelse(!dir.exists(diag_dir_name),dir.create(diag_dir_name,recursive=T),F)
  for(c in miss_cols){
    png(sprintf("%s/%s_col%d_Truth_allData.png",diag_dir_name,mechanism,c))
    p = ggplot(data.frame(value=data[,c]),aes(value)) + geom_density(alpha=0.2) + ggtitle(sprintf("Column %d: Density plot of all observations",c))
    print(p)
    dev.off()
    png(sprintf("%s/%s_col%d_Truth_MissVsObs.png",diag_dir_name,mechanism,c))
    overlap_hists(x1=data[Missing[,c]==0,c],lab1="Missing",
                  x2=data[Missing[,c]==1,c],lab2="Observed",
                  title=sprintf("Column %d: Density plot of Missing vs Observed Observations",c))
    dev.off()
  }
#
#   datas = split(data.frame(data), g)        # split by $train, $test, and $valid
#   Missings = split(data.frame(Missing), g)
#   probs_Missing = split(data.frame(prob_Missing),g)
#
#   phi0=fit_missing$params[[1]]$phi0; phi=fit_missing$params[[1]]$phi
#
#
#   norm_means=colMeans(datas$train); norm_sds=apply(datas$train,2,sd)    # calculate normalization mean/sd on training set --> use for all
#
#   #batches used just for training so i think it doesn't matter for test.
#   test_bs = 10000000L  # for miwae/nimiwae: number of observations to mini-batch in testing. should be as large as possible that can fit on 8GB GPU
#
#   ######## TOY NIMIWAE FUNCTION HERE ###########
#   # need to create 2 toy functions for nimiwae:
#   ## for both::: feed in missingness mask along with data to encoder
#   # 1) input two features --> 1 hidden layer/1 hidden dimension w 8 nodes? -->
#   # 2) input two features & input phis for p(R|X) --> same -->
#   source("tuneHyperparams.R")
#

  #if(dataset %in% c("TOYZ","TOYZa","TOYZ2","TOYZ2a")){
  if(TRUE){
    #### NIMIWAE ####
    print("NIMIWAE")
    if("NIMIWAE" %in% run_methods){
      # Fixed:
      dec_distrib="Normal"; learn_r=T

      covars_r=rep(1,ncol(data)); ignorable=F  # all as covariates
      # if(mechanism=="MCAR"){ignorable=T; covars_r = rep(0,ncol(data)); print(covars_r)
      # }else if(mechanism=="MAR"){ignorable=F; covars_r = rep(0,ncol(data)); covars_r[ref_cols] = 1; print(covars_r)  # ignorable, but testing nonignorable with MAR --> should still be fine
      # } else if(mechanism=="MNAR"){ignorable=F; covars_r = rep(0,ncol(data)); covars_r[miss_cols] = 1; print(covars_r)}

      # Variants:
      # if(dataset%in%c("TOYZ","TOYZ2")){rdeponzs = c(F,T); archs = c("IWAE","VAE"); betaVAEs = c(F,T)
      # }else{
      rdeponzs = c(F); archs = c("IWAE"); betaVAEs = c(F)
      # }

      # for arch="VAE" --> niw = L = M = 1. for "IWAE" --> niw = L = M = 5
      n_variants = length(rdeponzs)*length(archs)*length(betaVAEs)

      for(rr in 1:length(rdeponzs)){for(aa in 1:length(archs)){for(bb in 1:length(betaVAEs)){
        if(betaVAEs[bb] & archs[aa]=="IWAE"){next}   # no betaIWAE. only betaVAE
        dir_name2=sprintf("%s",dir_name)
        ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
        yesbeta=if(betaVAEs[bb]){"beta"}else{""}; yesrz = if(rdeponzs[rr]){"T"}else{"F"}
        fname0=sprintf("%s/res_NIMIWAE_%s_%d_%s%s_rz%s.RData",dir_name2,mechanism,miss_pct,yesbeta,archs[aa],yesrz)
        print(fname0)
        if(!file.exists(fname0)){
          t0=Sys.time()
          res_NIMIWAE = tune_hyperparams(FUN=run_NIMIWAE_toy_N16,data=data,Missing=Missing,g=g,
                                         rdeponz=rdeponzs[rr], learn_r=learn_r,
                                         phi0=phi0,phi=phi,
                                         covars_r=covars_r, dec_distrib=dec_distrib,
                                         arch=archs[aa], betaVAE=betaVAEs[bb], ignorable=ignorable)

          res_NIMIWAE$time = as.numeric(Sys.time()-t0,units="secs")
          print(paste("Time elapsed: ", res_NIMIWAE$time, "s."))
          # nMB = length(res_NIMIWAE$xhat_fits)
          # for(ii in 1:nMB){
          #   if(ii==1){
          #     res_NIMIWAE$zgivenx_flat=res_NIMIWAE$xhat_fits[[ii]]$xhat_fit$zgivenx_flat; res_NIMIWAE$imp_weights = res_NIMIWAE$xhat_fits[[ii]]$xhat_fit$imp_weights
          #     next
          #   }
          #   res_NIMIWAE$zgivenx_flat = rbind(res_NIMIWAE$zgivenx_flat, res_NIMIWAE$xhat_fits[[ii]]$xhat_fit$zgivenx_flat)
          #   res_NIMIWAE$imp_weights = cbind(res_NIMIWAE$imp_weights, res_NIMIWAE$xhat_fits[[ii]]$xhat_fit$imp_weights)
          # }
          # res_NIMIWAE$xhat_fits=NULL
          save(res_NIMIWAE,file=sprintf("%s",fname0))
        }else{
          load(sprintf("%s",fname0))
        }
        for(c in miss_cols){
          png(sprintf("%s/%s_col%d_NIMIWAE_%s%s_rz%s.png",diag_dir_name,mechanism,c,yesbeta,archs[aa],yesrz))
          overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
                        x2=res_NIMIWAE$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                        x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                        title=sprintf("NIMIWAE Column%d: True vs Imputed missing and observed values",c))
          dev.off()
        }
        rm("res_NIMIWAE")
      }}}
    }


    #### MIWAE  (baseline comparison) ####
    print("MIWAE")
    if("MIWAE" %in% run_methods){
      rdeponz=FALSE;  covars_r=NULL; dec_distrib="Normal"; learn_r=NULL
      dir_name2=sprintf("%s",dir_name)
      ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
      fname0=sprintf("%s/res_MIWAE_%s_%d.RData",dir_name2,mechanism,miss_pct)
      print(fname0)
      if(!file.exists(fname0)){
        t0=Sys.time()
        res_MIWAE = tune_hyperparams(FUN=run_MIWAE,data=data,Missing=Missing,g=g,
                                     rdeponz=rdeponz, learn_r=learn_r,
                                     phi0=phi0,phi=phi,
                                     covars_r=covars_r, dec_distrib=dec_distrib)
        res_MIWAE$time=as.numeric(Sys.time()-t0,units="secs")
        save(res_MIWAE,file=sprintf("%s",fname0))
        #rm("res_MIWAE")
      }else{
        load(fname0)
        for(c in miss_cols){
          png(sprintf("%s/%s_col%d_MIWAE.png",diag_dir_name,mechanism,c))
          overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
                        x2=res_MIWAE$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                        x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                        title=sprintf("MIWAE Column%d: True vs Imputed missing and observed values",c))
          dev.off()
        }
      }
      rm("res_MIWAE")
    }

    #### SETUP: Other methods ####
    data_types=list()
    one_hot_max_sizes=rep(NA,ncol(data))
    for(i in 1:ncol(data)){
      # factors/ordinal --> just treat as categorical to automatize
      if(is.character(data[,i]) | is.factor(data[,i])){
        nclass=as.character('length(unique(data[,i]))')
        data_types[i]=list(type='cat',dim=nclass,nclass=nclass)

        one_hot_max_sizes[i]=as.integer(nclass)
      }
      # numeric (real/pos/count)
      if(is.numeric(data[,i])){
        # positive
        if(all(data[,i]>=0)){
          # count (count is positive)
          if(all(data[,i]==round(data[,i],0))){
            data_types[[i]]=list(type='count',dim='1',nclass='')
          } else{
            data_types[[i]]=list(type='pos',dim='1',nclass='')
          }
        } else{
          data_types[[i]]=list(type='real',dim='1',nclass='')
        }

        one_hot_max_sizes[i]=1L
      }
    }

    # for VAEAC
    MissingData = data
    MissingData[Missing==0]=NaN

    MissingDatas = split(data.frame(MissingData),g)
    source_python("comparisons_missing.py")

    ### HIVAE ###
    print("HIVAE")
    if("HIVAE" %in% run_methods){
      dir_name2=sprintf("%s",dir_name)
      ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
      fname0=sprintf("%s/res_HIVAE_%s_%d.RData",dir_name2,mechanism,miss_pct)
      print(fname0)
      if(!file.exists(fname0)){
        t0=Sys.time()
        res_HIVAE = tune_hyperparams(FUN=run_HIVAE,data=data,Missing=Missing,g=g,
                                     data_types=data_types)
        res_HIVAE$time=as.numeric(Sys.time()-t0,units="secs")
        save(res_HIVAE,file=sprintf("%s",fname0))
        #rm("res_HIVAE")
      }else{
        load(fname0)
        for(c in miss_cols){
          png(sprintf("%s/%s_col%d_HIVAE.png",diag_dir_name,mechanism,c))
          overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
                        x2=res_HIVAE$data_reconstructed[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                        x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                        title=sprintf("HIVAE Column%d: True vs Imputed missing and observed values",c))
          dev.off()
        }
      }
      rm("res_HIVAE")
    }

    ### VAEAC ###
    print("VAEAC")
    if("VAEAC" %in% run_methods){
      dir_name2=sprintf("%s",dir_name)
      ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
      fname0=sprintf("%s/res_VAEAC_%s_%d.RData",dir_name2,mechanism,miss_pct)
      print(fname0)
      # n_hidden_layers = if(dataset%in% c("TOYZ","TOYZ2","BANKNOTE","IRIS","WINE","BREAST","YEAST","CONCRETE","SPAM","ADULT","GAS","POWER")){10L} # default
      if(!file.exists(fname0)){
        t0=Sys.time()
        res_VAEAC = tune_hyperparams(FUN=run_VAEAC,data=data,Missing=Missing,g=g,
                                     one_hot_max_sizes=one_hot_max_sizes, MissingDatas=MissingDatas)
        res_VAEAC$time=as.numeric(Sys.time()-t0,units="secs")
        save(res_VAEAC,file=sprintf("%s",fname0))
        #rm("res_VAEAC")
      }else{
        load(fname0)
        xhat_all = res_VAEAC$result    # this method reverses normalization intrinsically
        # average imputations
        xhat = matrix(nrow=nrow(datas$test),ncol=ncol(datas$test))
        n_imputations = res_VAEAC$train_params$n_imputations
        for(i in 1:nrow(datas$test)){ xhat[i,]=colMeans(xhat_all[((i-1)*n_imputations+1):(i*n_imputations),]) }

        for(c in miss_cols){
          png(sprintf("%s/%s_col%d_VAEAC.png",diag_dir_name,mechanism,c))
          overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
                        x2=xhat[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                        x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                        title=sprintf("VAEAC Column%d: True vs Imputed missing and observed values",c))
          dev.off()
        }
      }
      rm("res_VAEAC")
    }

    ### MF ###
    print("MF")
    if("MF" %in% run_methods){
      dir_name2=sprintf("%s",dir_name)
      ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
      fname0=sprintf("%s/res_MF_%s_%d.RData",dir_name2,mechanism,miss_pct)
      print(fname0)
      if(!file.exists(fname0)){
        t0=Sys.time()
        res_MF = tune_hyperparams(FUN=run_missForest,data=data,Missing=Missing,g=g)
        res_MF$time=as.numeric(Sys.time()-t0,units="secs")
        save(res_MF,file=sprintf("%s",fname0))
        #rm("res_MF")
      }else{
        load(fname0)
        if(is.null(res_MF$xhat_rev)){res_MF$xhat_rev = reverse_norm_MIWAE(res_MF$xhat_mf,norm_means,norm_sds)}
        for(c in miss_cols){
          png(sprintf("%s/%s_col%d_MF.png",diag_dir_name,mechanism,c))
          overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing",
                        x2=res_MF$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                        x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                        title=sprintf("MF Column%d: True vs Imputed missing and observed values",c))
          dev.off()
        }
      }
      rm("res_MF")
    }

    ### MEAN ###
    print("MEAN")
    if("MEAN" %in% run_methods){
      dir_name2=sprintf("%s",dir_name)
      ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
      fname0=sprintf("%s/res_MEAN_%s_%d.RData",dir_name2,mechanism,miss_pct)
      print(fname0)
      if(!file.exists(fname0)){
        t0=Sys.time()
        res_MEAN = tune_hyperparams(FUN=run_meanImputation,data=data,Missing=Missing,g=g)
        res_MEAN$time=as.numeric(Sys.time()-t0,units="secs")
        save(res_MEAN,file=sprintf("%s",fname0))
        #rm("res_MEAN")
      }else{
        load(fname0)
        if(is.null(res_MEAN$xhat_rev)){res_MEAN$xhat_rev = reverse_norm_MIWAE(res_MEAN$xhat_mean,norm_means,norm_sds)}
        for(c in miss_cols){
          png(sprintf("%s/%s_col%d_MEAN.png",diag_dir_name,mechanism,c))
          overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
                        x2=res_MEAN$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                        x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                        title=sprintf("MEAN Column%d: True vs Imputed missing and observed values",c))
          dev.off()
        }
      }
      rm("res_MEAN")
    }
  }

}


