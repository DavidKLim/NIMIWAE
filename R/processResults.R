# NRMSE(), process_results()
NRMSE = function(x,xhat,Missing){
  x=as.matrix(x);xhat=as.matrix(xhat);Missing=as.matrix(Missing)
  #x = (x-colMeans(x))/apply(x,2,sd)
  #xhat = (xhat-colMeans(x))/apply(x,2,sd)
  # Missing=1 --> observed

  MSE=rep(NA,ncol(x))
  RMSE=rep(NA,ncol(x))
  NRMSE=rep(NA,ncol(x))
  for(j in 1:ncol(x)){
    if(all(Missing[,j]==1)){next}
    #norm_term = (max(x[Missing[,j]==0,j])-min(x[Missing[,j]==0,j]))+0.001 # in case denom is 0
    # norm_term = (max(x[,j])-min(x[,j]))+0.001
    norm_term = sd(x[,j])
    MSE[j] = mean((x[Missing[,j]==0,j]-xhat[Missing[,j]==0,j])^2)
    RMSE[j] = sqrt(MSE[j])
    NRMSE[j] = RMSE[j]/norm_term
  }
  MSE=mean(MSE,na.rm=T); RMSE=mean(RMSE,na.rm=T); NRMSE=mean(NRMSE,na.rm=T)

  # MSE = mean((x[Missing==0]-xhat[Missing==0])^2)
  # RMSE = sqrt(MSE)
  # NRMSE = RMSE / sd(x[Missing==0])
  L1 = mean(abs(x[Missing==0]-xhat[Missing==0]))
  L2 = mean((x[Missing==0]-xhat[Missing==0])^2)
  return(list(MSE=MSE,RMSE=RMSE,NRMSE=NRMSE,L1=L1,L2=L2))
}

reverse_norm_MIWAE = function(x,norm_means,norm_sds){
  xnew=matrix(nrow=nrow(x),ncol=ncol(x))
  for(i in 1:ncol(xnew)){
    xnew[,i]=(x[,i]*(norm_sds[i]))+norm_means[i]
  }
  return(xnew)
}

# process_results() with each method. change file naming

toy_process=function(data.file.name, file.name, method=c("MIWAE","NIMIWAE","HIVAE","VAEAC","MEAN","MF")){
  call_name=match.call()

  # load data and split into training/valid/test sets
  load(data.file.name)
  datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  Missings = split(data.frame(Missing), g)
  probs_Missing = split(data.frame(prob_Missing),g)
  norm_means=colMeans(datas$train); norm_sds=apply(datas$train,2,sd)

  # MIWAE and NIMIWAE only
  load(file.name)
  print(file.name)
  fit=eval(parse(text=paste("res",method,sep="_")))

  #xhat=reverse_norm_MIWAE(fit$xhat,norm_means,norm_sds)   # already reversed
  if(method %in% c("MIWAE","NIMIWAE")){
    xhat=fit$xhat_rev
  }else if(method =="HIVAE"){
    xhat=fit$data_reconstructed
  }else if(method=="VAEAC"){
    xhat_all = fit$result    # this method reverses normalization intrinsically
    # average imputations
    xhat = matrix(nrow=nrow(datas$test),ncol=ncol(datas$test))
    n_imputations = fit$train_params$n_imputations
    for(i in 1:nrow(datas$test)){
      xhat[i,]=colMeans(xhat_all[((i-1)*n_imputations+1):(i*n_imputations),])
    }
  }else if(method=="MEAN"){
    # xhat = fit$xhat_rev
    if(is.null(fit$xhat_rev)){fit$xhat_rev = reverse_norm_MIWAE(fit$xhat_mean,norm_means,norm_sds)}
    xhat = fit$xhat_rev
  }else if(method=="MF"){
    # xhat = fit$xhat_rev
    if(is.null(fit$xhat_rev)){fit$xhat_rev = reverse_norm_MIWAE(fit$xhat_mf,norm_means,norm_sds)}
    xhat = fit$xhat_rev
  }

  # check same xhat:
  print("Mean Squared Error (Observed): should be 0")
  print(mean((xhat[Missings$test==1] - datas$test[Missings$test==1])^2))    # should be 0
  print("Mean Squared Error (Missing):")
  print(mean((xhat[Missings$test==0] - datas$test[Missings$test==0])^2))

  # Imputation metrics

  imputation_metrics=NRMSE(x=datas$test, xhat=xhat, Missing=Missings$test)
  #imputation_metrics=NRMSE(x=xfull, xhat=xhat, Missing=Missings$test)

  # Other metrics (names aren't consistent)
  #LB=fit$LB; time=fit$time

  # # Clustering (no classes right now. commented out)
  # # Average the Z across the #imputation weights L
  # z_split = split(as.data.frame(fit$zgivenx_flat),rep(1:fit$opt_params$L,each=nrow(datas$test)))
  # z_mean = Reduce(`+`, z_split) / length(z_split)
  # #PCA, etc. not coded yet

  #results = c(unlist(imputation_metrics),LB,time)
  #names(results)[(length(results)-1):length(results)]=c("LB","time")
  results = c(unlist(imputation_metrics))
  return(list(fit=fit,results=results,call=call_name))
}

output_file.name=function(dir_name,method=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"),
                          mechanism,miss_pct,betaVAE,arch,rdeponz){
  if(method=="NIMIWAE"){
    yesbeta = if(betaVAE){"beta"}else{""}; yesrdeponz = if(rdeponz){"rzT"}else{"rzF"}
    file.name=sprintf("%s/res_%s_%s_%d_%s%s_%s",
                      dir_name,method,mechanism,miss_pct,yesbeta,arch,yesrdeponz)
  }else{
    file.name=sprintf("%s/res_%s_%s_%d",
                      dir_name,method,mechanism,miss_pct) # for miwae, default = Normal (StudentT can be done later)
  }
  print(file.name)
  file.name=sprintf("%s.RData",file.name)
  return(file.name)
}

plot_diagnostics_others = function(dir_name="Results/TOYZ/phi2",mechanisms=c("MCAR","MAR","MNAR"),miss_pct=c(15,25,35),
                                   methods=c("MIWAE","HIVAE","VAEAC","MEAN","MF"),
                                   imputation_metric=c("MSE","NRMSE","L1","L2"), betaVAE=NULL, arch=NULL, rdeponz=NULL){
  # for each miss_pct, plot the MCAR -> MAR -> MNAR relationship (mechanisms)
  ## for each of the methods, using each of the imputation metrics (4 panel, 1 image per =miss_pct)
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(reshape2)
  g_legend<-function(a.gplot){
    tmp <- ggplot_gtable(ggplot_build(a.gplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    return(legend)}
  mats_res = list(); mats_params=list()
  print("Compiling results...")
  list_res = list()
  for(ii in 1:length(miss_pct)){
    params=list()
    # input true probs, no learning R, Normal distrib, input_r="r", vary rdeponz and sample_r (4)
    index=1

    for(i in 1:length(mechanisms)){for(j in 1:length(methods)){
      data.file.name=sprintf("%s/data_%s_%d.RData",dir_name,mechanisms[i],miss_pct[ii])
      file.name = output_file.name(dir_name=dir_name,method=methods[j], mechanism=mechanisms[i],
                                   miss_pct=miss_pct[ii], betaVAE=betaVAE, arch=arch, rdeponz=rdeponz)
      list_res[[index]]=toy_process(data.file.name,file.name,methods[j])
      params[[index]]=c(methods[j],mechanisms[i],miss_pct[ii])
      names(params[[index]])=c("method","mechanism","miss_pct")
      index = index+1
    }}

    # flatten list to matrix
    mat_res = matrix(unlist(lapply(list_res,function(x)x$results)),ncol=length(list_res))
    rownames(mat_res)=names(list_res[[1]]$results)     # MSE, NRMSE, ...
    colnames(mat_res)=paste("case",c(1:length(list_res)),sep="")
    mat_res=t(mat_res)

    mat_params=matrix(unlist(params),ncol=length(params))
    rownames(mat_params)=c("method","mechanism","miss_pct")
    colnames(mat_params)=paste("case",c(1:ncol(mat_params)),sep="")
    mat_params=t(mat_params)

    df_bar = data.frame(cbind(mat_res,mat_params,rownames(mat_res)))
    for(c in 1:ncol(mat_res)){df_bar[,c]=as.numeric(as.character(df_bar[,c]))}
    colnames(df_bar)[ncol(df_bar)]="case"; df_bar$case = factor(df_bar$case,levels=paste("case",c(1:nrow(mat_params)),sep=""))
    df_bar$mechanism = factor(df_bar$mechanism,levels=c("MCAR","MAR","MNAR"))

    df_bar$method=as.character(df_bar$method)
    df_bar$method[df_bar$method=="MEAN"]="Mean"; df_bar$method[df_bar$method=="MF"]="MissForest"
    other_methods = methods[methods!="NIMIWAE"]; other_methods[other_methods=="MEAN"]="Mean"; other_methods[other_methods=="MF"]="MissForest"
    df_bar$method = factor(df_bar$method,
                           levels=c(other_methods[order(other_methods)],"NIMIWAE"))
    # levels(df_bar$method)=methods[order(methods)]

    gg_color_hue <- function(n) {
      hues = seq(15, 375, length = n + 1)
      hcl(h = hues, l = 65, c = 100)[1:n]
    }
    colors = gg_color_hue(length(methods))

    p=ggplot(df_bar,aes(x=method,y=eval(parse(text=imputation_metric)),fill=mechanism,color=mechanism))+
      geom_bar(stat="identity",position=position_dodge(.9),alpha=0.4)+#ylim(c(0,3))+#ylim(c(0,0.5))+
      labs(title=sprintf("%s vs cases",imputation_metric),
           subtitle = "Imputation performance across missingness mechanisms",
           y = imputation_metric, x="Method")+
      theme(text=element_text(size = 20)) #+
    # scale_color_manual(breaks=methods[order(methods)],
    #                    values=colors[1:length(methods)])+scale_fill_manual(values=colors[1:length(methods)])


    png(sprintf("%s/%s_competing_miss%d.png",dir_name,imputation_metric,miss_pct[ii]),width=1200,height=500)
    #barplot(mat_res[rownames(mat_res)=="NRMSE",])
    print(p)
    dev.off()

    # df_bar2=data.frame(df_bar)
    # levels(df_bar2$method) = c("HIVAE","MEAN","MF","MIWAE","NIMIWAE")
    # df_bar2$miss_pct = as.numeric(as.character(df_bar2$miss_pct))
    # df_bar2$case = as.character(df_bar2$case)
    # NIMIWAE25 = c(0,0.0736714912877683,0.6252837688815652,0,"NIMIWAE","MNAR",25,"case")
    #
    # df_bar2 = rbind(df_bar2,NIMIWAE25)



    # MNAR

    # save in mats_res
    mats_res[[ii]]=mat_res
    mats_params[[ii]]=mat_params
  }
  names(mats_res)=miss_pct
  return(list(res=mats_res, params=mats_params))
}


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
