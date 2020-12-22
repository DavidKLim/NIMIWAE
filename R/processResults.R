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

#' Process results: return imputation metrics
#'
#' @param data Data matrix (N x P)
#' @param Missing Missingness mask matrix (N x P)
#' @param g Training-validation-test split partitioning
#' @param res Results object output from NIMIWAE function
#' @param data.file.name Path to data file, which contains "data", "Missing", and "g". These inputs need not be specified if data.file.name is specified
#' @param res.file.name Path to res file, which contains "res_<method>" results object from the method that was run. "res" need not be specified if res.file.name is specified
#' @param method Method used for imputation. "NIMIWAE" is used for this package, but results from competing methods ("MIWAE", "HIVAE", "VAEAC", "MEAN", "MF") can also be processed. See the NIMIWAE_Paper repo for more details
#' @return list of objects: res (original res input), results (named vector of imputation metrics), and call (user-input call to function)
#' @examples
#' processResults(data.file.name="Results/CONCRETE/data_MCAR_25.RData", res.file.name="Results/CONCRETE/res_NIMIWAE_MCAR_25_IWAE_rzF.RData", method="NIMIWAE")
#' @export
processResults=function(data=NULL, Missing=NULL, g=NULL, res=NULL,
                        data.file.name="", res.file.name="", method=c("MIWAE","NIMIWAE","HIVAE","VAEAC","MEAN","MF")){
  call_name=match.call()

  # load data and split into training/valid/test sets
  if(is.null(data)){ load(data.file.name) }     # if data is not specified, will look for data.file.name (which contains data, Missing, and g)
  datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  Missings = split(data.frame(Missing), g)
  norm_means=colMeans(datas$train); norm_sds=apply(datas$train,2,sd)

  # if res object is input, then use this. If no res object, look for file name from which to obtain res. res object should be saved as "res_<method>"
  if(is.null(res)){
    load(res.file.name)
    print(res.file.name)
    res=eval(parse(text=paste("res",method,sep="_")))
  }

  #xhat=reverse_norm_MIWAE(res$xhat,norm_means,norm_sds)   # already reversed
  if(method %in% c("MIWAE","NIMIWAE")){
    xhat=res$xhat_rev
  }else if(method =="HIVAE"){
    xhat=res$data_reconstructed
  }else if(method=="VAEAC"){
    xhat_all = res$result
    # average imputations
    xhat = matrix(nrow=nrow(datas$test),ncol=ncol(datas$test))
    n_imputations = res$train_params$n_imputations
    for(i in 1:nrow(datas$test)){
      xhat[i,]=colMeans(xhat_all[((i-1)*n_imputations+1):(i*n_imputations),])
    }
  }else if(method=="MEAN"){
    # xhat = res$xhat_rev
    if(is.null(res$xhat_rev)){res$xhat_rev = reverse_norm_MIWAE(res$xhat_mean,norm_means,norm_sds)}
    xhat = res$xhat_rev
  }else if(method=="MF"){
    # xhat = res$xhat_rev
    if(is.null(res$xhat_rev)){res$xhat_rev = reverse_norm_MIWAE(res$xhat_mf,norm_means,norm_sds)}
    xhat = res$xhat_rev
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
  #LB=res$LB; time=res$time

  # # Clustering (no classes right now. commented out)
  # # Average the Z across the #imputation weights L
  # z_split = split(as.data.frame(res$zgivenx_flat),rep(1:res$opt_params$L,each=nrow(datas$test)))
  # z_mean = Reduce(`+`, z_split) / length(z_split)
  # #PCA, etc. not coded yet

  #results = c(unlist(imputation_metrics),LB,time)
  #names(results)[(length(results)-1):length(results)]=c("LB","time")
  results = c(unlist(imputation_metrics))
  return(list(res=res,results=results,call=call_name))
}
