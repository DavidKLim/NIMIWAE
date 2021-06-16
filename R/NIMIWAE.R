# call this NIMIWAE.R
# toy_sim.R procedure: simulate (or read) data --> simulate missingness --> 60-20-20 split --> save --> tune_hyperparams() --> output results

# process_results() with each method. change file naming

#' Process results: return imputation metrics
#'
#' @param data Data matrix (N x P)
#' @param data_types Vector of data types ("real", "count", "pos", "cat")
#' @param Missing Missingness mask matrix (N x P)
#' @param g Training-validation-test split partitioning
#' @param rdeponz TRUE/FALSE: Whether to allow missingness (r) to depend on the latent variable (z). Default is FALSE
#' @param learn_r TRUE/FALSE: Whether to learn missingness model via appended NN (TRUE, default), or fit a known logistic regression model (FALSE). If FALSE, `phi0` and `phi` must be specified
#' @param phi0 (optional) Intercept of logistic regression model, if learn_r = FALSE.
#' @param phi (optional) Vector of coefficients of logistic regression model for each input covariates `covars_r`, if learn_r = FALSE. `phi` must be the same length as the number of input covariates, or `sum(covars_r)`.
#' @param ignorable TRUE/FALSE: Whether missingness is ignorable (MCAR/MAR) or nonignorable (MNAR, default). If missingness is known to be ignorable, "ignorable=T" omits missingness model.
#' @param covars_r Vector of 1's and 0's of whether each feature is included as covariates in the missingness model. Need not be specified if `ignorable = T`. Default is using all features as covariates in missingness model. Must be length P (or `ncol(data)`)
#' @param arch Architecture of NIMIWAE. Can be "IWAE" or "VAE". "VAE" is specific case of the "IWAE" where only one sample is drawn from the joint posterior of (z, xm).
#' @param hyperparameters List of grid of hyperparameter values to search. Relevant hyperparameters: `sigma`: activation function ("relu" or "elu"), `h`: number of nodes per hidden layer, `n_hidden_layers`: #hidden layers (except missingness model Decoder_r), `n_hidden_layers_r`: #hidden layers in missingness model (Decoder_r). If "NULL" then set as the same value as each n_hidden_layers (not tuned). Otherwise, can tune a different grid of values; `bs`: batch size, `lr`: learning rate, `dim_z`: dimensionality of latent z, `niw`: number of importance weights (samples drawn from each latent space), `n_imputations`, `n_epochs`: maximum number of epochs
#' @return res object: NIMIWAE fit containing ... on the test set
#' @examples
#' fit_data = read_data("CONCRETE"); data = fit_data$data
#' # fit_data = simulate_data(N=100000, D=1, P=2, sim_index=1)   # optionally: simulate data with 100K obs, 1 latent dim, 2 features
#' set.seed(111); ref_cols=sample(c(1:ncol(data)),ceiling(ncol(data)/2),replace=F); miss_cols=(1:ncol(data))[-ref_cols]
#' set.seed(222); phis=rlnorm(ncol(data),log(5),0.2)
#' fit_Missing = simulate_missing(data, miss_cols, ref_cols, pi=0.5, phis, NULL, "UV", "MNAR")
#' data=fit_data$data; Missing=fit_Missing$Missing; g=fit_data$g
#' res=NIMIWAE(data, Missing, g)    # using default hyperparameters grid
#' imp_metrics = processResults(data=data, Missing=Missing, g=g, res=res)
#'
#' @author David K. Lim, \email{deelim@live.unc.edu}
#' @references \url{https://github.com/DavidKLim/NIMIWAE}
#'
#' @importFrom reticulate source_python import
#'
#' @export
NIMIWAE = function(data, dataset, data_types, Missing, g=NULL, rdeponz=F, learn_r=T, phi0=NULL, phi=NULL, ignorable=F, covars_r=rep(1,ncol(data)), arch="IWAE", draw_xmiss=T,
                   hyperparameters=list(sigma="elu", h=c(64L), n_hidden_layers=c(1L,2L), n_hidden_layers_r0=c(0L,1L),
                                        bs=c(1000L), lr=c(0.001,0.01), dim_z=as.integer(c(floor(ncol(data)/2),floor(ncol(data)/4))),
                                        niw=5L, n_imputations=5L, n_epochs=2002L), save_imps=F, dir_name=".", normalize=T
                   ){

  ## n_hidden_layers_r is set as the same as n_hidden_layers, unless an integer is specified
  #############################################################################################################
  ############ DEFINE Cs, and create X_aug (split categorical values to dummy variables of 1/0) ###############
  #############################################################################################################

  np = reticulate::import("numpy")

  data_types_0 = data_types

  N = nrow(data); P=ncol(data)
  if(sum(data_types=="cat")==0){
    # if no categorical variables
    data_aug = data
    Missing_aug = Missing
    covars_r_aug = covars_r
    Cs = np$empty(shape=c(0L,0L))
    data_types_aug = data_types
  } else{
    data_aug = data[, data_types != "cat"]
    Missing_aug = Missing[, data_types != "cat"]
    covars_r_aug = covars_r[data_types != "cat"]
    # if any categorical variables --> need to dummy encode
    Cs = rep(0, sum(data_types=="cat"))
    cat_ids = which(data_types=="cat")
    for(i in 1:length(cat_ids)){
      data_cat = as.numeric(as.factor(data[,cat_ids[i]]))-1
      Cs[i] = length(unique(data_cat))
      data_cat_onehot = matrix(ncol=Cs[i], nrow=length(data_cat))
      for(ii in 1:Cs[i]){
        data_cat_onehot[,ii] = (data_cat==ii-1)^2
      }
      data_aug = cbind(data_aug, data_cat_onehot)
      Missing_aug = cbind(Missing_aug, matrix(Missing[,cat_ids[i]], nrow=N, ncol=Cs[i]))
      covars_r_aug = c(covars_r_aug, rep(covars_r[data_types=="cat"][i],Cs[i]))
    }
    data_types_aug = c( data_types[!(data_types %in% c("cat"))], rep("cat",sum(Cs)) )
    Cs = np$array(Cs)
  }

  # 2) set up g= ... splits in this function
  # datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  # Missings = split(data.frame(Missing), g)

  # norm_means=colMeans(datas$train); norm_sds=apply(datas$train,2,sd)    # calculate normalization mean/sd on training set --> use for all

  reticulate::source_python(system.file("NIMIWAE.py", package = "NIMIWAE"))
  t0 = Sys.time()
  res = do.call(NIMIWAE::tuneHyperparams, c(list(method="NIMIWAE",data=data_aug,dataset=dataset,data_types=data_types_aug, data_types_0=data_types_0,Missing=Missing_aug,g=g,
                                            rdeponz=rdeponz, learn_r=learn_r,
                                            phi0=phi0,phi=phi,
                                            covars_r=covars_r_aug,
                                            arch=arch, draw_xmiss=draw_xmiss, Cs=Cs, ignorable=ignorable, save_imps=save_imps, dir_name=dir_name, normalize=normalize), hyperparameters))
  res$time = as.numeric(Sys.time()-t0, units="secs")
  # res = tuneHyperparams(method="NIMIWAE",data=data,Missing=Missing,g=g,
  #                                rdeponz=rdeponz, learn_r=learn_r,
  #                                phi0=phi0,phi=phi,
  #                                covars_r=covars_r,
  #                                arch=arch, ignorable=ignorable)

  ## Code these defaults in NIMIWAE_Paper
  # if(dataset%in%c("TOYZ","TOYZa")){sigma="elu"; hs=c(4L,8L); bss=c(5000L); lrs=c(0.001,0.01); dim_zs=c(1L,2L); niws=5L; n_epochss=2002L}
  # if(dataset%in%c("TOYZ2","TOYZ2a")){sigma="elu"; hs=c(16L,8L); bss=c(10000L); lrs=c(0.001,0.01); dim_zs=c(4L,2L); niws=5L; n_epochss=2002L}
  # if(dataset=="TOYZ50"){sigma="elu"; hs=c(128L,64L); bss=c(10000L); lrs=c(0.001,0.01); dim_zs=c(8L,4L); niws=5L; n_epochss=2002L}
  # if(dataset=="TOYZ_CLUSTER"){sigma="elu"; hs=c(128L,64L); bss=c(10000L); lrs=c(0.001,0.01); dim_zs=c(8L,4L); niws=5L; n_epochss=2002L}
  # if(dataset%in%c("BANKNOTE","WINE","BREAST","YEAST","CONCRETE","SPAM","ADULT","RED")){
  #   sigma="elu"; hs=c(128L,64L); bss=c(200L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  # if(dataset %in% c("SPAM","ADULT","WHITE")){
  #   sigma="elu"; hs=c(128L,64L); bss=c(1000L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  # if(dataset%in% c("GAS","POWER","HEPMASS","MINIBOONE")){
  #   sigma="elu"; hs=c(128L,64L); bss=c(20000L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  # if(dataset=="IRIS"){
  #   sigma="elu"; hs=c(128L,64L); bss=c(20L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  # if(dataset=="Physionet_mean"){
  #   sigma="elu"; hs=c(128L,64L); bss=c(300L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  # if(dataset=="Physionet_all"){
  #   sigma="elu"; hs=c(128L,64L); bss=c(5000L); lrs=c(0.001,0.01); dim_zs=as.integer(c(floor(p/2),floor(p/4))); niws=5L; n_epochss=2002L}
  return(res)
  #save(res,file=sprintf(""...))
}
