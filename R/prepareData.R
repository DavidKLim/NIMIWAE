
split_data = function(data, ratio=c(8,2), seed=333){
  # splits the data into training-validation-test sets. Default = 8-2-0 --> test is train set
  # output g is the partition that can be used to split data and Missing (and probMissing optionally).
  ratio = ratio/sum(ratio) # if ratio doesn't sum to 1
  set.seed(333)
  g = sample(cut(
    seq(nrow(data)),
    nrow(data)*cumsum(c(0,ratio)),
    # labels = c("train","valid","test")
    labels = c("train","valid")
  ))
  return(g)
}

#' Simulate data
#'
#' @param N Number of observations
#' @param D Dimension of latent Z
#' @param P Number of features
#' @param sim_index Integer index for sim. Can use to set seed
#' @param seed Seed. Either set custom or set some value x sim_index. Default: 9 x sim_index
#' @param ratio Train-valid-test ratio for splitting of observations
#' @param g_seed Seed for train-valid-test dataset splitting of observations
#' @param beta coefficients for each column of X in simulating a binary class response variable
#' @return list of objects: data (N x P matrix), classes (subgroups of observations), params (those used for simulating data), and g (partitioning of data into train-valid-test sets)
#' @examples
#' simulate_data(N = 10000, D = 2, P = 8, sim_index = 1)
#'
#' @author David K. Lim, \email{deelim@live.unc.edu}
#' @references \url{https://github.com/DavidKLim/NIMIWAE}
#'
#' @importFrom qrnn elu
#'
#' @export
simulate_data = function(N, D, P, sim_index, seed = 9*sim_index, ratio=c(8,2), g_seed = 333,
                         beta=c(rep(-1/4,floor(P/2)), rep(1/4, P-floor(P/2))), nonlinear=F){
  # simulate data by simulating D-dimensional Z latent variable for N observations
  # and then apply W and B (both drawn from N(0,1)) weights and biases to obtain X
  set.seed(seed)

  #### NEED TO UPDATE THESE GENERATIONS
  if(nonlinear){
    print("Nonlinear data generation")
    # H = 64
    # sd1 = 1/4; sd2 = 1/4
    # W1 = matrix(rnorm(D*H,mean=0,sd=sd1),nrow=D,ncol=H) # weights
    # # W2 = matrix(rnorm(H*H,mean=0,sd=sd2),nrow=H,ncol=H)
    # W2 = matrix(rnorm(H*P,mean=0,sd=sd1),nrow=H,ncol=P)
    # B1 = matrix(rnorm(N*H,mean=0,sd=sd2),nrow=N,ncol=H,byrow=T) # biases: same for each obs
    # # B2 = matrix(rnorm(N*H,mean=0,sd=sd2),nrow=N,ncol=H,byrow=T)
    # B2 = matrix(rnorm(N*P,mean=0,sd=sd2),nrow=N,ncol=P,byrow=T)
    #
    # # W0 = matrix(rnorm(D*P,mean=0,sd=sd2),nrow=D,ncol=P)
    # # B0 = matrix(rnorm(N*P,mean=0,sd=sd3),nrow=N,ncol=P,byrow=T)
    #
    # # library(qrnn) # import "elu" function
    # X = qrnn::elu(Z%*%W1 + B1) %*% W2 + B2  # mimicking 1 HL with elu activation functions -- > NxP matrix

    Z1 = MASS::mvrnorm(N, rep(0,D), Sigma=diag(D))
    Z2 = MASS::mvrnorm(N, rep(0,D), Sigma=diag(D))

    sd1 = 0.5

    W1 = matrix(runif(D*floor(P/2),0.5,1),nrow=D,ncol=floor(P/2)) # weights
    W2 = matrix(runif(D*(P-floor(P/2)),0.5,1),nrow=D,ncol=P-floor(P/2)) # weights
    B1 = matrix(rnorm(N*floor(P/2),0,sd1), nrow=N, ncol=floor(P/2))
    B2 = matrix(rnorm(N*(P-floor(P/2)),0,sd1),nrow=N,ncol=P-floor(P/2))
    X1 = qrnn::sigmoid(Z2%*%W1+B1)
    X2 = cos(Z1%*%W2+B2) + qrnn::sigmoid(Z2%*%W2+B2)


    X=cbind(X1,X2)

    sds=list(sd1=sd1); W=list(W1=W2,W2=W2); B=list(B1=B1,B2=B2)
  }else{
    Z = MASS::mvrnorm(N, rep(0,D), Sigma=diag(D))
    sd1 = 0.5; sd2=1
    print("Linear data generation")
    W = matrix(runif(D*P, 0, sd1),nrow=D,ncol=P) # weights
    B = matrix(rnorm(N*P, 0, sd2),nrow=N,ncol=P)

    X = Z%*%W + B
    sds=list(W=sd1, B=sd2)
  }
  X = apply(X,2,function(x){(x-mean(x))/sd(x)}) # pre normalize

  params=list(N=N, D=D, P=P, Z=Z, W=W, B=B, seed=seed, sds=sds)

  find_int = function(p,beta) {
    # Define a path through parameter space
    f = function(t){
      sapply(t, function(y) mean(1 / (1 + exp(-y -X %*% beta))))
    }
    alpha <- uniroot(function(t) f(t) - p, c(-1e6, 1e6), tol = .Machine$double.eps^0.5)$root
    return(alpha)
  }
  inv_logit = function(x){
    return(1/(1+exp(-x)))
  }
  logit = function(x){
    if(x<=0 | x>=1){stop('x must be in (0,1)')}
    return(log(x/(1-x)))
  }
  alph <- sapply(0.5, function(y)find_int(y,beta))

  beta0 = alph
  mod = beta0 + X%*%beta

  probs = inv_logit(mod)
  classes = rbinom(N,1,probs)


  params=list(N=N, D=D, P=P, Z=Z, H=H, W=W, B=B, beta0=beta0, beta=beta, probs=probs, seed=seed)

  # ## simulate clustered data?
  # if(dataset=="TOYZ_CLUSTER"){
  #   N=100000; D=2; P=8; seed=9*sim_index
  #   set.seed(seed)
  #   classes=sample(c(1,2,3,4),N,replace=T)
  #
  #   Z=matrix(nrow=N,ncol=D)
  #   for(d in 1:D){
  #     for(c in 1:length(unique(classes))){
  #       Z[classes==c,d]=rnorm(sum(classes==c),mean=0+c*3,sd=1)
  #     }
  #   }
  #   W = matrix(nrow=D,ncol=P) # weights
  #   B = matrix(nrow=N,ncol=P) # biases
  #   for(p in 1:P){
  #     W[,p]=rnorm(D,mean=0,sd=1)
  #     B[,p]=rnorm(N,mean=0,sd=1)
  #   }
  #   X = Z%*%W + B
  #   data = X
  #   params=list(N=N, D=D, P=P, Z=Z, W=W, B=B, seed=seed)
  # }

  set.seed(g_seed)
  g = split_data(data=X, ratio=ratio)
  return(list(data=X, classes=classes, params=params, g=g))
}




###### NEED TO ADJUST INPUTS FOR SIMULATIONS, AND INCORPORATE simulate_data() function #####
#' Read UCI or Physionet 2012 Challenge Data
#'
#' @param dataset String for name of dataset. Valid datasets: "BANKNOTE","....".
#' @param ratio Train-valid-test ratio for splitting of observations
#' @param g_seed Seed for train-valid-test dataset splitting of observations
#' @return list of objects: data (N x P matrix), classes (subgroups of observations), params (those used for simulating data), and g (partitioning of data into train-valid-test sets)
#' @examples
#' read_data(dataset = "BANKNOTE")
#'
#' @author David K. Lim, \email{deelim@live.unc.edu}
#' @references \url{https://github.com/DavidKLim/NIMIWAE}
#'
#' @importFrom gdata read.xls
#' @importFrom reticulate import
#' @importFrom lubridate hms hour minute
#'
#' @export
read_data = function(dataset=c("Physionet_mean","Physionet_all","HEPMASS","POWER","GAS","IRIS","RED","WHITE","YEAST","BREAST","CONCRETE","BANKNOTE"), ratio=c(8,2), g_seed = 333){
  if(grepl("Physionet",dataset)){
    np <- reticulate::import("numpy")
    npz1 <- np$load("data/PhysioNet2012/physionet.npz")
    classes=c(npz1$f$y_train, npz1$f$y_val, npz1$f$y_test)
    params=NULL
    if(strsplit(dataset,"_")[[1]][2] == "mean"){
      # dataset=="Physionet_mean"
      data = rbind(apply(npz1$f$x_train_miss, c(1,3), mean),
                   apply(npz1$f$x_val_miss, c(1,3), mean),
                   apply(npz1$f$x_test_miss, c(1,3), mean))
    } else if(strsplit(dataset,"_")[[1]][2] == "all"){
      # dataset=="Physionet_all"
      library(abind)
      X3D = aperm(abind(npz1$f$x_train_miss,
                        npz1$f$x_val_miss,
                        npz1$f$x_test_miss,
                        along=1),
                  c(2,1,3))                                             # switch dims so 48 time points is first dim
      data = matrix(X3D, nrow=dim(X3D)[1]*dim(X3D)[2], ncol=dim(X3D)[3])   # stack time series data: 1st subject is 1st - 48th observations, 2nd subj is 49th - 96th, ...
      classes=rep(classes, each = 48)
    }
  }else if(dataset=="MINIBOONE"){  # 130K x 50
    # pre-processing steps from https://github.com/gpapamak/maf/blob/master/datasets/miniboone.py
    f = sprintf("./Results/%s/1000_train.csv.gz",dataset)
    if(!file.exists(f)){ download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt", destfile=f) }
    data <- read.table(f, skip=1)
    classes=NULL; params=NULL
    indices = (data[,1] < -100)
    data=data[!indices, ]

    data = (data - colMeans(data)) / apply(data,2,sd)
    ## no pruning of features (not sure if it's necessary)
    # features_to_remove = apply(data,2,function(x) any(table(x)>6))
  }else if(dataset=="HEPMASS"){ # 7M x 27 --> 300K x 21 feats
    # temp <- tempfile()
    f1 = sprintf("./Results/%s/1000_train.csv.gz",dataset)
    f2 = sprintf("./Results/%s/1000_test.csv.gz",dataset)
    if(!file.exists(f1)){ download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00347/1000_train.csv.gz", destfile=f1) }
    if(!file.exists(f2)){ download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00347/1000_test.csv.gz", destfile=f2) }
    data1 <- read.csv(gzfile(f1),header=F,skip=1, nrows=700000)
    data2 <- read.csv(gzfile(f2),header=F,skip=1, nrows=350000)

    data=rbind(data1,data2)

    data = data[data[,1]==1,]    # only class 1 (class 0 is noise)

    classes = data[,1]
    data = data[,-1]; data = data[,-ncol(data)]   # first column: class. last column is bad? pre-processed out

    features_to_remove = apply(data,2,function(x) max(table(x))) > 100    # if more than 100 repeats in feature, remove
    data = data[, !features_to_remove]

    data = (data - colMeans(data)) / apply(data,2,sd)   # normalize
    params=NULL
  }else if(dataset=="POWER"){ # 2.05M --> 1M x 6
    f1 = sprintf("./Results/%s/household_power_consumption.zip",dataset)
    if(!file.exists(f1)){download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip",destfile=f1)}
    data <- read.csv(unz(f1, "household_power_consumption.txt"),header=T,sep=";")
    data=data[,-1]; data=data[,-3]; data=data[,-4]    # following https://github.com/gpapamak/maf/blob/master/datasets/power.py . They get rid of Date, Global reactive power, and Global Intensity
    time_res = lubridate::hms(data[,1]); data[,1] = 60*lubridate::hour(time_res) + lubridate::minute(time_res)

    for(j in 2:5){ data[,j] = as.numeric(data[,j]) }
    data = data[!is.na(rowSums(data)),]
    data = (data - matrix(colMeans(data),nrow=nrow(data),ncol=ncol(data),byrow=T)) / matrix(apply(data,2,sd),nrow=nrow(data),ncol=ncol(data),byrow=T)
    classes=NULL
    params=NULL
    set.seed(9)
    data=data[sample(1:nrow(data),1000000,replace=F),]
  }else if(dataset=="GAS"){ # 4.21M --> 1M x 8. Some features taken out due to high correlation
    temp <- tempfile()
    download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00322/data.zip",temp)
    data <- read.table(unz(temp, "ethylene_CO.txt"),header=F,skip=1,sep="")
    unlink(temp)
    classes=NULL
    params=NULL

    # pre-processing step from https://github.com/gpapamak/maf/blob/master/datasets/gas.py
    ## drop first three columns (time, meth, eth concentrations)
    data=data[,-c(1,2,3)]
    #set.seed(99); data=data[sample(1:nrow(data),floor(0.25*nrow(data)),replace=F),] # sample 25% of data

    ## remove highly correlated columns
    get_corr_nums = function(data){
      C=cor(data)
      A=C>0.98
      B=colSums(A)
      return(B)
    }
    B=get_corr_nums(data)
    while(any(B>1)){
      col_to_remove = which(B > 1)[1]
      col_name = names(B)[col_to_remove]
      data=data[,-which(colnames(data)==col_name)]
      B = get_corr_nums(data)
    }

    set.seed(9)
    data=data[sample(1:nrow(data),1000000,replace=F),]     # sample 1M of the 4.209M observations (ACFlow does this too)

  }else if(dataset=="IRIS"){ # 150 x 4
    data <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"),
                     header=FALSE, col.names=c("sepal.length","sepal.width","petal.length","petal.width","species"))
    classes=data[,ncol(data)]
    data=data[,-ncol(data)]     # take out class (3)
    params=NULL
  } else if(dataset=="RED"){ # 1599 x 12
    data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"),header=TRUE,sep=";")
    classes=data[,ncol(data)]
    data=data[,-ncol(data)]     # take out class (quality, 1-10)
    params=NULL
  } else if(dataset=="WHITE"){ # 4898 x 12
    data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"),header=TRUE,sep=";")
    classes=data[,ncol(data)]
    data=data[,-ncol(data)]     # take out class (quality, 1-10)
    params=NULL
  } else if(dataset=="BREAST"){ # 569 x 30
    # Wisconsin BC (Diagnostic)
    data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"),
                     header=FALSE)
    colnames(data)=c("ID","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                     "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
                     "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                     "concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst",
                     "area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst")
    data=data[,-1]       # remove ID's
    classes=data[,1]
    data=data[,-1]      # take out class (2)
    params=NULL
  } else if(dataset=="YEAST"){  # 1484 x 8
    data <- read.table(url("https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"),header=FALSE)
    colnames(data)=c("sequence_name","mcg","gvh","alm","mit","erl","pox","vac","nuc","class")
    data=data[,-1]   # remove sequence names
    classes=data[,ncol(data)]
    data=data[,-ncol(data)] # take out class (10)
    params=NULL
  } else if(dataset=="CONCRETE"){ # 1030 x 8
    data <- gdata::read.xls("http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls")
    classes=data[,ncol(data)]
    data=data[,-ncol(data)] # take out class: here it's continuous
    params=NULL
  } else if(dataset=="BANKNOTE"){ # 1372 x 4
    data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"),header=FALSE)
    classes=data[,ncol(data)]
    data=data[,-ncol(data)] # take out class (2)
    params=NULL
  }

  set.seed(g_seed)
  g = split_data(data=data.frame(data), ratio=ratio)
  return(list(data=data, classes=classes, params=params, g=g))
}







#' Simulate different mechanisms of missingness
#'
#' @param data data frame of data (N x P)
#' @param miss_cols Columns to impose missingness on
#' @param ref_cols Column(s) to use as covariates of missingness for MAR or MNAR. If scheme="UV" and mechanism="MAR", then each element of ref_cols is used as a covariate for each corresponding element of miss_cols (length of miss_cols must be smaller than ref_cols)
#' @param pi Proportion of entries that are missing (for all miss_cols)
#' @param phis Coefficients of each covariate in missingness model. Corresponding element is coef of corr element in miss_cols if MNAR, and in ref_cols if MAR (for scheme=UV)
#' @param phi_z Coefficient of Z in missingness model, only used if fmodel="PM"
#' @param scheme "UV" or "MV": "UV" uses one covariate in missingness model of each miss_cols element (mechanism = "MAR" uses corresponding element of ref_cols for each corresponding element of miss_cols, and mechanism = "MNAR" uses itself for miss model for each miss_cols). "MV" uses all ref_cols of each missingness model
#' @param mechanism "MCAR", "MAR", or "MNAR" missingness. Only pertinent for fmodel="S".
#' @param sim_index Index of simulation run: varies seed based on sim_index and column index for each element of miss_cols for reproducibility.
#' @param fmodel "S" or "PM" for selection model or pattern-mixture model. If "PM", mechanism will be MNAR regardless of specification, and latent matrix Z must be specified and will be used as the only covariate of all missingness models
#' @param Z Matrix of simulated values of the latent variable
#' @return list of objects: Missing (N x P mask matrix), probs (N x P matrix of probabilities of each entry being missing), params (pertaining to simulations),
#' mechanism (of missingness), scheme (univariate or multivariate logistic regression model), phi0s (intercepts for each missingness model),
#' phis (coefficients for each covariate of each missingness model), phi_z (coefficient of Z, if fmodel="PM"))
#' @examples
#' data = read_data("BANKNOTE",NULL,1)$data
#' set.seed(111)
#' ref_cols=sample(c(1:ncol(data)),ceiling(ncol(data)/2),replace=F); miss_cols=(1:ncol(data))[-ref_cols]
#' simulate_missing(data, miss_cols, ref_cols, 0.5, rep(5,length(miss_cols)), NULL, "UV", "MNAR")
#' @export
simulate_missing = function(data,miss_cols,ref_cols,pi,
                            phis,phi_z,
                            scheme,mechanism,sim_index=1,fmodel="S", Z=NULL){

  #sim_index=1 unless otherwise specified
  n=nrow(data)
  # function s.t. expected proportion of nonmissing (Missing=1) is p. let p=1-p_miss
  find_int = function(p,phi) {
    # Define a path through parameter space
    f = function(t){
      sapply(t, function(y) mean(1 / (1 + exp(-y -x %*% phi))))
    }
    alpha <- uniroot(function(t) f(t) - p, c(-1e6, 1e6), tol = .Machine$double.eps^0.5)$root
    return(alpha)
  }
  inv_logit = function(x){
    return(1/(1+exp(-x)))
  }
  logit = function(x){
    if(x<=0 | x>=1){stop('x must be in (0,1)')}
    return(log(x/(1-x)))
  }

  Missing = matrix(1,nrow=nrow(data),ncol=ncol(data))
  prob_Missing = matrix(1,nrow=nrow(data),ncol=ncol(data))
  params=list()
  phi0s = rep(0,length(miss_cols))
  for(j in 1:length(miss_cols)){
    # specifying missingness model covariates
    if(mechanism=="MCAR"){
      x <- matrix(rep(0,n),ncol=1) # for MCAR: no covariate (same as having 0 for all samples)
      phi=0                       # for MCAR: no effect of covariates on missingness (x is 0 so phi doesn't matter)
    }else if(mechanism=="MAR"){
      if(scheme=="UV"){
        x <- matrix(data[,ref_cols[j]],ncol=1)             # missingness dep on just the corresponding ref column (randomly paired)
        phi=phis[ref_cols[j]]
      }else if(scheme=="MV"){
        # check if missing column in ref. col: this would be MNAR (stop computation)
        if(any(ref_cols %in% miss_cols)){stop(sprintf("missing cols in reference. is this intended? this is MNAR not MAR."))}
        x <- matrix(data[,ref_cols],ncol=length(ref_cols)) # missingness dep on all ref columns
        phi=phis[ref_cols]
      }else if(scheme=="NL"){
        ## Nonlinear
        ## if(any(is.na(data[,1:3]))){stop("Missingness in 1st 3 cols of data used for nonlinear MAR interactions")}
        ## x = cbind(data[,1]*data[,2], data[,2]*data[,3], data[,1]*data[,2]*data[,3])  # 3 nonlinear predictors (observed)
        ## beta = betas[1:3]/colMeans(x)  # scale effect of each nonlinear term by mean --> scale down effect of large predictors
        # x <- matrix(log(data[,ref_cols[j]] + abs(min(data[,ref_cols[j]])) + 0.01),ncol=1) # just the corresponding ref column
        x <- matrix((data[,ref_cols[j]])^2,ncol=1) # just the corresponding ref column
        beta=betas[miss_cols[j]]
        # jj = if(j>1){j-1}else{jj=length(ref_cols)}
        # x <- cbind(exp(data[,ref_cols[j]]), (data[,ref_cols[jj]])^2, data[,ref_cols[jj]]*data[,ref_cols[j]]) # just the corresponding ref column
        # beta = rep(betas[ref_cols[j]], 3)
      }
    }else if(mechanism=="MNAR"){
      if(fmodel=="S"){
        # Selection Model
        if(scheme=="UV"){
          # MISSINGNESS OF EACH MISS COL IS ITS OWN PREDICTOR
          x <- matrix(data[,miss_cols[j]],ncol=1) # just the corresponding ref column
          phi=phis[miss_cols[j]]
        }else if(scheme=="MV"){
          # check if missing column not in ref col. this might be MAR if missingness not dep on any other missing data
          if(all(!(ref_cols %in% miss_cols))){warning(sprintf("no missing cols in reference. is this intended? this might be MAR not MNAR"))}
          x <- matrix(data[,ref_cols],ncol=length(ref_cols)) # all ref columns
          phi=phis[ref_cols]         # in MNAR --> ref_cols can overlap with miss_cols (dependent on missingness)

          # address when miss_cols/ref_cols/phis are not null (i.e. want to induce missingness on col 2 & 5 based on cols 1, 3, & 4)
        }else if(scheme=="NL"){
          # ## Nonlinear
          # # x = cbind(data[,1]*data[,2], data[,2]*data[,3], data[,1]*data[,2]*data[,3], log(data[,4]), exp(data[,5]))  # 5 nonlinear predictors
          # # x = cbind(data[,1]*data[,2], data[,2]*data[,3], data[,1]*data[,2]*data[,3])  # 3 nonlinear predictors
          # x = cbind(data[,1], data[,1]*data[,2], data[,3]^2)  # 3 nonlinear predictors
          # # beta = betas[1:5]/colMeans(x)  # scale effect of each nonlinear term by mean --> scale down effect of large predictors
          # # beta = betas[1:3]/colMeans(x)  # scale effect of each nonlinear term by mean --> scale down effect of large predictors
          # beta = betas[1:3]  # scale effect of each nonlinear term by mean --> scale down effect of large predictors
          x <- matrix((data[,miss_cols[j]])^2,ncol=1) # just the corresponding ref column
          beta=betas[miss_cols[j]]
          # jj = if(j>1){j-1}else{jj=length(miss_cols)}
          # x <- cbind(exp(data[,miss_cols[j]]), (data[,miss_cols[jj]])^2, data[,miss_cols[jj]]*data[,miss_cols[j]]) # just the corresponding ref column
          # beta = rep(betas[miss_cols[j]], 3)
        }
      } else if(fmodel=="PM"){
        x <- Z
        phi = phi_z # phis should be length Z
      }
    }
    alph <- sapply(pi, function(y)find_int(y,phi))
    phi0s[j] = alph
    mod = alph + x%*%phi

    # Pr(Missing_i = 1)
    probs = inv_logit(mod)
    prob_Missing[,miss_cols[j]] = probs

    # set seed for each column for reproducibility, but still different across columns
    set.seed(j+(sim_index-1)*length(miss_cols))
    Missing[,miss_cols[j]] = rbinom(n,1,probs)

    params[[j]]=list(phi0=alph, phi=phi, miss=miss_cols[j], ref=ref_cols[j], scheme=scheme)
  }

  return(list(Missing=Missing,
              probs=prob_Missing,params=params, mechanism=mechanism,scheme=scheme, phi0s=phi0s, phis=phis, phi_z=phi_z))
}
