
# sparsity = "prune" (turned off), "dropout" or "none"
def run_NIMIWAE(rdeponz,data,data_types,data_types_0,data_val,Missing,Missing_val,covars_r,norm_means,norm_sds,learn_r,Cs,ignorable=False,n_hidden_layers=2,n_hidden_layers_r=0,L1_weight=0,L2_weight=0,sparse="none",dropout_pct=None,prune_pct=None,covars_miss=None,covars_miss_val=None,impute_bs=None,arch="IWAE",draw_xmiss=False,pre_impute_value=0,h1=64,h2=None,h3=None,h4=None,phi0=None,phi=None,train=1,warm_start=False,saved_model=None,early_stop=False,sigma="relu",bs = 64,n_epochs = 2002,lr=0.001,niw=20,dim_z=5,L=20,M=20,save_imps=False,dir_name=".",trace=False):
  # add_miss_term = True --> adds p(x^m) term into loss function --> reconstruction of msising values
  ## only applicable when true data input --> essentially improves x^m reconstruction directly as if no missing data
  # rdeponz : True or False --> if True, then q(z|x^o) -> q(z|x^o,r) and p(r|x) -> p(r|x,z)
  ## "r" concatenates binary 0/1's when conditional on r
  ## "pr" concatenates input prob_Missing p(r|x) or p(r|x,z). if no input prob_Missing, then iteratively learns
  # dec_distrib = "Normal" or "StudentT"
  # if draw_xmiss=False --> feed true missing values into learning of R (or logistic regression if learn_r=False)
  if (h2 is None) and (h3 is None) and (h4 is None):
    h2=h1; h3=h1; h4=h1  # h1: encoder, h2: decoder_x, h3: decoder_r, h4: encoder_xr
  import torch     # this module not found in Longleaf
  #import torchvision
  import torch.nn as nn
  import numpy as np
  import scipy.stats
  import scipy.io
  import scipy.sparse
  from scipy.io import loadmat
  import pandas as pd
  import matplotlib.pyplot as plt
  import torch.distributions as td
  from torch import nn, optim
  from torch.nn import functional as F
  #import torch.nn.utils.prune as prune
  #from torchvision import datasets, transforms
  #from torchvision.utils import save_image
  import time
  import sys

  from torch.distributions import constraints
  from torch.distributions.distribution import Distribution
  from torch.distributions.utils import broadcast_all
  import torch.nn.functional as F
  from torch.autograd import Variable
  #import torch.nn.utils.prune as prune
  from collections import OrderedDict
  
  ## default for beta-VAE (deprecated)
  # beta=0; beta_anneal_rate=1/500
  ## not using beta-VAE
  # beta=1; beta_anneal_rate=0
  
  # add_miss_term=False; draw_xobs=False; draw_xmiss=True   # deprecated.
  
  print("norm_means (first 10):");print(norm_means[:10])
  print("norm_sds (first 10):");print(norm_sds[:10])
  temp0 = torch.ones([1], dtype=torch.float64, device='cuda:0')
  temp = torch.ones([1], dtype=torch.float64, device='cuda:0')
  # temp0 = torch.tensor([0.5], dtype=torch.float64, device='cuda:0')
  # temp = torch.tensor([0.5], dtype=torch.float64, device='cuda:0')
  # temp_min = torch.tensor(0.5,device="cuda:0",dtype=torch.float64)
  temp_min = torch.tensor(0.01,device="cuda:0",dtype=torch.float64)
  ANNEAL_RATE = torch.tensor(0.001,device="cuda:0",dtype=torch.float64)  # https://github.com/vithursant/VAE-Gumbel-Softmax
  # ANNEAL_RATE = torch.tensor(0.00003,device="cuda:0",dtype=torch.float64)  # https://github.com/vithursant/VAE-Gumbel-Softmax
  # ANNEAL_RATE = torch.tensor(0.0001,device="cuda:0",dtype=torch.float64)  # https://github.com/vithursant/VAE-Gumbel-Softmax
  
  # temp = temp_min; temp0 = temp_min
  # ANNEAL_RATE = torch.tensor(0,device="cuda:0",dtype=torch.float64)  # https://github.com/vithursant/VAE-Gumbel-Softmax
  
  # ANNEAL_RATE = torch.tensor(0.01,device="cuda:0",dtype=torch.float64)  # HIVAE (linear annealing)

  
  ids_real = data_types=='real'; p_real=np.sum(ids_real)
  ids_count = data_types=='count'; p_count=np.sum(ids_count)
  ids_cat = data_types=='cat'; p_cat = len(Cs) #p_cat=np.sum(ids_cat)
  ids_pos = data_types=='pos'; p_pos = np.sum(ids_pos)
  
  exists_types = [p_real>0, p_count>0, p_pos>0, p_cat>0]   # real, count, cat types. do they exist?
  print("exists_types (real, count, cat, pos):")
  print(exists_types)
  # print("p_real, p_count, p_pos, p_cat:")
  # print(str(p_real) + ", " + str(p_count) + ", " + str(p_pos) + ", " + str(p_cat))
  ids_types = [ids_real, ids_count, ids_pos, ids_cat]
  # print("ids_types:")
  # print(ids_types)
  
  if (not (np.array(covars_miss)==None).all()):
    covars=True
    pr1 = np.shape(covars_miss)[1]
  else:
    covars=False
    pr1=0
  
  #decoder_r = nn.Sequential(OrderedDict({
  #      'r1': torch.nn.Linear(8, 4),
  #}))
  p0 = len(data_types_0)


  # do "r" only for now
  def mse(xhat,xtrue,mask):
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return {'miss':np.mean(np.power(xhat-xtrue,2)[~mask]),'obs':np.mean(np.power(xhat-xtrue,2)[mask])}
    #return {'miss':np.mean(np.power(xhat-xtrue,2)[~mask]),'obs':np.mean(np.power(xhat-xtrue,2)[mask])}
  
  #xfull = (data - np.mean(data,0))/np.std(data,0)
  xfull = (data - norm_means)/norm_sds
  xfull_val = (data_val - norm_means)/norm_sds
  
  # Loading and processing data
  n = xfull.shape[0] # number of observations
  n_val = xfull_val.shape[0]
  p = xfull.shape[1] # number of features (should be same for train/val)
  
  np.random.seed(1234)

  bs = min(bs,n)
  bs_val = min(bs,n_val)
  impute_bs = min(bs, n)
  
  xmiss = np.copy(xfull)
  xmiss[Missing==0]=np.nan
  mask = np.isfinite(xmiss) # binary mask that indicates which values are missing
  # mask0 = np.copy(mask)
  xhat_0 = np.copy(xmiss)
  
  xmiss_val = np.copy(xfull_val)
  xmiss_val[Missing_val==0]=np.nan
  mask_val = np.isfinite(xmiss_val) # binary mask that indicates which values are missing
  # mask0_val = np.copy(mask_val)
  xhat_0_val = np.copy(xmiss_val)
  
  mask0 = np.empty([mask.shape[0],len(data_types_0)])
  mask0_val = np.empty([mask_val.shape[0],len(data_types_0)])
  covars_r0 = np.empty([len(data_types_0)])
  i_real=0; i_count=0; i_pos=0; i_cat=0; C0=0
  for i in range(0,len(data_types_0)):
    if data_types_0[i]=="real":
      mask0[:,i] = mask[:,np.where(ids_real)[0][i_real]]
      mask0_val[:,i] = mask_val[:,np.where(ids_real)[0][i_real]]
      covars_r0[i] = covars_r[np.where(ids_real)[0][i_real]]
      i_real=i_real+1
    if data_types_0[i]=="count":
      mask0[:,i] = mask[:,np.where(ids_count)[0][i_count]]
      mask0_val[:,i] = mask_val[:,np.where(ids_count)[0][i_count]]
      covars_r0[i] = covars_r[np.where(ids_count)[0][i_count]]
      i_count=i_count+1
    if data_types_0[i]=="pos":
      mask0[:,i] = mask[:,np.where(ids_pos)[0][i_pos]]
      mask0_val[:,i] = mask_val[:,np.where(ids_pos)[0][i_pos]]
      covars_r0[i] = covars_r[np.where(ids_pos)[0][i_pos]]
      i_pos=i_pos+1
    if data_types_0[i]=="cat":
      idd = np.where(ids_cat)[0][int(C0*i_cat):int(C0*i_cat + Cs[i_cat])]
      mask0[:,i] = mask[:,idd[0]]
      mask0_val[:,i] = mask_val[:,idd[0]]
      covars_r0[i] = covars_r[idd[0]]
      C0=Cs[i_cat]
      i_cat = i_cat+1
  
  # full_obs_ids = np.sum(Missing==0,axis=0)==0    # columns that are fully observed need not have missingness modelled
  full_obs_ids = np.sum(mask0==0,axis=0)==0    # columns that are fully observed: need to redo the missingness model
  p_miss = np.sum(~full_obs_ids)
  print("p_miss:" + str(p_miss))
  print("p_obs:" + str(np.sum(full_obs_ids)))
  
  #print(bs_val)
  #print(n_val)
  # print(xfull_val[:10])
  # print(xhat_0_val[:10])

  # Custom pre-impute values
  if (pre_impute_value == "mean_obs"): xhat_0[Missing==0] = np.mean(xmiss[Missing==1],0); xhat_0_val[Missing_val==0] = np.mean(xmiss_val[Missing_val==1],0)
  elif (pre_impute_value == "mean_miss"): xhat_0[Missing==0] = np.mean(xmiss[Missing==0]); xhat_0_val[Missing_val==0] = np.mean(xmiss_val[Missing_val==0])
  elif (pre_impute_value == "truth"): xhat_0 = np.copy(xfull); xhat_0_val = np.copy(xfull_val)
  else: xhat_0[np.isnan(xmiss)] = pre_impute_value; xhat_0_val[np.isnan(xmiss_val)] = pre_impute_value

  init_mse = mse(xfull,xhat_0,mask)
  print("Pre-imputation MSE (obs, should be 0): " + str(init_mse['obs']))
  print("Pre-imputation MSE (miss): " + str(init_mse['miss']))
  
  d = dim_z # dimension of the latent space
  K = niw # number of IS during training

  pr = np.sum(covars_r).astype(int)
  if not learn_r and not ignorable: phi=torch.from_numpy(phi).float().cuda()
  
  # Define decoder/encoder
  p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)
  if (sigma=="relu"): act_fun=torch.nn.ReLU()
  elif (sigma=="elu"): act_fun=torch.nn.ELU()
  
  def network_maker(act_fun, n_hidden_layers, in_h, h, out_h, bias=True, dropout=False):
    # create NN layers
    if n_hidden_layers==0:
      layers = [ nn.Linear(in_h, out_h, bias), ]
    elif n_hidden_layers>0:
      layers = [ nn.Linear(in_h , h, bias), act_fun, ]
      for i in range(n_hidden_layers-1):
        layers.append( nn.Linear(h, h, bias), )
        layers.append( act_fun, )
      layers.append(nn.Linear(h, out_h, bias))
    elif n_hidden_layers<0:
      raise Exception("n_hidden_layers must be >= 0")
    
    # insert dropout layer (if applicable)
    if dropout:
      layers.insert(0, nn.Dropout())
    
    # create NN
    model = nn.Sequential(*layers)
    
    # initialize weights
    def weights_init(layer):
      if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
    model.apply(weights_init)
    
    return model
  
  # if (dec_distrib=="Normal"): num_dec_params=2
  # elif (dec_distrib=="StudentT"): num_dec_params=3
  
  num_enc_params = p + p*(rdeponz==True)
  # num_enc_params = 2*p

  encoder = network_maker(act_fun, n_hidden_layers, num_enc_params, h1, 2*d, True, False).cuda()
  decoders = {}
  if exists_types[0]:
    decoders['real'] = network_maker(act_fun, n_hidden_layers, d, h2, 2*p_real, True, False).cuda()
    # decoders['real'] = network_maker(act_fun, n_hidden_layers, d+p, h2, 2*p_real, True, False).cuda()
  if exists_types[1]:
    decoders['count'] = network_maker(act_fun, n_hidden_layers, d, h2, 2*p_count, True, False).cuda()
    # decoders['count'] = network_maker(act_fun, n_hidden_layers, d+p, h2, 2*p_count, True, False).cuda()
  if exists_types[2]:
    decoders['pos'] = network_maker(act_fun, n_hidden_layers, d, h2, 2*p_pos, True, False).cuda()
    # decoders['pos'] = network_maker(act_fun, n_hidden_layers, d+p, h2, 2*p_pos, True, False).cuda()
  if exists_types[3]:
    decoders['cat']=[]
    for ii in range(0, p_cat):
      decoders['cat'].append( network_maker(act_fun, n_hidden_layers, d, h2, int(Cs[ii]), True, False).cuda() )
      # decoders['cat'].append( network_maker(act_fun, n_hidden_layers, d+p, h2, int(Cs[ii]), True, False).cuda() )
    
  if ignorable: p2=p+d         # if ignorable, only feed xo and z into q(xm|xo,z), since xm indep of r
  else: p2=p+p+d
  
  print("p2:")
  print(p2)
  print("ignorable:")
  print(ignorable)
  
  # encoder_xr = network_maker(act_fun, n_hidden_layers, p2, h4, 2*p, True, False)
  encoders_xr = {}
  if exists_types[0]:
    encoders_xr['real'] = network_maker(act_fun, n_hidden_layers, p2, h4, 2*p_real, True, False).cuda()
  if exists_types[1]:
    encoders_xr['count'] = network_maker(act_fun, n_hidden_layers, p2, h4, 2*p_count, True, False).cuda()
  if exists_types[2]:
    encoders_xr['pos'] = network_maker(act_fun, n_hidden_layers, p2, h4, 2*p_pos, True, False).cuda()
  if exists_types[3]:
    encoders_xr['cat'] = []
    for ii in range(0,p_cat):
      encoders_xr['cat'].append(network_maker(act_fun, n_hidden_layers, p2, h4, int(Cs[ii]), True, False).cuda())
  
  if not ignorable:
    # pr: number of features of data included as covariates in Decoder 2
    # pr1: number of additional covariates (like class) included as covariates in Decoder 2
    if (rdeponz): num_dec_r_params = pr + pr1 + d
    else: num_dec_r_params = pr + pr1; num_enc_params = p
    if learn_r:
      decoder_r = network_maker(act_fun, n_hidden_layers_r, num_dec_r_params, h3, p_miss, True, (sparse=="dropout") ).cuda()


  def forward(niw, iota_xfull, iota_x, mask, batch_size, tiledmask, tiled_iota_x, tiled_iota_xfull, tiled_tiled_covars_miss, temp):
    tiledtiledmask = torch.Tensor.repeat(tiledmask,[M,1]).cuda()
    tiled_tiled_iota_x = torch.Tensor.repeat(tiled_iota_x,[M,1]).cuda()
    if not draw_xmiss: tiled_tiled_iota_xfull = torch.Tensor.repeat(tiled_iota_xfull,[M,1]).cuda()
    else: tiled_tiled_iota_xfull = None
    # tiled_tiled_iota_xfull = None
    ## ENCODER ##
    if rdeponz:
      out_encoder = encoder(torch.cat([iota_x,mask],1))
    else:
      out_encoder = encoder(iota_x)
    # out_encoder = encoder(torch.cat([iota_x,mask],1))    # encoder input both R and Xo?
    # sample from ENCODER #
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])+0.001),1)
    params_z = {'mean': out_encoder[..., :d].reshape([batch_size,d]).detach().cpu().data.numpy(), 'scale': torch.nn.Softplus()(out_encoder[..., d:(2*d)]).reshape([batch_size,d]).detach().cpu().data.numpy()+0.001 }
    zgivenx = q_zgivenxobs.rsample([niw])
    zgivenx_flat = zgivenx.reshape([niw*batch_size,d])
    

    ## DECODER_X ##       p(xm,xo|z)
    
    out_decoders = {}; out_decoders['cat'] = []; p_xs={}; p_xs['cat'] = []
    params_x={}; params_x['cat']=[]
    
    if exists_types[0]:
      out_decoders['real'] = decoders['real'](zgivenx_flat)
      # out_decoders['real'] = decoders['real'](torch.cat([zgivenx_flat, tiled_iota_x],1))
      p_xs['real'] = td.Normal(loc=out_decoders['real'][..., :p_real], scale=torch.nn.Softplus()(out_decoders['real'][..., p_real:(2*p_real)]) + 0.001)
      # params_x['real'] = {'mean': torch.mean(out_decoders['real'][..., :p_real].reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy(), 'scale': torch.mean(torch.nn.Softplus()(out_decoders['real'][..., p_real:(2*p_real)]).reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy() + 0.001}
      params_x['real'] = {'mean': out_decoders['real'][..., :p_real].reshape([niw,batch_size,p_real]).detach().cpu().data.numpy(), 'scale': torch.nn.Softplus()(out_decoders['real'][..., p_real:(2*p_real)]).reshape([niw,batch_size,p_real]).detach().cpu().data.numpy() + 0.001}
    if exists_types[1]:
      out_decoders['count'] = decoders['count'](zgivenx_flat)
      # out_decoders['count'] = decoders['count'](torch.cat([zgivenx_flat, tiled_iota_x],1))
      p_xs['count'] = td.Normal(loc=out_decoders['count'][..., :p_count], scale=torch.nn.Softplus()(out_decoders['real'][..., p_count:(2*p_count)]) + 0.001)
      # params_x['count'] = {'mean': torch.mean(out_decoders['count'][..., :p_count].reshape([niw,batch_size,p_count]),0).detach().cpu().data.numpy(), 'scale': torch.mean(torch.nn.Softplus()(out_decoders['count'][..., p_count:(2*p_count)]).reshape([niw,batch_size,p_count]),0).detach().cpu().data.numpy() + 0.001}
      params_x['count'] = {'mean': out_decoders['count'][..., :p_count].reshape([niw,batch_size,p_count]).detach().cpu().data.numpy(), 'scale': torch.nn.Softplus()(out_decoders['count'][..., p_count:(2*p_count)]).reshape([niw,batch_size,p_count]).detach().cpu().data.numpy() + 0.001}
    if exists_types[2]:
      out_decoders['pos'] = decoders['pos'](zgivenx_flat)
      # out_decoders['pos'] = decoders['pos'](torch.cat([zgivenx_flat, tiled_iota_x],1))
      p_xs['pos'] = td.LogNormal(loc=out_decoders['pos'][..., :p_pos], scale=torch.nn.Softplus()(out_decoders['pos'][..., p_pos:(2*p_pos)]) + 0.001)
      # params_x['pos'] = {'mean': torch.mean(out_decoders['pos'][..., :p_pos].reshape([niw,batch_size,p_pos]),0).detach().cpu().data.numpy(), 'scale': torch.mean(torch.nn.Softplus()(out_decoders['pos'][..., p_pos:(2*p_pos)]).reshape([niw,batch_size,p_pos]),0).detach().cpu().data.numpy() + 0.001}
      params_x['pos'] = {'mean': out_decoders['pos'][..., :p_pos].reshape([niw,batch_size,p_pos]).detach().cpu().data.numpy(), 'scale': torch.nn.Softplus()(out_decoders['pos'][..., p_pos:(2*p_pos)]).reshape([niw,batch_size,p_pos]).detach().cpu().data.numpy() + 0.001}
    if exists_types[3]:
      for ii in range(0,p_cat):
        out_decoders['cat'].append( torch.clamp(torch.nn.Softmax(dim=1)(decoders['cat'][ii](zgivenx_flat)), min=0.0001, max=0.9999))
        # out_decoders['cat'].append( torch.clamp(torch.nn.Softmax(dim=1)(decoders['cat'][ii](torch.cat([zgivenx_flat, tiled_iota_x],1))), min=0.0001, max=0.9999))
        p_xs['cat'].append(td.OneHotCategorical(probs=out_decoders['cat'][ii]))
        # p_xs['cat'].append(td.RelaxedOneHotCategorical(temperature=temp, probs=out_decoders['cat'][ii]))
        # params_x['cat'].append(torch.mean(out_decoders['cat'][ii].reshape([niw,batch_size,-1]),0).detach().cpu().data.numpy())
        params_x['cat'].append(out_decoders['cat'][ii].reshape([niw,batch_size,-1]).detach().cpu().data.numpy())
    
    xm_flat = torch.ones([M*niw*batch_size,p]).cuda()
    xogivenz_flat = tiled_tiled_iota_x*tiledtiledmask
    
    ########## NEED TO SAMPLE M TIMES ############
    
    ## encoder_xr ##      p(xm|z,r)
    # out_encoder_xr = encoder_xr(torch.cat([tiled_iota_x,zgivenx_flat,tiledmask],1))  # samp_r may be pr or r when Gibbs
    out_encoders_xr = {}; out_encoders_xr['cat'] = []; q_xs={}; q_xs['cat'] = []
    params_xr={}; params_xr['cat']=[]
    if exists_types[0]:
      if ignorable:out_encoders_xr['real'] = encoders_xr['real'](torch.cat([tiled_iota_x,zgivenx_flat],1))
      else: out_encoders_xr['real'] = encoders_xr['real'](torch.cat([tiled_iota_x,zgivenx_flat,tiledmask],1))
      q_xs['real'] = td.Normal(loc=out_encoders_xr['real'][..., :p_real], scale=torch.nn.Softplus()(out_encoders_xr['real'][..., p_real:(2*p_real)]) + 0.001)
      # params_xr['real'] = {'mean': torch.mean(out_encoders_xr['real'][..., :p_real].reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy(), 'scale': torch.mean(torch.nn.Softplus()(out_encoders_xr['real'][..., p_real:(2*p_real)]).reshape([niw,batch_size,p_real]),0).detach().cpu().data.numpy() + 0.001}
      params_xr['real'] = {'mean': out_encoders_xr['real'][..., :p_real].reshape([niw,batch_size,p_real]).detach().cpu().data.numpy(), 'scale': torch.nn.Softplus()(out_encoders_xr['real'][..., p_real:(2*p_real)]).reshape([niw,batch_size,p_real]).detach().cpu().data.numpy() + 0.001}
    if exists_types[1]:
      if ignorable: out_encoders_xr['count'] = encoders_xr['count'](torch.cat([tiled_iota_x,zgivenx_flat],1))
      else: out_encoders_xr['count'] = encoders_xr['count'](torch.cat([tiled_iota_x,zgivenx_flat,tiledmask],1))
      q_xs['count'] = td.Normal(loc=out_encoders_xr['count'][..., :p_count], scale=torch.nn.Softplus()(out_encoders_xr['real'][..., p_count:(2*p_count)]) + 0.001)
      # params_xr['count'] = {'mean': torch.mean(out_encoders_xr['count'][..., :p_count].reshape([niw,batch_size,p_count]),0).detach().cpu().data.numpy(), 'scale': torch.mean(torch.nn.Softplus()(out_encoders_xr['count'][..., p_count:(2*p_count)]).reshape([niw,batch_size,p_count]),0).detach().cpu().data.numpy() + 0.001}
      params_xr['count'] = {'mean': out_encoders_xr['count'][..., :p_count].reshape([niw,batch_size,p_count]).detach().cpu().data.numpy(), 'scale': torch.nn.Softplus()(out_encoders_xr['count'][..., p_count:(2*p_count)]).reshape([niw,batch_size,p_count]).detach().cpu().data.numpy() + 0.001}
    if exists_types[2]:
      if ignorable: out_encoders_xr['pos'] = encoders_xr['pos'](torch.cat([tiled_iota_x,zgivenx_flat],1))
      else: out_encoders_xr['pos'] = encoders_xr['pos'](torch.cat([tiled_iota_x,zgivenx_flat,tiledmask],1))
      q_xs['pos'] = td.LogNormal(loc=out_encoders_xr['pos'][..., :p_pos], scale=torch.nn.Softplus()(out_encoders_xr['pos'][..., p_pos:(2*p_pos)]) + 0.001)
      # params_xr['pos'] = {'mean': torch.mean(out_encoders_xr['pos'][..., :p_pos].reshape([niw,batch_size,p_pos]),0).detach().cpu().data.numpy(), 'scale': torch.mean(torch.nn.Softplus()(out_encoders_xr['pos'][..., p_pos:(2*p_pos)]).reshape([niw,batch_size,p_pos]),0).detach().cpu().data.numpy() + 0.001}
      params_xr['pos'] = {'mean': out_encoders_xr['pos'][..., :p_pos].reshape([niw,batch_size,p_pos]).detach().cpu().data.numpy(), 'scale': torch.nn.Softplus()(out_encoders_xr['pos'][..., p_pos:(2*p_pos)]).reshape([niw,batch_size,p_pos]).detach().cpu().data.numpy() + 0.001}
    if exists_types[3]:
      for ii in range(0,p_cat):
        if ignorable: out_encoders_xr['cat'].append( torch.clamp(torch.nn.Softmax(dim=1)(encoders_xr['cat'][ii](torch.cat([tiled_iota_x,zgivenx_flat],1))), min=0.0001, max=0.9999) )
        else: out_encoders_xr['cat'].append( torch.clamp(torch.nn.Softmax(dim=1)(encoders_xr['cat'][ii](torch.cat([tiled_iota_x,zgivenx_flat,tiledmask],1))), min=0.0001, max=0.9999) )
        q_xs['cat'].append(td.RelaxedOneHotCategorical(temperature=temp, probs=out_encoders_xr['cat'][ii]))
        params_xr['cat'].append(out_encoders_xr['cat'][ii].reshape([niw,batch_size,-1]).detach().cpu().data.numpy())
        
      
    # if draw_xmiss, draw missing values from q(xm|...)
    if draw_xmiss:
      # print(ids_real)
      # print(q_xs['real'].rsample([M]).shape)
      if exists_types[0]: xm_flat[:,ids_real] = q_xs['real'].rsample([M]).reshape([M*niw*batch_size,-1])
      if exists_types[1]: xm_flat[:,ids_count] = q_xs['count'].rsample([M]).reshape([M*niw*batch_size,-1])
      if exists_types[2]: xm_flat[:,ids_pos] = q_xs['pos'].rsample([M]).reshape([M*niw*batch_size,-1])
      if exists_types[3]:
        for ii in range(0,p_cat):
          if ii==0: C0=0; C1=int(Cs[ii])
          else: C0=C1; C1=C0+int(Cs[ii])
          xm_flat[:, (p_real+p_count+p_pos+C0):(p_real+p_count+p_pos+C1)] = q_xs['cat'][ii].rsample([M]).reshape([M*niw*batch_size, int(Cs[ii])])
      xmgivenz_flat = xm_flat*(1-tiledtiledmask)
      xincluded = xogivenz_flat + xmgivenz_flat
      xincluded[:,ids_cat] = torch.clamp(xincluded[:,ids_cat], min=0.0001, max=0.9999)
    else:
      xincluded = tiled_tiled_iota_xfull
    
    if not ignorable:  
      ## DECODER_R ##
      # p(r|x)
      logits_Missing = torch.zeros(M*niw*batch_size, p0).cuda()
      if learn_r:
        if (not covars):
          if (rdeponz): out_decoder_r = decoder_r(torch.cat([torch.Tensor.repeat(zgivenx_flat,[M,1]), xincluded[:,covars_r==1]],1))
          else: out_decoder_r = decoder_r(xincluded[:,covars_r==1])
        else:
          if (rdeponz): out_decoder_r = decoder_r(torch.cat([torch.Tensor.repeat(zgivenx_flat,[M,1]), xincluded[:,covars_r==1], tiled_tiled_covars_miss],1))
          else: out_decoder_r = decoder_r(torch.cat([xincluded[:,covars_r==1], tiled_tiled_covars_miss],1))
        #logits_Missing = out_decoder_r[..., :p]
        logits_Missing[:,~full_obs_ids] = out_decoder_r[..., :(p_miss)]
        logits_Missing[:,full_obs_ids] =  torch.Tensor(float("Inf")*torch.ones(M*niw*batch_size, p0-p_miss)).cuda()          #################### NEW
      else:
        logits_Missing = torch.Tensor(float("Inf")*torch.ones(M*niw*batch_size, p0)).cuda()
        # logits_Missing[:,covars_r==1] = phi0 + torch.sum(phi*(xincluded[:,covars_r==1]),1).reshape(M*niw*batch_size, pr)
        logits_Missing[:,~full_obs_ids] = torch.Tensor(phi0).cuda() + torch.sum(phi*(xincluded[:,covars_r==1]),1).reshape(M*niw*batch_size, p_miss)
      prob_Missing = torch.nn.Sigmoid()(logits_Missing)
      # params_r={'probs':torch.mean(torch.mean(prob_Missing.reshape([M,niw,batch_size,p0]),0),0).detach().cpu().data.numpy()}
      params_r={'probs': prob_Missing.reshape([M,niw,batch_size,p0]).detach().cpu().data.numpy()}
    
      p_rgivenx = td.Bernoulli(probs=prob_Missing)
    else:
      p_rgivenx=None; params_r=None
    
    ## OUTPUTS ##
    # if dec_distrib=="Normal":
    #   params_x={'mean':all_means_obs_model,'sd':all_scales_obs_model}
    # elif dec_distrib=="StudentT":
    #   params_x={'mean':all_means_obs_model,'sd':all_scales_obs_model,'df':all_degfreedom_obs_model}
    # if not ignorable:
    #   params_r={'probs':prob_Missing.detach().cpu().data.numpy()}
    #   if dec_distrib=="Normal":
    #     params_xr={'mean':all_means_miss_model,'sd':all_scales_miss_model}
    #   elif dec_distrib=="StudentT":
    #     params_xr={'mean':all_means_miss_model,'sd':all_scales_miss_model,'df':all_degfreedom_miss_model}
    # else:
    #   params_r = None
    #   params_xr = None
    # params_z={'mean':out_encoder[..., :d], 'sd':torch.nn.Softplus()(out_encoder[..., d:(2*d)])+0.001}
    # return p_rgivenx, pxgivenz, qxgivenzr, p_z, q_zgivenxobs, params_x, params_xr, params_r, params_z, zgivenx_flat, xgivenzr
    return p_rgivenx, p_xs, q_xs, p_z, q_zgivenxobs, params_x, params_xr, params_r, params_z, zgivenx_flat, xincluded

  ############################## END FORWARD #####################

  # Functions to calculate nimiwae loss and impute using nimiwae
  def nimiwae_loss(iota_xfull, iota_x, mask, mask0, covar_miss, temp):
    iota_x[:,ids_cat] = torch.clamp(iota_x[:,ids_cat], min=0.0001, max=0.9999)
    #mask[mask==1]=0.999; mask[mask==0]=0.001
    batch_size = iota_x.shape[0]
    tiledmask = torch.Tensor.repeat(mask,[K,1]).cuda()
    tiledtiledmask = torch.Tensor.repeat(tiledmask,[M,1]).cuda()
    tiled_iota_x = torch.Tensor.repeat(iota_x,[K,1]).cuda()
    #tiled_tiled_iota_x = torch.Tensor.repeat(tiled_iota_x,[M,1]).cuda()
    if (not draw_xmiss) and (not ignorable): tiled_iota_xfull = torch.Tensor.repeat(iota_xfull,[K,1]).cuda(); tiled_tiled_iota_xfull = torch.Tensor.repeat(tiled_iota_xfull,[M,1]).cuda()
    else: tiled_iota_xfull = None; tiled_tiled_iota_xfull = None
    # tiled_iota_xfull = None; tiled_tiled_iota_xfull = None
    
    if covars: tiled_tiled_covars_miss = torch.Tensor.repeat(torch.Tensor.repeat(covar_miss,[K,1]),[M,1])
    else: tiled_tiled_covars_miss=None

    #tiled_probs_Missing = torch.Tensor.repeat(prM,[K,1]).cuda()
    # concat batch data with corresponding mask. iota_x: (n_batch x p). concatenation: (n_batch x 2p)

    # p_rgivenx, pxgivenz, qxgivenzr, p_z, q_zgivenxobs, params_x, params_xr, params_r, params_z, zgivenx_flat, xgivenzr = forward(K, iota_xfull, iota_x, mask, batch_size, tiledmask, tiled_iota_x, tiled_iota_xfull, tiled_tiled_covars_miss, temp)
    p_rgivenx, p_xs, q_xs, p_z, q_zgivenxobs, params_x, params_xr, params_r, params_z, zgivenx_flat, xincluded = forward(K, iota_xfull, iota_x, mask, batch_size, tiledmask, tiled_iota_x, tiled_iota_xfull, tiled_tiled_covars_miss, temp)
    
    # change pxgivenz --> p_xs
    # change qxgivenzr --> q_xs
    # change xgivenzr --> xincluded
    
    ## COMPUTE LOG PROBABILITIES ##
    if not ignorable:
      tiledtiledmask0 = torch.Tensor.repeat(torch.Tensor.repeat(mask0,[K,1]),[M,1]).cuda()
      all_logprgivenx = p_rgivenx.log_prob(tiledtiledmask0)  # M*niw*bs x p0
      logprgivenx = torch.sum(torch.sum(all_logprgivenx,1).reshape([M,K*batch_size]),0).reshape([K,batch_size])
      sum_logpr = np.sum(logprgivenx.cpu().data.numpy())
    else:
      # if ignorably missing, no p(r|x), no q(xm|z,r), and no p(xm|z)
      all_logprgivenx = torch.zeros([M*K*batch_size,p]).cuda(); logprgivenx=torch.zeros([1]).cuda(); sum_logpr=np.zeros(1)
    
    logqxmissgivenzr = {}; logpxmissgivenz = {}
    # can combine real, count, pos if we're continuing to treat them exactly the same (would save computation time)
    if exists_types[0]:
      logqxmissgivenzr['real'] = torch.sum(q_xs['real'].log_prob(xincluded[:,ids_real].reshape([M,K*batch_size,p_real])).reshape([M*K*batch_size,p_real])*(1-tiledtiledmask[:,ids_real]),axis=1).reshape([M,K*batch_size])
      logpxmissgivenz['real'] = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real].reshape([M,K*batch_size,p_real])).reshape([M*K*batch_size,p_real])*(1-tiledtiledmask[:,ids_real]),axis=1).reshape([M,K*batch_size])
    if exists_types[1]:
      logqxmissgivenzr['count'] = torch.sum(q_xs['count'].log_prob(xincluded[:,ids_count].reshape([M,K*batch_size,p_count])).reshape([M*K*batch_size,p_count])*(1-tiledtiledmask[:,ids_count]),axis=1).reshape([M,K*batch_size])
      logpxmissgivenz['count'] = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count].reshape([M,K*batch_size,p_count])).reshape([M*K*batch_size,p_count])*(1-tiledtiledmask[:,ids_count]),axis=1).reshape([M,K*batch_size])
    if exists_types[2]:
      logqxmissgivenzr['pos'] = torch.sum(q_xs['pos'].log_prob(xincluded[:,ids_pos].reshape([M,K*batch_size,p_pos])).reshape([M*K*batch_size,p_pos])*(1-tiledtiledmask[:,ids_pos]),axis=1).reshape([M,K*batch_size])
      logpxmissgivenz['pos'] = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos].reshape([M,K*batch_size,p_pos])).reshape([M*K*batch_size,p_pos])*(1-tiledtiledmask[:,ids_pos]),axis=1).reshape([M,K*batch_size])
    if exists_types[3]:
      for ii in range(0,p_cat):
        if ii==0: C0=0; C1=int(Cs[ii])
        else: C0=C1; C1=C0+int(Cs[ii])
        
        logqxs = q_xs['cat'][ii].log_prob(xincluded[:,(p_real+p_count+p_pos+C0):(p_real+p_count+p_pos+C1)].reshape([M,K*batch_size,-1])).reshape([M*K*batch_size])
        logpxs = p_xs['cat'][ii].log_prob(xincluded[:,(p_real+p_count+p_pos+C0):(p_real+p_count+p_pos+C1)].reshape([M,K*batch_size,-1])).reshape([M*K*batch_size])
        if ii==0:
          
          logqxmissgivenzr['cat'] = (logqxs*(1-tiledtiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([M,K*batch_size])
          logpxmissgivenz['cat'] = (logpxs*(1-tiledtiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([M,K*batch_size])
        else:
          logqxmissgivenzr['cat'] = logqxmissgivenzr['cat'] + (logqxs*(1-tiledtiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([M,K*batch_size])
          logpxmissgivenz['cat'] = logpxmissgivenz['cat'] + (logpxs*(1-tiledtiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([M,K*batch_size])
        
    logpxmissgivenzsum = torch.zeros([M,K*batch_size]).cuda(); logqxmissgivenzrsum = torch.zeros([M,K*batch_size]).cuda()
    if exists_types[0]: logpxmissgivenzsum = logpxmissgivenzsum + logpxmissgivenz['real']; logqxmissgivenzrsum = logqxmissgivenzrsum + logqxmissgivenzr['real']
    if exists_types[1]: logpxmissgivenzsum = logpxmissgivenzsum + logpxmissgivenz['count']; logqxmissgivenzrsum = logqxmissgivenzrsum + logqxmissgivenzr['count']
    if exists_types[2]: logpxmissgivenzsum = logpxmissgivenzsum + logpxmissgivenz['pos']; logqxmissgivenzrsum = logqxmissgivenzrsum + logqxmissgivenzr['pos']
    if exists_types[3]: logpxmissgivenzsum = logpxmissgivenzsum + logpxmissgivenz['cat']; logqxmissgivenzrsum = logqxmissgivenzrsum + logqxmissgivenzr['cat']
    
    KL2 = torch.sum((logpxmissgivenzsum - logqxmissgivenzrsum),axis=0).reshape([K,batch_size])
    
    
    # all_log_pxgivenz = pxgivenz.log_prob(tiled_iota_x)
    # logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])
    # sum_logpxobs = np.sum(logpxobsgivenz.cpu().data.numpy())
    
    all_logpxobsgivenz = {}; logpxobsgivenz = torch.zeros([K,batch_size]).cuda()
    if exists_types[0]:
      all_logpxobsgivenz['real'] = torch.sum(p_xs['real'].log_prob(tiled_iota_x[:,ids_real]) * tiledmask[:,ids_real],1).reshape([K,batch_size])
      logpxobsgivenz = logpxobsgivenz + all_logpxobsgivenz['real']
    if exists_types[1]:
      all_logpxobsgivenz['count'] = torch.sum(p_xs['count'].log_prob(tiled_iota_x[:,ids_count]) * tiledmask[:,ids_count],1).reshape([K,batch_size])
      logpxobsgivenz = logpxobsgivenz + all_logpxobsgivenz['count']
    if exists_types[2]:
      all_logpxobsgivenz['pos'] = torch.sum(p_xs['pos'].log_prob(tiled_iota_x[:,ids_pos]) * tiledmask[:,ids_pos],1).reshape([K,batch_size])
      logpxobsgivenz = logpxobsgivenz + all_logpxobsgivenz['pos']
    if exists_types[3]:
      for ii in range(0,p_cat):
        if ii==0: C0=0; C1=int(Cs[ii])
        else: C0=C1; C1=C0+int(Cs[ii])
        logpxs = p_xs['cat'][ii].log_prob(tiled_iota_x[:,(p_real+p_count+p_pos+C0):(p_real+p_count+p_pos+C1)]).reshape([K*batch_size])
        if ii==0: all_logpxobsgivenz['cat'] = (logpxs*(tiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([K,batch_size])
        else: all_logpxobsgivenz['cat'] = all_logpxobsgivenz['cat'] + (logpxs*(tiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([K,batch_size])
        
      logpxobsgivenz = logpxobsgivenz + all_logpxobsgivenz['cat']
    
    sum_logpxobs = np.sum(logpxobsgivenz.cpu().data.numpy())
    # logpz = p_z.log_prob(zgivenx)      # p_z: bs x d, zgivenx: niw x bs x d
    logpz = p_z.log_prob(zgivenx_flat.reshape([K,batch_size,d]))      # p_z: bs x d, zgivenx: niw x bs x d
    sum_logpz = np.sum(logpz.cpu().data.numpy())

    logqz = q_zgivenxobs.log_prob(zgivenx_flat.reshape([K,batch_size,d]))
    sum_logqz = np.sum(logqz.cpu().data.numpy())
    
    # beta=1.2  # beta-VAE. should tune this too if i use this
    # KL = beta*torch.sum((logpz - logqz).reshape([K*batch_size,-1]),axis=1).reshape([K,batch_size])    # actually (-KL)
    KL = torch.sum((logpz - logqz).reshape([K*batch_size,-1]),axis=1).reshape([K,batch_size])    # actually (-KL)

    if arch=="VAE":
      ## VAE NEGATIVE LOG-LIKE ## logpxobsgivenz, KL: (K x bs), logprgivenx, KL2: (K x bs)
      neg_bound = -torch.sum(logpxobsgivenz + KL) - (1/M)*torch.sum(logprgivenx) - (1/M)*torch.sum(KL2)   # need to do: f(X) = X/(K*bs)
      #neg_bound = -torch.mean(logpxobsgivenz + KL) - (1/M)*torch.mean(logprgivenx) - (1/M)*torch.mean(KL2)   # full neg_bound (averaged over K & bs)
    elif arch=="IWAE":
      ## IWAE NEGATIVE LOG-LIKE ##  L1, L2: (K x bs); logsumexp(L1+L2,0): (bs x 1)
      # L2: logsumexp_l=1^M [log { p(xm|z)*p(r|xm,z)/q(xm|z) }]
      # L1: log{ p(xo|z)p(z)/q(z|xo,r) }
      # LB = log(1/K) + log(1/M) + logsumexp_k=1^K [L1 + L2]

      L2 = torch.logsumexp(torch.sum(all_logprgivenx,1).reshape([M,K*batch_size]) + logpxmissgivenzsum - logqxmissgivenzrsum,axis=0).reshape([K,batch_size])
      L1 = KL + logpxobsgivenz
      
      ####neg_bound = np.log(K) + np.log(M) - torch.mean(torch.logsumexp(logpxobsgivenz + logprgivenx + KL + KL2,0)) # need to check this. see if 1/M or M* term is missing somewhere
      neg_bound = - torch.sum(torch.logsumexp(L1+L2,0))   # need to do f(X) = X/(bs) + log(K) + log(M)
      #neg_bound = np.log(K) + np.log(M) - torch.mean(torch.logsumexp(L1 + L2,0))   # full neg_bound (averaged over K & bs)
    

    return{'neg_bound':neg_bound, 'params_x': params_x, 'params_xr': params_xr, 'params_r': params_r, 'params_z': params_z, 'sum_logpz': sum_logpz,'sum_logqz': sum_logqz,'sum_logpr': sum_logpr, 'sum_logpxobs': sum_logpxobs}
  
  def nimiwae_impute(iota_xfull,iota_x, mask, mask0, covar_miss,L, temp):
    iota_x[:,ids_cat] = torch.clamp(iota_x[:,ids_cat], min=0.0001, max=0.9999)
    batch_size = iota_x.shape[0]
    tiledmask = torch.Tensor.repeat(mask,[L,1]).cuda()
    tiledtiledmask = torch.Tensor.repeat(tiledmask,[M,1]).cuda()
    tiled_iota_x = torch.Tensor.repeat(iota_x,[L,1]).cuda()
    #tiled_tiled_iota_x = torch.Tensor.repeat(tiled_iota_x,[M,1]).cuda()
    if (not draw_xmiss) and not ignorable: tiled_iota_xfull = torch.Tensor.repeat(iota_xfull,[L,1]).cuda(); tiled_tiled_iota_xfull = torch.Tensor.repeat(tiled_iota_xfull,[M,1]).cuda()
    else: tiled_iota_xfull = None; tiled_tiled_iota_xfull = None
    # tiled_iota_xfull = None; tiled_tiled_iota_xfull = None
    
    if covars: tiled_tiled_covars_miss = torch.Tensor.repeat(torch.Tensor.repeat(covar_miss,[L,1]),[M,1])
    else: tiled_tiled_covars_miss = None

    #tiled_probs_Missing = torch.Tensor.repeat(prM,[L,1]).cuda()
    # concat batch data with corresponding mask. iota_x: (n_batch x p). concatenation: (n_batch x 2p)

    p_rgivenx, p_xs, q_xs, p_z, q_zgivenxobs, params_x, params_xr, params_r, params_z, zgivenx_flat, xincluded = forward(L, iota_xfull, iota_x, mask, batch_size, tiledmask, tiled_iota_x, tiled_iota_xfull, tiled_tiled_covars_miss, temp)

    ## COMPUTE LOG PROBABILITIES ##
    if not ignorable:
      # all_logprgivenx = p_rgivenx.log_prob(tiledtiledmask)  # M*niw*bs x p
      tiledtiledmask0 = torch.Tensor.repeat(torch.Tensor.repeat(mask0,[L,1]),[M,1]).cuda()
      all_logprgivenx = p_rgivenx.log_prob(tiledtiledmask0)  # M*niw*bs x p0
      # sum across p features --> M*niw*bs --> sum over M --> niw*bs
      logprgivenx = torch.sum(all_logprgivenx,1).reshape([M,L*batch_size])
      logprgivenx1 = torch.sum(logprgivenx,0).reshape([L,batch_size])
      sum_logpr = np.sum(logprgivenx.cpu().data.numpy())
    else:
      # if ignorably missing, no p(r|x), no q(xm|z,r), and no p(xm|z)
      all_logprgivenx = torch.zeros([M*L*batch_size,p]).cuda(); logprgivenx=torch.zeros([1]).cuda(); sum_logpr=np.zeros(1)

    logqxmissgivenzr = {}; logpxmissgivenz = {}
    # can combine real, count, pos if we're continuing to treat them exactly the same (would save computation time)
    if exists_types[0]:
      logqxmissgivenzr['real'] = torch.sum(q_xs['real'].log_prob(xincluded[:,ids_real].reshape([M,L*batch_size,p_real])).reshape([M*L*batch_size,p_real])*(1-tiledtiledmask[:,ids_real]),axis=1).reshape([M,L*batch_size])
      logpxmissgivenz['real'] = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real].reshape([M,L*batch_size,p_real])).reshape([M*L*batch_size,p_real])*(1-tiledtiledmask[:,ids_real]),axis=1).reshape([M,L*batch_size])
    if exists_types[1]:
      logqxmissgivenzr['count'] = torch.sum(q_xs['count'].log_prob(xincluded[:,ids_count].reshape([M,L*batch_size,p_count])).reshape([M*L*batch_size,p_count])*(1-tiledtiledmask[:,ids_count]),axis=1).reshape([M,L*batch_size])
      logpxmissgivenz['count'] = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count].reshape([M,L*batch_size,p_count])).reshape([M*L*batch_size,p_count])*(1-tiledtiledmask[:,ids_count]),axis=1).reshape([M,L*batch_size])
    if exists_types[2]:
      logqxmissgivenzr['pos'] = torch.sum(q_xs['pos'].log_prob(xincluded[:,ids_pos].reshape([M,L*batch_size,p_pos])).reshape([M*L*batch_size,p_pos])*(1-tiledtiledmask[:,ids_pos]),axis=1).reshape([M,L*batch_size])
      logpxmissgivenz['pos'] = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos].reshape([M,L*batch_size,p_pos])).reshape([M*L*batch_size,p_pos])*(1-tiledtiledmask[:,ids_pos]),axis=1).reshape([M,L*batch_size])
    if exists_types[3]:
      for ii in range(0,p_cat):
        if ii==0: C0=0; C1=int(Cs[ii])
        else: C0=C1; C1=C0+int(Cs[ii])
        
        logqxs = q_xs['cat'][ii].log_prob(xincluded[:,(p_real+p_count+p_pos+C0):(p_real+p_count+p_pos+C1)].reshape([M,L*batch_size,-1])).reshape([M*L*batch_size])
        logpxs = p_xs['cat'][ii].log_prob(xincluded[:,(p_real+p_count+p_pos+C0):(p_real+p_count+p_pos+C1)].reshape([M,L*batch_size,-1])).reshape([M*L*batch_size])
        if ii==0:
          
          logqxmissgivenzr['cat'] = (logqxs*(1-tiledtiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([M,L*batch_size])
          logpxmissgivenz['cat'] = (logpxs*(1-tiledtiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([M,L*batch_size])
        else:
          logqxmissgivenzr['cat'] = logqxmissgivenzr['cat'] + (logqxs*(1-tiledtiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([M,L*batch_size])
          logpxmissgivenz['cat'] = logpxmissgivenz['cat'] + (logpxs*(1-tiledtiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([M,L*batch_size])
        
        
    logpxmissgivenzsum = torch.zeros([M,L*batch_size]).cuda(); logqxmissgivenzrsum = torch.zeros([M,L*batch_size]).cuda()
    if exists_types[0]: logpxmissgivenzsum = logpxmissgivenzsum + logpxmissgivenz['real']; logqxmissgivenzrsum = logqxmissgivenzrsum + logqxmissgivenzr['real']
    if exists_types[1]: logpxmissgivenzsum = logpxmissgivenzsum + logpxmissgivenz['count']; logqxmissgivenzrsum = logqxmissgivenzrsum + logqxmissgivenzr['count']
    if exists_types[2]: logpxmissgivenzsum = logpxmissgivenzsum + logpxmissgivenz['pos']; logqxmissgivenzrsum = logqxmissgivenzrsum + logqxmissgivenzr['pos']
    if exists_types[3]: logpxmissgivenzsum = logpxmissgivenzsum + logpxmissgivenz['cat']; logqxmissgivenzrsum = logqxmissgivenzrsum + logqxmissgivenzr['cat']
    
    logpxmissgivenzsum1 = torch.sum(logpxmissgivenzsum,0).reshape([L, batch_size])
    logqxmissgivenzrsum1 = torch.sum(logqxmissgivenzrsum,0).reshape([L, batch_size])
    
    all_logpxobsgivenz = {}; logpxobsgivenz = torch.zeros([L,batch_size]).cuda()
    if exists_types[0]:
      all_logpxobsgivenz['real'] = torch.sum(p_xs['real'].log_prob(tiled_iota_x[:,ids_real]) * tiledmask[:,ids_real],1).reshape([L,batch_size])
      logpxobsgivenz = logpxobsgivenz + all_logpxobsgivenz['real']
    if exists_types[1]:
      all_logpxobsgivenz['count'] = torch.sum(p_xs['count'].log_prob(tiled_iota_x[:,ids_count]) * tiledmask[:,ids_count],1).reshape([L,batch_size])
      logpxobsgivenz = logpxobsgivenz + all_logpxobsgivenz['count']
    if exists_types[2]:
      all_logpxobsgivenz['pos'] = torch.sum(p_xs['pos'].log_prob(tiled_iota_x[:,ids_pos]) * tiledmask[:,ids_pos],1).reshape([L,batch_size])
      logpxobsgivenz = logpxobsgivenz + all_logpxobsgivenz['pos']
    if exists_types[3]:
      for ii in range(0,p_cat):
        if ii==0: C0=0; C1=int(Cs[ii])
        else: C0=C1; C1=C0+int(Cs[ii])
        logpxs = p_xs['cat'][ii].log_prob(tiled_iota_x[:,(p_real+p_count+p_pos+C0):(p_real+p_count+p_pos+C1)]).reshape([L*batch_size])
        if ii==0: all_logpxobsgivenz['cat'] = (logpxs*(tiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([L,batch_size])
        else: all_logpxobsgivenz['cat'] = all_logpxobsgivenz['cat'] + (logpxs*(tiledmask[:,(p_real+p_count+p_pos+C0)])).reshape([L,batch_size])
        
      logpxobsgivenz = logpxobsgivenz + all_logpxobsgivenz['cat']
    
    sum_logpxobs = np.sum(logpxobsgivenz.cpu().data.numpy())
    # logpz = p_z.log_prob(zgivenx)      # p_z: bs x d, zgivenx: niw x bs x d
    logpz = p_z.log_prob(zgivenx_flat.reshape([L,batch_size,d]))      # p_z: bs x d, zgivenx: niw x bs x d
    sum_logpz = np.sum(logpz.cpu().data.numpy())

    logqz = q_zgivenxobs.log_prob(zgivenx_flat.reshape([L,batch_size,d]))
    sum_logqz = np.sum(logqz.cpu().data.numpy())
    
    
    ## SELF-NORMALIZING IMPORTANCE WEIGHTS, USING SAMPLES OF Xm AND Z ##
    # if ignorable:
    #   imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logqz,0) # these are w_1,....,w_L for all observations in the batch
    #   xms = torch.mean(xincluded.reshape([M,L*batch_size,p]),0).reshape([L,batch_size,p])    # xincluded: [M*L*batch_size, p]
    # else:
    #   imp_weights = torch.nn.functional.softmax(torch.Tensor.repeat(logpxobsgivenz,[M,1]) + logpxmissgivenzsum.reshape([M*L,batch_size]) + torch.Tensor.repeat(logpz,[M,1,1]).reshape([M*L,batch_size]) + logprgivenx.reshape([M*L,batch_size]) - torch.Tensor.repeat(logqz,[M,1,1]).reshape([M*L,batch_size]) - logqxmissgivenzrsum.reshape([M*L,batch_size]),0) # these are w_1,....,w_L for all observations in the batch
    #   xms = xincluded.reshape([M*L,batch_size,p])    # xincluded: [M*L*batch_size, p]
    if ignorable:
      imp_weights = torch.nn.functional.softmax(torch.Tensor.repeat(logpxobsgivenz,[M,1]) + logpxmissgivenzsum.reshape([M*L,batch_size]) + torch.Tensor.repeat(logpz,[M,1,1]).reshape([M*L,batch_size]) - torch.Tensor.repeat(logqz,[M,1,1]).reshape([M*L,batch_size]) - logqxmissgivenzrsum.reshape([M*L,batch_size]),0) # these are w_1,....,w_L for all observations in the batch
    else:
      imp_weights = torch.nn.functional.softmax(torch.Tensor.repeat(logpxobsgivenz,[M,1]) + logpxmissgivenzsum.reshape([M*L,batch_size]) + torch.Tensor.repeat(logpz,[M,1,1]).reshape([M*L,batch_size]) + logprgivenx.reshape([M*L,batch_size]) - torch.Tensor.repeat(logqz,[M,1,1]).reshape([M*L,batch_size]) - logqxmissgivenzrsum.reshape([M*L,batch_size]),0) # these are w_1,....,w_L for all observations in the batch
    
    xms = xincluded.reshape([M*L,batch_size,p])    # xincluded: [M*L*batch_size, p]
    # print(xms)
    xm = torch.einsum('ki,kij->ij', imp_weights.float(), xms.float())
    
    return {'xm': xm.detach(), 'imp_weights': imp_weights.detach(), 'xms': xms.detach(),'zgivenx': zgivenx_flat.reshape([L,batch_size,d]).detach()}
  
  # initialize weights
  def weights_init(layer):
    if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
  
  # Define ADAM optimizer
  # if not ignorable:
  #   if learn_r:
  #     # params = list(encoder.parameters()) + list(encoder_xr.parameters()) + list(decoder_x.parameters()) + list(decoder_r.parameters())
  #     params = list(encoder.parameters()) + list(decoder_r.parameters())
  #     optimizer = optim.Adam(params,lr=lr, weight_decay=L2_weight)
  #   else:
  #     # params = list(encoder.parameters()) + list(encoder_xr.parameters()) + list(decoder_x.parameters())
  #     params = list(encoder.parameters())
  #     optimizer = optim.Adam(params,lr=lr, weight_decay=L2_weight)
  # else:
  #   # params = list(encoder.parameters()) + list(decoder_x.parameters())
  #   params = list(encoder.parameters())
  #   optimizer = optim.Adam(params,lr=lr, weight_decay=L2_weight)
    
  
  params = list(encoder.parameters())
  if exists_types[0]: params = params + list(decoders['real'].parameters()); decoders['real'].apply(weights_init)
  if exists_types[1]: params = params + list(decoders['count'].parameters()); decoders['count'].apply(weights_init)
  if exists_types[2]: params = params + list(decoders['pos'].parameters()); decoders['pos'].apply(weights_init)
  if exists_types[3]:
    for ii in range(0, p_cat):
      params = params + list(decoders['cat'][ii].parameters()); decoders['cat'][ii].apply(weights_init)
  
  if learn_r and not ignorable:
    params = params + list(decoder_r.parameters())
  if exists_types[0]: params = params + list(encoders_xr['real'].parameters()); encoders_xr['real'].apply(weights_init)
  if exists_types[1]: params = params + list(encoders_xr['count'].parameters()); encoders_xr['count'].apply(weights_init)
  if exists_types[2]: params = params + list(encoders_xr['pos'].parameters()); encoders_xr['pos'].apply(weights_init)
  if exists_types[3]:
    for ii in range(0, p_cat):
      params = params + list(encoders_xr['cat'][ii].parameters()); encoders_xr['cat'][ii].apply(weights_init)

  optimizer = optim.Adam(params, lr=lr, weight_decay=L2_weight)

  # Train and impute every 100 epochs
  nimiwae_loss_train=np.array([])
  mse_train_miss=np.array([])
  mse_train_obs=np.array([])
  mse_pr_epoch = np.array([])
  CEL_epoch=np.array([]) # Cross-entropy error
  xhat = np.copy(xhat_0) # This will be out imputed data matrix

  trace_ids = np.concatenate([np.where(Missing[:,0]==0)[0][0:2],np.where(Missing[:,0]==1)[0][0:2]])
  if (trace): print(xhat_0[trace_ids,0:min(4,p)])
  
  time_train=[]
  time_impute=[]
  NIMIWAE_LB_epoch=[]
  NIMIWAE_val_LB_epoch=[]
  sum_logpz_epoch =[]
  sum_logqz_epoch=[]
  sum_logpr_epoch=[]
  sum_logpxobs_epoch=[]

  # only assign xfull to cuda if it's necessary (save GPU ram)
  if (not draw_xmiss) and not ignorable: cuda_xfull = torch.from_numpy(xfull).float().cuda()
  else: cuda_xfull = None
  # cuda_xfull = None
  
  # initialize early stop criteria/variables
  #n_epochs_stop = 101   # number of epochs system can not improve consecutively before early stop
  early_stopped = False  # will be changed to True if early stop happens
  early_stop_epochs = n_epochs
  max_NIMIWAE_val_LB = float("-inf")  # initialize as this: first epoch val_LB will always replace
  # early_stop_check_epochs = 500001       # relative change in val_LB checked across this many epochs  #turned off
  early_stop_check_epochs = 101       # relative change in val_LB checked across this many epochs  #turned off
  early_stop_tol = 1e-4               # tolerance of change in val_LB across early_stop_check_epochs

  ## Initialize params
  #######################################################################
  ######## NEED TO SAVE params_x, params_z, params_r, params_xr #########
  #######################################################################
  params_x = {}; params_x['cat']=[]
  params_z = {'mean': np.empty([n,d]), 'scale': np.empty([n,d])}
  # params_r = {'probs': np.empty([n,p0])}
  params_r = {'probs': np.empty([M,niw,n,p0])}
  params_xr = {}; params_xr['cat']=[]
  
  if exists_types[0]:
    # params_x['real'] = {'mean': np.empty([n,p_real]), 'scale': np.empty([n,p_real])}
    # if not ignorable: params_xr['real'] = {'mean': np.empty([n,p_real]), 'scale': np.empty([n,p_real])}
    params_x['real'] = {'mean': np.empty([niw,n,p_real]), 'scale': np.empty([niw,n,p_real])}
    params_xr['real'] = {'mean': np.empty([niw,n,p_real]), 'scale': np.empty([niw,n,p_real])}
  if exists_types[1]:
    # params_x['count'] = {'mean': np.empty([n,p_count]), 'scale': np.empty([n,p_count])}
    # if not ignorable: params_xr['count'] = {'mean': np.empty([n,p_real]), 'scale': np.empty([n,p_real])}
    params_x['count'] = {'mean': np.empty([niw,n,p_count]), 'scale': np.empty([niw,n,p_count])}
    params_xr['count'] = {'mean': np.empty([niw,n,p_real]), 'scale': np.empty([niw,n,p_real])}
  if exists_types[2]:
    # params_x['pos'] = {'mean': np.empty([n,p_pos]), 'scale': np.empty([n,p_pos])}
    # if not ignorable: params_xr['pos'] = {'mean': np.empty([n,p_real]), 'scale': np.empty([n,p_real])}
    params_x['pos'] = {'mean': np.empty([niw,n,p_pos]), 'scale': np.empty([niw,n,p_pos])}
    params_xr['pos'] = {'mean': np.empty([niw,n,p_real]), 'scale': np.empty([niw,n,p_real])}
  if exists_types[3]:
    for ii in range(0, p_cat):
      # params_x['cat'].append(np.empty([n,int(Cs[ii])]))
      # if not ignorable: params_xr['cat'].append(np.empty([n,int(Cs[ii])]))
      params_x['cat'].append(np.empty([niw,n,int(Cs[ii])]))
      params_xr['cat'].append(np.empty([niw,n,int(Cs[ii])]))

  if train==1:
    if warm_start:
      encoder=saved_model['encoder']
      decoders=saved_model['decoders']
      encoders_xr=saved_model['encoders_xr']
      if (learn_r and not ignorable): decoder_r=saved_model['decoder_r']
    # Training+Imputing
    for ep in range(1,n_epochs):
      if ep % 10==0:
        print("Epoch " + str(ep))
      # print("Epoch " + str(ep))
      perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
      if (not draw_xmiss) and not ignorable: batches_full = np.array_split(xfull[perm,],n/bs)
      batches_data = np.array_split(xhat_0[perm,], n/bs)
      batches_mask = np.array_split(mask[perm,], n/bs)
      batches_mask0 = np.array_split(mask0[perm,], n/bs)
      if covars: batches_covar = np.array_split(covars_miss[perm,], n/bs)
      splits = np.array_split(perm,n/bs)
      batches_loss = []
      # loss_fits = []
      #'sum_logpz': sum_logpz,'sum_logqz': sum_logqz,'sum_logpr': sum_logpr, 'sum_logpxobs': sum_logpxobs
      sum_logpz=0; sum_logqz=0; sum_logpr=0; sum_logpxobs=0
      t0_train=time.time()
      
      for it in range(len(batches_data)):
        # print("minibatch " + str(it))
        if (not draw_xmiss) and not ignorable: b_full = torch.from_numpy(batches_full[it]).float().cuda()
        else: b_full = None
        # b_full = None
        b_data = torch.from_numpy(batches_data[it]).float().cuda()
        b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
        b_mask0 = torch.from_numpy(batches_mask0[it]).float().cuda()
        if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
        else: b_covar = None

        optimizer.zero_grad()
        encoder.zero_grad()
        if exists_types[0]: decoders['real'].zero_grad()
        if exists_types[1]: decoders['count'].zero_grad()
        if exists_types[2]: decoders['pos'].zero_grad()
        if exists_types[3]:
          for ii in range(0,p_cat):
            decoders['cat'][ii].zero_grad()
        if exists_types[0]: encoders_xr['real'].zero_grad()
        if exists_types[1]: encoders_xr['count'].zero_grad()
        if exists_types[2]: encoders_xr['pos'].zero_grad()
        if exists_types[3]:
          for ii in range(0,p_cat):
            encoders_xr['cat'][ii].zero_grad()
        if (learn_r and not ignorable): decoder_r.zero_grad()
        
        loss_fit = nimiwae_loss(iota_xfull=b_full, iota_x = b_data, mask = b_mask, mask0 = b_mask0, covar_miss = b_covar, temp=temp)
        loss = loss_fit['neg_bound']
        sum_logpz += loss_fit['sum_logpz']; sum_logqz += loss_fit['sum_logqz']; sum_logpr += loss_fit['sum_logpr']; sum_logpxobs += loss_fit['sum_logpxobs']
        
        temp_params_x, temp_params_xr, temp_params_r, temp_params_z = loss_fit['params_x'], loss_fit['params_xr'], loss_fit['params_r'], loss_fit['params_z']
        
        ##### SAVE ALL params_x, params_xr, params_r, params_z ##### (maybe do this every 100 iters --> save some time)
        if ep % 100 == 0:
          if exists_types[0]:
            # params_x['real']['mean'][splits[it],:] = temp_params_x['real']['mean']; params_x['real']['scale'][splits[it],:] = temp_params_x['real']['scale']
            # if not ignorable: params_xr['real']['mean'][splits[it],:] = temp_params_xr['real']['mean']; params_xr['real']['scale'][splits[it],:] = temp_params_xr['real']['scale']
            params_x['real']['mean'][:,splits[it],:] = temp_params_x['real']['mean']; params_x['real']['scale'][:,splits[it],:] = temp_params_x['real']['scale']
            params_xr['real']['mean'][:,splits[it],:] = temp_params_xr['real']['mean']; params_xr['real']['scale'][:,splits[it],:] = temp_params_xr['real']['scale']
          if exists_types[1]:
            # params_x['count']['mean'][splits[it],:] = temp_params_x['count']['mean']; params_x['count']['scale'][splits[it],:] = temp_params_x['count']['scale']
            # if not ignorable: params_xr['count']['mean'][splits[it],:] = temp_params_xr['count']['mean']; params_xr['count']['scale'][splits[it],:] = temp_params_xr['count']['scale']
            params_x['count']['mean'][:,splits[it],:] = temp_params_x['count']['mean']; params_x['count']['scale'][:,splits[it],:] = temp_params_x['count']['scale']
            params_xr['count']['mean'][:,splits[it],:] = temp_params_xr['count']['mean']; params_xr['count']['scale'][:,splits[it],:] = temp_params_xr['count']['scale']
          if exists_types[2]:
            # params_x['pos']['mean'][splits[it],:] = temp_params_x['pos']['mean']; params_x['pos']['scale'][splits[it],:] = temp_params_x['pos']['scale']
            # if not ignorable: params_xr['pos']['mean'][splits[it],:] = temp_params_xr['pos']['mean']; params_xr['pos']['scale'][splits[it],:] = temp_params_xr['pos']['scale']
            params_x['pos']['mean'][:,splits[it],:] = temp_params_x['pos']['mean']; params_x['pos']['scale'][:,splits[it],:] = temp_params_x['pos']['scale']
            params_xr['pos']['mean'][:,splits[it],:] = temp_params_xr['pos']['mean']; params_xr['pos']['scale'][:,splits[it],:] = temp_params_xr['pos']['scale']
          if exists_types[3]:
            for ii in range(0, p_cat):
              # params_x['cat'][ii][splits[it],:] = temp_params_x['cat'][ii]
              # if not ignorable: params_xr['cat'][ii][splits[it],:] = temp_params_xr['cat'][ii]
              params_x['cat'][ii][:,splits[it],:] = temp_params_x['cat'][ii]
              params_xr['cat'][ii][:,splits[it],:] = temp_params_xr['cat'][ii]
          # if not ignorable: params_r['probs'][splits[it],:] = temp_params_r['probs']
          if not ignorable: params_r['probs'][:,:,splits[it],:] = temp_params_r['probs']
          params_z['mean'][splits[it],:] = temp_params_z['mean']; params_z['scale'][splits[it],:] = temp_params_z['scale']
        
        loss_fit.pop("neg_bound")  # remove loss to not save computational graph associated with it
        # loss_fits = np.append(loss_fits, {'loss_fit': loss_fit, 'obs_ids': splits[it]})

        ############### L1 weight regularization #############
        if learn_r and not ignorable:
          L1_reg = torch.tensor(0., requires_grad=True).cuda()
          for name, param in decoder_r[0].named_parameters():
            if 'weight' in name:
              L1_reg = L1_reg + torch.norm(param, 1)
          loss = loss + L1_weight*L1_reg
        ######################################################

        # save the losses
        batches_loss = np.append(batches_loss, loss.cpu().data.numpy())

        loss.backward()
        optimizer.step()

        # Impose L1 thresholding to 0 for weight if norm < 1e-2
        if learn_r and not ignorable and L1_weight>0: #or L2_weight>0:
          with torch.no_grad(): decoder_r[0].weight[torch.abs(decoder_r[0].weight) < L1_weight] = 0           ####################### NEW
      
      # print("params_xr['cat'][0] (first 4)")
      # print(params_xr['cat'][0][:4])
      # print("params_x['cat'][0] (first 4)")
      # print(params_x['cat'][0][:4])
      # print("test1")
      # time_train=np.append(time_train,time.time()-t0_train)  # something wrong here
      # print("test2")
      # The LB is just for tracking --> need not do a full pass each epoch (can omit for saving memory later on)
      if covars: torch_covars_miss = torch.from_numpy(covars_miss).float().cuda()
      else: torch_covars_miss = None

      #NIMIWAE_LB=(-np.log(K) - np.log(M) - loss_fit['neg_bound'].cpu().data.numpy())
      if learn_r and not ignorable and L1_weight>0: #or L2_weight>0:
        with torch.no_grad(): decoder_r[0].weight[torch.abs(decoder_r[0].weight) < L1_weight] = 0
      
      total_loss = -np.sum(batches_loss)   # negative of the total loss (summed over K & bs)
      if(arch=="VAE"):
        NIMIWAE_LB = total_loss / (niw*n)
        ## loss = loss/(K*b_data.shape[0])                        # loss for a batch
      elif(arch=="IWAE"):
        NIMIWAE_LB = total_loss / (niw*n) + np.log(niw) + np.log(M)
        ## loss = loss/(b_data.shape[0]) + np.log(K) + np.log(M)   # loss for a batch

      NIMIWAE_LB_epoch=np.append(NIMIWAE_LB_epoch,NIMIWAE_LB)
      #learned_probMissing = np.mean(np.mean(params_r['probs'].reshape([M,-1]),axis=0).reshape([niw,-1]),axis=0).reshape([n,p])  #.cpu().data.numpy()
      #mse_pr=np.mean(pow(learned_probMissing[:,0]-probMissing[:,0],2)) # just the first column (missing column in toy, adjust later)
      #mse_pr_epoch=np.append(mse_pr_epoch, mse_pr)
      #CEL=np.sum(-np.log(learned_probMissing[mask==1])) + np.sum(-np.log(1-learned_probMissing[mask==0]))
      #CEL_epoch = np.append(CEL_epoch, CEL)
      sum_logpz_epoch=np.append(sum_logpz_epoch,loss_fit['sum_logpz'])
      sum_logqz_epoch=np.append(sum_logqz_epoch,loss_fit['sum_logqz'])
      sum_logpr_epoch=np.append(sum_logpr_epoch,loss_fit['sum_logpr'])
      sum_logpxobs_epoch=np.append(sum_logpxobs_epoch,loss_fit['sum_logpxobs'])

      ## beta-VAE (deprecated)
      # if (beta<1): beta=beta + beta_anneal_rate  # Sonderby
      #else:
      #  beta=1  # if beta > 1 --> beta-VAE (weight KL divergene higher) 
      if ep % 100 == 1:
        print("temp: " + str(temp))
        print('Epoch %g' %ep)
        print('NIMIWAE likelihood bound  %g' %NIMIWAE_LB) # Gradient step   

        #if trace:
          #print("mean (avg over K samples), p(x|z):")
          #print(np.mean(params_x['mean'].reshape([niw,-1]),axis=0).reshape([n,p])[trace_ids,0:min(4,p)])
          #print("sd (avg over K samples), p(x|z):")
          #print(np.mean(params_x['sd'].reshape([niw,-1]),axis=0).reshape([n,p])[trace_ids,0:min(4,p)])
          #print("mean (avg over K samples), q(x|z,r):")
          #print(np.mean(params_xr['mean'].reshape([niw,-1]),axis=0).reshape([n,p])[trace_ids,0:min(4,p)])
          #print("sd (avg over K samples), q(x|z,r):")
          #print(np.mean(params_xr['sd'].reshape([niw,-1]),axis=0).reshape([n,p])[trace_ids,0:min(4,p)])

          #print("probs P(r=1|x) (avg over M, then K samples):")
          #print(np.mean(np.mean(params_r['probs'].reshape([M,-1]),axis=0).reshape([niw,-1]),axis=0).reshape([n,p])[trace_ids,0:min(4,p)])
        ### Now we do the imputation

        if learn_r and not ignorable:
          print("Decoder_r weights (columns = input, rows = output) first 4:")
          print(decoder_r[0].weight[0:min(4,p),0:min(4,p)])

        t0_impute=time.time()
        if (not draw_xmiss) and (not ignorable): batches_full = np.array_split(xfull,n/impute_bs)
        batches_data = np.array_split(xhat_0, n/impute_bs)
        batches_mask = np.array_split(mask, n/impute_bs)
        batches_mask0 = np.array_split(mask0, n/impute_bs)
        if covars: batches_covar = np.array_split(covars_miss, n/impute_bs)
        splits = np.array_split(range(n),n/impute_bs)
        xhat_fits=[]
        for it in range(len(batches_data)):
          if (not draw_xmiss) and (not ignorable): b_full = torch.from_numpy(batches_full[it]).float().cuda()
          else: b_full = None
          # b_full = None
          b_data = torch.from_numpy(batches_data[it]).float().cuda()
          b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
          b_mask0 = torch.from_numpy(batches_mask0[it]).float().cuda()
          if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
          else: b_covar = None
          xhat_fit=nimiwae_impute(iota_xfull = b_full, iota_x = b_data, mask = b_mask, mask0 = b_mask0, covar_miss = b_covar, L=L, temp=temp)
          xhat_fits = np.append(xhat_fits, {'xhat_fit': xhat_fit, 'obs_ids': splits[it]})
          #print(b_data[:4]); print(xhat_0[:4]); print(b_mask[:4]); print(mask[:4])
          b_xhat = xhat[splits[it],:]
          #b_xhat[batches_mask[it]] = np.mean(params_x['mean'].reshape([niw,-1]),axis=0).reshape([n,p])[splits[it],:][batches_mask[it]]   #  .cpu().data.numpy()[batches_mask[it]]  # keep observed data as truth
          b_xhat[~batches_mask[it]] = xhat_fit['xm'].cpu().data.numpy()[~batches_mask[it]] # just missing imputed

          xhat[splits[it],:] = b_xhat
        
        time_impute=np.append(time_impute,time.time()-t0_impute)
        
        ######## undo normalization/any transformation
        # To re-create categorical variables, create placeholders
        xhat2 = np.empty([xhat.shape[0],len(data_types_0)])
        mask2 = np.empty([mask.shape[0],len(data_types_0)])
        xfull2 = np.empty([xhat.shape[0],len(data_types_0)])
        covars_r2 = np.empty([len(data_types_0)])
        i_real=0; i_count=0; i_pos=0; i_cat=0; C0=0
        
        
        if exists_types[0]:
          err_real = mse(xhat[:,ids_real], xfull[:,ids_real], mask[:,ids_real])
          # err_real = mse(xhat[:,ids_real], (xfull*norm_sds + norm_means)[:,ids_real], mask[:,ids_real])
          print(err_real)
          print('Observed MSE x_real:  %g' %err_real['obs'])
          print('Missing MSE x_real:  %g' %err_real['miss'])
        if exists_types[1]:
          err_count = mse(xhat[:, ids_count], xfull[:, ids_count], mask[:, ids_count])
          # err_count = mse(xhat[:, ids_count], (xfull*norm_sds + norm_means)[:, ids_count], mask[:, ids_count])
          print('Observed MSE x_count:  %g' %err_count['obs'])
          print('Missing MSE x_count:  %g' %err_count['miss'])
        if exists_types[2]:
          err_pos = mse(xhat[:, ids_pos], xfull[:, ids_pos], mask[:, ids_pos])
          # err_pos = mse(xhat[:, ids_pos], (xfull*norm_sds + norm_means)[:, ids_pos], mask[:, ids_pos])
          print('Observed MSE x_pos:  %g' %err_pos['obs'])
          print('Missing MSE x_pos:  %g' %err_pos['miss'])
        
        # print("L1 (missing):")
        # print(np.mean(np.abs(xhat-xfull)[~mask]))
        # print("L1 (observed):")
        # print(np.mean(np.abs(xhat-xfull)[mask]))
        
        print("L1 (missing):")
        print(np.mean(np.abs((xhat-xfull)*norm_sds)[~mask]))
        print("L1 (observed):")
        print(np.mean(np.abs((xhat-xfull)*norm_sds)[mask]))
        ## Undo normalization
        # xhat = (xhat*norm_sds) + norm_means
        
        # Fill in xhat in original data_types_0 order (as input into NIMIWAE's wrapper R function)
        for i in range(0,len(data_types_0)):
          if data_types_0[i]=="real":
            xhat2[:,i] = xhat[:,np.where(ids_real)[0][i_real]]
            mask2[:,i] = mask[:,np.where(ids_real)[0][i_real]]
            xfull2[:,i] = xfull[:,np.where(ids_real)[0][i_real]]
            covars_r2[i] = covars_r[np.where(ids_real)[0][i_real]]
            i_real=i_real+1
          if data_types_0[i]=="count":
            # xhat0[:,i] = np.exp(xhat[:,np.where(ids_count)[0][i_count]])  ## if you a priori log-transform count, undo transf
            xhat2[:,i] = xhat[:,np.where(ids_count)[0][i_count]]
            mask2[:,i] = mask[:,np.where(ids_count)[0][i_count]]
            xfull2[:,i] = xfull[:,np.where(ids_count)[0][i_count]]
            covars_r2[i] = covars_r[np.where(ids_count)[0][i_count]]
            i_count=i_count+1
          if data_types_0[i]=="pos":
            xhat2[:,i] = xhat[:,np.where(ids_pos)[0][i_pos]]
            mask2[:,i] = mask[:,np.where(ids_pos)[0][i_pos]]
            xfull2[:,i] = xfull[:,np.where(ids_pos)[0][i_pos]]
            covars_r2[i] = covars_r[np.where(ids_pos)[0][i_pos]]
            i_pos=i_pos+1
          if data_types_0[i]=="cat":
            idd = np.where(ids_cat)[0][int(C0*i_cat):int(C0*i_cat + Cs[i_cat])]
            xhat2[:,i] = np.argmax(xhat[:,idd], axis=1) + 1
            xfull2[:,i] = np.argmax(xfull[:,idd], axis=1) + 1
            mask2[:,i] = mask[:,idd[0]]
            covars_r2[i] = covars_r[idd[0]]
            C0=Cs[i_cat]
            i_cat = i_cat+1
        
        if exists_types[3]:
          # err_cat = mse(xhat[:, ids_cat], xfull[:, ids_cat], mask[:, ids_cat])
          # print('Observed MSE x_cat (probs):  %g' %err_cat['obs'])
          # print('Missing MSE x_cat (probs):  %g' %err_cat['miss'])
          err_cat={}
          ids_cat2 = data_types_0=="cat"
          err_cat['obs'] = np.mean(xhat2[:,ids_cat2][mask2[:,ids_cat2]==1] == xfull2[:,ids_cat2][mask2[:,ids_cat2]==1])
          err_cat['miss'] = np.mean(xhat2[:,ids_cat2][mask2[:,ids_cat2]==0] == xfull2[:,ids_cat2][mask2[:,ids_cat2]==0])
          print('Prediction accuracy X_cat (obs):  %g' %err_cat['obs'])
          print('Prediction accuracy X_cat (miss):  %g' %err_cat['miss'])
        
        print("xfull (first 4):")
        print(xfull[:4,:min(4,p)])
        # print("xfull2 (first 2):")
        # print(xfull2[:2])
        print("xhat (first 4):")
        print(xhat[:4,:min(4,p)])
        print("xhat2 (first 4):")
        print(xhat2[:4,:min(4,p)])
        print("mask (first 4):")
        print(mask[:4,:min(4,p)])

        zgivenx = xhat_fit['zgivenx'].cpu().data.numpy()   # L samples*batch_size x d (d: latent dimension)
        imp_weights = xhat_fit['imp_weights'].cpu().data.numpy()
        # print('Observed MSE  %g' %err['obs'])   # these aren't reconstructed/imputed
        # print('Missing MSE  %g' %err['miss'])
        print('-----')
        # temp = torch.max(temp0*torch.exp(-ANNEAL_RATE*ep), temp_min)  # anneal the temp once every 100 iters? (Jang et al does every 1000 iters)
      # temp = torch.max(temp0*torch.exp(-ANNEAL_RATE*ep), temp_min)  # anneal the temp once every iter? (Jang et al does every 1000 iters)
      temp = torch.max(temp0-ANNEAL_RATE*ep, temp_min)  # anneal every epoch (HIVAE)
      if early_stop:
        ##################################################################
        ###### COMPUTE VALIDATION LOSS (for early stopping criteria) #####
        ##################################################################
        perm = np.random.permutation(n_val) # We use the "random reshuffling" version of SGD
        if (not draw_xmiss) and (not ignorable): batches_full = np.array_split(xfull_val[perm,],n_val/bs_val)
        batches_data = np.array_split(xhat_0_val[perm,], n_val/bs_val)
        batches_mask = np.array_split(mask_val[perm,], n_val/bs_val)
        batches_mask0 = np.array_split(mask0_val[perm,], n_val/bs_val)
        if covars: batches_covar = np.array_split(covars_miss_val[perm,], n_val/bs_val)
        #batches_prM = np.array_split(prM[perm,],n/bs)
        splits = np.array_split(perm,n_val/bs_val)
        # minibatch save:
        # losses
        batches_val_loss = []
        for it in range(len(batches_data)):
          if (not draw_xmiss) and (not ignorable): b_full = torch.from_numpy(batches_full[it]).float().cuda()
          else: b_full = None
          # b_full = None
          b_data = torch.from_numpy(batches_data[it]).float().cuda()
          b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
          b_mask0 = torch.from_numpy(batches_mask0[it]).float().cuda()
          if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
          else: b_covar = None
  
          optimizer.zero_grad()
          encoder.zero_grad()
          decoders.zero_grad()
          encoders_xr.zero_grad()
          if (learn_r and not ignorable): decoder_r.zero_grad()
          
          #print(b_data_val[:20])
          # print(b_mask_val[:20])
          loss_fit = nimiwae_loss(iota_xfull=b_full, iota_x = b_data, mask = b_mask, mask0 = b_mask0, covar_miss = b_covar, temp=temp)
          val_loss = loss_fit['neg_bound'].detach()

          # save the validation losses
          batches_val_loss = np.append(batches_val_loss, val_loss.cpu().data.numpy())
        total_val_loss = -np.sum(batches_val_loss)   # negative of the total loss (summed over K & bs)
        if(arch=="VAE"):
          NIMIWAE_val_LB = total_val_loss / (niw*n)
          ## loss = loss/(K*b_data.shape[0])                        # loss for a batch
        elif(arch=="IWAE"):
          NIMIWAE_val_LB = total_val_loss / (niw*n) - np.log(niw) - np.log(M)
        
        NIMIWAE_val_LB_epoch=np.append(NIMIWAE_val_LB_epoch,NIMIWAE_val_LB)
        #### example: (people usually don't skip first epochs)
        ## If the validation loss is at a minimum
        # if (NIMIWAE_val_LB > max_NIMIWAE_val_LB):
        #   epochs_no_improve = 0
        #   max_NIMIWAE_val_LB = NIMIWAE_val_LB
        # else:
        #   epochs_no_improve += 1
        # # Check early stopping condition
        # if epochs_no_improve == n_epochs_stop:
        #   print('Early stopping at epoch %d!' %ep)
        #   early_stop=True
        #   early_stop_epochs = ep
        if ep > early_stop_check_epochs:
          delta_val_LB = (NIMIWAE_val_LB_epoch[ep-1] - NIMIWAE_val_LB_epoch[(ep-1) - early_stop_check_epochs])/np.absolute(NIMIWAE_val_LB_epoch[(ep-1) - early_stop_check_epochs])
          #print("delta_val_LB: %g" %delta_val_LB)
          if delta_val_LB < early_stop_tol:
            early_stopped = True
            print('Early stopping at epoch %d!' %ep)
            early_stop_epochs = ep
        if early_stopped: break
      sys.stdout.flush()   # output everything
    if (learn_r and not ignorable): saved_model={'encoder': encoder, 'encoders_xr': encoders_xr, 'decoders': decoders, 'decoder_r':decoder_r}
    else: saved_model={'encoder': encoder, 'encoders_xr': encoders_xr, 'decoders': decoders}
    
    all_params = {'x': params_x, 'z': params_z}
    if (not ignorable) and learn_r and not ignorable: all_params['r'] = params_r
    all_params['xm'] = params_xr
    
    # mse_train={'miss':mse_train_miss,'obs':mse_train_obs}
    train_params = {'h1':h1, 'h2':h2, 'h3':h3, 'h4':h4, 'sigma':sigma, 'bs':bs, 'n_epochs':n_epochs, 'lr':lr, 'niw':niw, 'dim_z':dim_z, 'L':L, 'M':M, 'n_hidden_layers': n_hidden_layers, 'n_hidden_layers_r': n_hidden_layers_r, 'L1_weight': L1_weight, 'L2_weight': L2_weight, "early_stopped":early_stop, "early_stop_epochs":ep}
    #fit = {'params_x': params_x, 'params_xr': params_xr, 'params_r': params_r, 'params_z': params_z}
    #return {'train_params':train_params, 'loss_fit':loss_fit, 'xhat_fit':xhat_fit,'saved_model': saved_model,'zgivenx_flat': zgivenx_flat,'NIMIWAE_LB_epoch': NIMIWAE_LB_epoch,'time_train': time_train,'time_impute': time_impute,'imp_weights': imp_weights,'MSE': mse_train, 'xhat': xhat, 'mask': mask, 'norm_means':norm_means, 'norm_sds':norm_sds}
    # return {'train_params':train_params, 'loss_fits': loss_fits,'xhat_fits':xhat_fits,'saved_model': saved_model,'LB': NIMIWAE_LB,'zgivenx_flat': zgivenx_flat,'NIMIWAE_LB_epoch': NIMIWAE_LB_epoch,'NIMIWAE_val_LB_epoch': NIMIWAE_val_LB_epoch,'time_train': time_train,'time_impute': time_impute,'imp_weights': imp_weights,'MSE': mse_train, 'xhat': xhat, 'mask': mask, 'norm_means':norm_means, 'norm_sds':norm_sds}
    return {'all_params': all_params,'train_params':train_params,'xhat_fits':xhat_fits,'saved_model': saved_model,'LB': NIMIWAE_LB,'zgivenx': zgivenx,'NIMIWAE_LB_epoch': NIMIWAE_LB_epoch,'NIMIWAE_val_LB_epoch': NIMIWAE_val_LB_epoch,'time_train': time_train,'time_impute': time_impute,'imp_weights': imp_weights,'xhat': xhat, 'mask': mask, 'norm_means':norm_means, 'norm_sds':norm_sds, 'covars_r':covars_r}
  else:
    import h5py
    hf = h5py.File(dir_name + '/samples.h5', 'w')
    temp=temp_min
    # validating (hyperparameter values) or testing
    encoder=saved_model['encoder']
    decoders=saved_model['decoders']
    encoders_xr=saved_model['encoders_xr']
    if (learn_r and not ignorable): decoder_r=saved_model['decoder_r']

    for ep in range(1,n_epochs):
      # Validation set is much smaller, so including all observations should be fine?
      #if covars: torch_covars_miss = torch.from_numpy(covars_miss).float().cuda()
      #else: torch_covars_miss = None

      perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
      if (not draw_xmiss) and (not ignorable): batches_full = np.array_split(xfull[perm,],n/bs)
      batches_data = np.array_split(xhat_0[perm,], n/bs)
      batches_mask = np.array_split(mask[perm,], n/bs)
      batches_mask0 = np.array_split(mask0[perm,], n/bs)
      if covars: batches_covar = np.array_split(covars_miss[perm,], n/bs)
      #batches_prM = np.array_split(prM[perm,],n/bs)
      splits = np.array_split(perm,n/bs)

      batches_loss = []
      t0_train=time.time()
      encoder.zero_grad()
      if exists_types[0]:
        decoders['real'].zero_grad()
        encoders_xr['real'].zero_grad()
      if exists_types[1]:
        decoders['count'].zero_grad()
        encoders_xr['count'].zero_grad()
      if exists_types[2]:
        decoders['pos'].zero_grad()
        encoders_xr['pos'].zero_grad()
      if exists_types[3]:
        for ii in range(0,p_cat):
          decoders['cat'][ii].zero_grad()
          encoders_xr['cat'][ii].zero_grad()

      if learn_r and not ignorable: decoder_r.zero_grad()

      loss_fits = []
      
      print("# batches:")
      print(len(batches_data))
      for it in range(len(batches_data)):
        print("batch" + str(it) + ":")
        if (not draw_xmiss) and (not ignorable): b_full = torch.from_numpy(batches_full[it]).float().cuda()
        else: b_full = None
        # b_full = None
        b_data = torch.from_numpy(batches_data[it]).float().cuda()
        b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
        b_mask0 = torch.from_numpy(batches_mask0[it]).float().cuda()
        if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
        else: b_covar = None
            
        loss_fit = nimiwae_loss(iota_xfull=b_full, iota_x = b_data, mask = b_mask, mask0 = b_mask0, covar_miss = b_covar, temp=temp)
        loss = loss_fit['neg_bound']
        batches_loss = np.append(batches_loss, loss.cpu().data.numpy())
        
        temp_params_x, temp_params_xr, temp_params_r, temp_params_z = loss_fit['params_x'], loss_fit['params_xr'], loss_fit['params_r'], loss_fit['params_z']

        ##### SAVE ALL params_x, params_xr, params_r, params_z #####
        if exists_types[0]:
          # params_x['real']['mean'][splits[it],:] = temp_params_x['real']['mean']; params_x['real']['scale'][splits[it],:] = temp_params_x['real']['scale']
          # if not ignorable: params_xr['real']['mean'][splits[it],:] = temp_params_xr['real']['mean']; params_xr['real']['scale'][splits[it],:] = temp_params_xr['real']['scale']
          params_x['real']['mean'][:,splits[it],:] = temp_params_x['real']['mean']; params_x['real']['scale'][:,splits[it],:] = temp_params_x['real']['scale']
          params_xr['real']['mean'][:,splits[it],:] = temp_params_xr['real']['mean']; params_xr['real']['scale'][:,splits[it],:] = temp_params_xr['real']['scale']
        if exists_types[1]:
          # params_x['count']['mean'][splits[it],:] = temp_params_x['count']['mean']; params_x['count']['scale'][splits[it],:] = temp_params_x['count']['scale']
          # if not ignorable: params_xr['count']['mean'][splits[it],:] = temp_params_xr['count']['mean']; params_xr['count']['scale'][splits[it],:] = temp_params_xr['count']['scale']
          params_x['count']['mean'][:,splits[it],:] = temp_params_x['count']['mean']; params_x['count']['scale'][:,splits[it],:] = temp_params_x['count']['scale']
          params_xr['count']['mean'][:,splits[it],:] = temp_params_xr['count']['mean']; params_xr['count']['scale'][:,splits[it],:] = temp_params_xr['count']['scale']
        if exists_types[2]:
          # params_x['pos']['mean'][splits[it],:] = temp_params_x['pos']['mean']; params_x['pos']['scale'][splits[it],:] = temp_params_x['pos']['scale']
          # if not ignorable: params_xr['pos']['mean'][splits[it],:] = temp_params_xr['pos']['mean']; params_xr['pos']['scale'][splits[it],:] = temp_params_xr['pos']['scale']
          params_x['pos']['mean'][:,splits[it],:] = temp_params_x['pos']['mean']; params_x['pos']['scale'][:,splits[it],:] = temp_params_x['pos']['scale']
          params_xr['pos']['mean'][:,splits[it],:] = temp_params_xr['pos']['mean']; params_xr['pos']['scale'][:,splits[it],:] = temp_params_xr['pos']['scale']
        if exists_types[3]:
          for ii in range(0, p_cat):
            # params_x['cat'][ii][splits[it],:] = temp_params_x['cat'][ii]
            # if not ignorable: params_xr['cat'][ii][splits[it],:] = temp_params_xr['cat'][ii]
            params_x['cat'][ii][:,splits[it],:] = temp_params_x['cat'][ii]
            params_xr['cat'][ii][:,splits[it],:] = temp_params_xr['cat'][ii]
        # if not ignorable: params_r['probs'][splits[it],:] = temp_params_r['probs']
        if not ignorable: params_r['probs'][:,:,splits[it],:] = temp_params_r['probs']
        params_z['mean'][splits[it],:] = temp_params_z['mean']; params_z['scale'][splits[it],:] = temp_params_z['scale']
        
        loss_fit.pop("neg_bound")
        # loss_fits = np.append(loss_fits, {'loss_fit': loss_fit, 'obs_ids': splits[it]})
       
      print("Done with forward statement.")
      total_loss = -np.sum(batches_loss)   # negative of the total loss (summed over K & bs)
      if(arch=="VAE"):
        NIMIWAE_LB = total_loss / (niw*n)
        ## loss = loss/(K*b_data.shape[0])                        # loss for a batch
      elif(arch=="IWAE"):
        NIMIWAE_LB = total_loss / (niw*n) - np.log(niw) - np.log(M)
        ## loss = loss/(b_data.shape[0]) + np.log(K) + np.log(M)   # loss for a batch
      
      t0_impute=time.time()
      if (not draw_xmiss) and (not ignorable): batches_full = np.array_split(xfull,n/impute_bs)
      batches_data = np.array_split(xhat_0, n/impute_bs)
      batches_mask = np.array_split(mask, n/impute_bs)
      batches_mask0 = np.array_split(mask0, n/impute_bs)
      if covars: batches_covar = np.array_split(covars_miss, n/impute_bs)
      splits = np.array_split(range(n),n/impute_bs)
      xhat_fits = []   # not saved now
      
      if save_imps:
        
        # if not ignorable: all_imp_weights = np.empty([M*L,n])
        # else: all_imp_weights = np.empty([L,n])
        # if ignorable: all_xms = np.empty([L,np.sum(Missing==0)])
        # else: all_xms = np.empty([M*L,np.sum(Missing==0)])
        all_imp_weights = np.empty([M*L,n])
        all_xms = np.empty([M*L,np.sum(Missing==0)])
        
        idsx, idsy = np.where(Missing==0)
        # np.savetxt(dir_name + '/miss_XY.csv',np.stack((idsx,idsy),axis=1),delimiter=",")   # X coords in 1st column, Y coords in 2nd column
        hf.create_dataset("miss_XY", data=np.stack((idsx,idsy),axis=1), compression="gzip", compression_opts=9)
        
      print("# batches:")
      print(len(batches_data))
      for it in range(len(batches_data)):
        print("batch" + str(it) + ":")
        if (not draw_xmiss) and (not ignorable): b_full = torch.from_numpy(batches_full[it]).float().cuda()
        else: b_full = None
        # b_full = None
        b_data = torch.from_numpy(batches_data[it]).float().cuda()
        b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
        b_mask0 = torch.from_numpy(batches_mask0[it]).float().cuda()
        if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
        else: b_covar = None
        xhat_fit=nimiwae_impute(iota_xfull = b_full, iota_x = b_data, mask = b_mask, mask0 = b_mask0, covar_miss = b_covar, L=L, temp=temp)
        # xhat_fits = np.append(xhat_fits, {'xhat_fit': xhat_fit, 'obs_ids': splits[it]})
        #print(b_data[:4]); print(xhat_0[:4]); print(b_mask[:4]); print(mask[:4])
        b_xhat = xhat[splits[it],:]
        #b_xhat[batches_mask[it]] = torch.mean(loss_fit['params_x']['mean'].reshape([niw,-1]),axis=0).reshape([n,p])[splits[it],:].cpu().data.numpy()[batches_mask[it]]  # keep observed data as truth
        b_xhat[~batches_mask[it]] = xhat_fit['xm'].cpu().data.numpy()[~batches_mask[it]] # just missing imputed

        xhat[splits[it],:] = b_xhat
        
        zgivenx = xhat_fit['zgivenx'].cpu().data.numpy()   # [L,batch_size,d] (d: latent dimension)
        xms = xhat_fit['xms'].cpu().data.numpy() # [M*L,batch_size,p] for non-ignorable, [L,batch_size,p] for ignorable
        imp_weights = xhat_fit['imp_weights'].cpu().data.numpy()
        
        
        if save_imps:
          ids = np.isin(idsx, splits[it])   # ids to replace in all_xms (just missing values)
          for ii in range(0, xms.shape[0]):
            all_xms[ii,ids] = xms[ii,:,:][b_mask.cpu().data.numpy()==0]
          all_imp_weights[:,splits[it]] = imp_weights
      
      if save_imps:
        for ii in range(0, xms.shape[0]):
          # np.savetxt(dir_name + '/Xm'+str(ii)+".csv",all_xms[ii,:],delimiter=",")
          hf.create_dataset("Xm"+str(ii), data=all_xms[ii,:], compression="gzip", compression_opts=9)
        # np.savetxt(dir_name + "/IWs.csv",all_imp_weights, delimiter=",")  # should be L x n or M*L x n
        hf.create_dataset("IWs", data=all_imp_weights, compression="gzip", compression_opts=9)
      hf.close()
      #xhat_fit=nimiwae_impute(iota_xfull = cuda_xfull, iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda(),covar_miss = torch_covars_miss,L=L,temp=temp_min)
      time_impute=np.append(time_impute,time.time()-t0_impute)
      
      # ## if save_imps: then save all_zgivenxs and all_xms
      # if save_imps:
      #   # for ii in range(0, all_zgivenxs.shape[0]):
      #   #   np.savetxt(dir_name + '/Z'+str(ii)+".csv",all_zgivenxs[ii,:,:],delimiter=",")   ### Z's need not be saved
      #   for ii in range(0, all_xms.shape[0]):
      #     np.savetxt(dir_name + '/Xm'+str(ii)+".csv",all_xms[ii,:,:],delimiter=",")
      #   np.savetxt(dir_name + "/IWs.csv",all_imp_weights, delimiter=",")  # should be L x n or M*L x n
        
      #### need to reconstruct tensors (b/c minibatching)
      #### need to pass directory to save: Results/SIM../phi../sim../MCAR/miss../)
      #### save to IWs.csv, xm.csv, xms.csv, zgivenx.csv, params.csv
      #### params.csv: contain M (=1 for ignorable), L, n, p, d to reshape xms and zgivenx to original dimensions
      #### save imp_weights, xm, and xincluded (M files, with L*batch_size rows and p cols), and zgivenx_flat (L files with bs rows and d cols)
      
      print("xfull (first 4):")
      print(xfull[:4,:min(4,p)])
      # print("xfull2 (first 2):")
      # print(xfull2[:2])
      print("xhat (first 4):")
      print(xhat[:4,:min(4,p)])
      
      if exists_types[0]:
        err_real = mse(xhat[:, ids_real], xfull[:, ids_real], mask[:, ids_real])
        print('Observed MSE x_real: %g' %err_real['obs'])
        print('Missing MSE x_real: %g' %err_real['miss'])
      if exists_types[1]:
        err_count = mse(xhat[:, ids_count], xfull[:, ids_count], mask[:, ids_count])
        print('Observed MSE x_count: %g' %err_count['obs'])
        print('Missing MSE x_count: %g' %err_count['miss'])
      if exists_types[2]:
        err_pos = mse(xhat[:, ids_pos], xfull[:, ids_pos], mask[:, ids_pos])
        print('Observed MSE x_pos: %g' %err_pos['obs'])
        print('Missing MSE x_pos: %g' %err_pos['miss'])
      if exists_types[3]:
        err_cat = mse(xhat[:, ids_cat], xfull[:, ids_cat], mask[:, ids_cat])
        print('Observed MSE x_cat (probs): %g' %err_cat['obs'])
        print('Missing MSE x_cat (probs): %g' %err_cat['miss'])

      #xhat[mask] = torch.mean(loss_fit['params_x']['mean'].reshape([niw,-1]),axis=0).reshape([n,p]).cpu().data.numpy()[mask]
      #xhat[~mask] = xhat_fit['xm'].cpu().data.numpy()[~mask]
      #####xhat = xhat_fit['xm'].cpu().data.numpy()

      # err = mse(xhat,xfull,mask)
      # mse_train_miss = np.append(mse_train_miss,np.array([err['miss']]),axis=0)
      # mse_train_obs = np.append(mse_train_obs,np.array([err['obs']]),axis=0)
    # mse_test={'miss':err['miss'],'obs':err['obs']}
    
    ######## undo normalization/any transformation
    # To re-create categorical variables, create placeholders
    xhat0 = np.empty([xhat.shape[0],len(data_types_0)])
    mask0 = np.empty([mask.shape[0],len(data_types_0)])
    covars_r0 = np.empty([len(data_types_0)])
    i_real=0; i_count=0; i_pos=0; i_cat=0; C0=0
    
    # Undo normalization
    xhat = (xhat*norm_sds) + norm_means
    
    # Fill in xhat in original data_types_0 order (as input into NIMIWAE's wrapper R function)
    for i in range(0,len(data_types_0)):
      if data_types_0[i]=="real":
        xhat0[:,i] = xhat[:,np.where(ids_real)[0][i_real]]
        mask0[:,i] = mask[:,np.where(ids_real)[0][i_real]]
        covars_r0[i] = covars_r[np.where(ids_real)[0][i_real]]
        i_real=i_real+1
      if data_types_0[i]=="count":
        # xhat0[:,i] = np.exp(xhat[:,np.where(ids_count)[0][i_count]])  ## if you a priori log-transform count, undo transf
        xhat0[:,i] = xhat[:,np.where(ids_count)[0][i_count]]
        mask0[:,i] = mask[:,np.where(ids_count)[0][i_count]]
        covars_r0[i] = covars_r[np.where(ids_count)[0][i_count]]
        i_count=i_count+1
      if data_types_0[i]=="pos":
        xhat0[:,i] = xhat[:,np.where(ids_pos)[0][i_pos]]
        mask0[:,i] = mask[:,np.where(ids_pos)[0][i_pos]]
        covars_r0[i] = covars_r[np.where(ids_pos)[0][i_pos]]
        i_pos=i_pos+1
      if data_types_0[i]=="cat":
        idd = np.where(ids_cat)[0][int(C0*i_cat):int(C0*i_cat + Cs[i_cat])]
        xhat0[:,i] = np.argmax(xhat[:,idd], axis=1) + 1
        mask0[:,i] = mask[:,idd[0]]
        covars_r0[i] = covars_r[idd[0]]
        C0=Cs[i_cat]
        i_cat = i_cat+1
    
    print("xhat")
    print(xhat[:4,:min(4,p)])
    print("L1 (missing):")
    print(np.mean(np.abs(xhat-(xfull*norm_sds+norm_means))[~mask]))
    print("L1 (observed):")
    print(np.mean(np.abs(xhat-(xfull*norm_sds+norm_means))[mask]))
    
    # replace
    xhat = xhat0
    mask = mask0
    data_types = data_types_0
    covars_r = covars_r0
    
    print("xhat0")
    print(xhat0[:4,:min(4,p)])
    
    sys.stdout.flush()
    ########
    
    
    if (learn_r and not ignorable): saved_model={'encoder': encoder, 'encoders_xr': encoders_xr, 'decoders': decoders, 'decoder_r':decoder_r}
    else: saved_model={'encoder': encoder, 'encoders_xr': encoders_xr, 'decoders': decoders}
    
    all_params = {'x': params_x, 'z': params_z}
    if (not ignorable) and learn_r: all_params['r'] = params_r
    all_params['xm'] = params_xr
    
    if learn_r and not ignorable: decoder_r_weights = (decoder_r[0].weight).cpu().data.numpy()
    else: decoder_r_weights=None
    # omitted saved_model from output when test time
    return {'all_params': all_params,'decoder_r_weights': decoder_r_weights,'loss_fits':loss_fits, 'xhat_fits':xhat_fits,'zgivenx': zgivenx,'LB': NIMIWAE_LB,'time_impute': time_impute, 'xhat': xhat, 'xfull': xfull, 'mask': mask, 'norm_means':norm_means, 'norm_sds':norm_sds, 'covars_r':covars_r}
    #return {'loss_fit':loss_fit,'xhat_fit':xhat_fit,'zgivenx_flat': zgivenx_flat,'saved_model': saved_model,'LB': NIMIWAE_LB,'time_impute': time_impute,'imp_weights': imp_weights,'MSE': mse_test, 'xhat': xhat, 'xfull': xfull, 'mask': mask, 'norm_means':norm_means, 'norm_sds':norm_sds}
  




















































def run_MIWAE(data,Missing,norm_means,norm_sds,n_hidden_layers=2,dec_distrib="Normal",train=1,saved_model=None,h=10,sigma="relu",bs = 64,n_epochs = 2002,lr=0.001,niw=20,dim_z=5,L=20,trace=False):
  # L: number of MC samples used in imputation
  import torch
  #import torchvision
  import torch.nn as nn
  import numpy as np
  import scipy.stats
  import scipy.io
  import scipy.sparse
  from scipy.io import loadmat
  import pandas as pd
  from matplotlib.backends.backend_pdf import PdfPages
  import matplotlib.pyplot as plt
  import torch.distributions as td

  from torch import nn, optim
  from torch.nn import functional as F
  # from torchvision import datasets, transforms
  # from torchvision.utils import save_image

  import time

  def mse(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return {'miss':np.mean(np.power(xhat-xtrue,2)[~mask]),'obs':np.mean(np.power(xhat-xtrue,2)[mask])}
  
  time0 = time.time()
    
  # xfull = (data - np.mean(data,0))/np.std(data,0)
  xfull = (data - norm_means)/norm_sds
  n = xfull.shape[0] # number of observations
  p = xfull.shape[1] # number of features
  
  np.random.seed(1234)
  
  xmiss = np.copy(xfull)
  xmiss[Missing==0]=np.nan
  mask = np.isfinite(xmiss) # binary mask that indicates which values are missing
  mask0 = np.copy(mask)
  
  xhat_0 = np.copy(xmiss)
  xhat_0[np.isnan(xmiss)] = 0
  
  d = dim_z # dimension of the latent space
  K = niw # number of IS during training
  
  bs = min(bs,n)
  impute_bs = min(bs, n)

  p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)     # THIS IS NORMAL vs. student T used in CPU version!!
  if (dec_distrib=="Normal"): num_dec_params=2
  elif (dec_distrib=="StudentT"): num_dec_params=3
  
  if (sigma=="relu"): act_fun=torch.nn.ReLU()
  elif (sigma=="elu"): act_fun=torch.nn.ELU()
  
  def network_maker(act_fun, n_hidden_layers, in_h, h, out_h, dropout=False):
    if n_hidden_layers==0:
      layers = [ nn.Linear(in_h, out_h), ]
    elif n_hidden_layers>0:
      layers = [ nn.Linear(in_h , h), act_fun, ]
      for i in range(n_hidden_layers-1):
        layers.append( nn.Linear(h, h), )
        layers.append( act_fun, )
      layers.append(nn.Linear(h, out_h))
    elif n_hidden_layers<0:
      raise Exception("n_hidden_layers must be >= 0")
    if dropout:
      layers.insert(0, nn.Dropout())
    model = nn.Sequential(*layers)
    return model
  
  encoder = network_maker(act_fun, n_hidden_layers, p, h, 2*d, False)
  decoder = network_maker(act_fun, n_hidden_layers, d, h, num_dec_params*p, False)
  
  # decoder = nn.Sequential(
  #   torch.nn.Linear(d, h),
  #   torch.nn.ReLU(),
  #   torch.nn.Linear(h, num_dec_params*p),
  # )
  # encoder = nn.Sequential(
  #   torch.nn.Linear(p, h),
  #   torch.nn.ReLU(),
  #   torch.nn.Linear(h, 2*d),  # the encoder will output both the mean and the diagonal covariance
  # )
  
  encoder.cuda() # we'll use the GPU
  decoder.cuda()
  
  def miwae_loss(iota_x,mask):
    batch_size = iota_x.shape[0]
    out_encoder = encoder(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
    
    zgivenx = q_zgivenxobs.rsample([K])
    zgivenx_flat = zgivenx.reshape([K*batch_size,d])
    
    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
    if dec_distrib=="StudentT":
      all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3
    
    data_flat = torch.Tensor.repeat(iota_x,[K,1]).reshape([-1,1])
    tiledmask = torch.Tensor.repeat(mask,[K,1])
    
    if dec_distrib=="Normal":
      all_log_pxgivenz_flat = torch.distributions.Normal(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1])).log_prob(data_flat)
      params_x={'mean':all_means_obs_model,'sd':all_scales_obs_model}
    elif dec_distrib=="StudentT":
      all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
      params_x={'mean':all_means_obs_model,'sd':all_scales_obs_model,'df':all_degfreedom_obs_model}
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p])     # p(x|z) : Product of 1-D student's T. q(z|x) : MV-Gaussian
    
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)
    
    # neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))
    neg_bound = -torch.sum(torch.logsumexp(logpxobsgivenz + logpz - logq,0))  # average this after summing minibatches
    params_z={'mean':out_encoder[..., :d], 'sd':torch.nn.Softplus()(out_encoder[..., d:(2*d)])}
    return{'neg_bound':neg_bound, 'params_x': {'mean': params_x['mean'].detach(), 'sd': params_x['sd'].detach()}, 'params_z': {'mean': params_z['mean'].detach(), 'sd': params_z['sd'].detach()}}
  
  optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=lr)
  
  def miwae_impute(iota_x,mask,L):
    batch_size = iota_x.shape[0]
    out_encoder = encoder(iota_x)

    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
    
    zgivenx = q_zgivenxobs.rsample([L])
    zgivenx_flat = zgivenx.reshape([L*batch_size,d])
    
    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
    if dec_distrib=="StudentT":
      all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3
    
    data_flat = torch.Tensor.repeat(iota_x,[L,1]).reshape([-1,1]).cuda()
    tiledmask = torch.Tensor.repeat(mask,[L,1]).cuda()
    
    if dec_distrib=="Normal":
      all_log_pxgivenz_flat = td.Normal(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1])).log_prob(data_flat)
      xgivenz = td.Independent(td.Normal(loc=all_means_obs_model, scale=all_scales_obs_model),1)
    elif dec_distrib=="StudentT":
      all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
      xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,p])
    
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)
    
    imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
    xms = xgivenz.sample().reshape([L,batch_size,p])
    xm=torch.einsum('ki,kij->ij', imp_weights, xms) 
    return {'xm': xm.detach(), 'imp_weights': imp_weights.detach(),'zgivenx_flat': zgivenx_flat.detach()}
  def weights_init(layer):
    if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
  
  miwae_loss_train=np.array([])
  mse_train_miss=np.array([])
  mse_train_obs=np.array([])
  bs = bs # batch size
  n_epochs = n_epochs
  xhat = np.copy(xhat_0) # This will be out imputed data matrix
  
  trace_ids = np.concatenate([np.where(Missing[:,0]==0)[0][0:2],np.where(Missing[:,0]==1)[0][0:2]])
  
  if trace:
    print(xhat_0[trace_ids])

  encoder.apply(weights_init)
  decoder.apply(weights_init)
  
  time_train=[]
  time_impute=[]
  MIWAE_LB_epoch=[]
  if train==1:
    # Training+Imputing
    for ep in range(1,n_epochs):
      perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
      batches_data = np.array_split(xhat_0[perm,], n/bs)
      batches_mask = np.array_split(mask0[perm,], n/bs)
      t0_train=time.time()
      splits = np.array_split(perm, n/bs)
      batches_loss = []
      loss_fits = []
      for it in range(len(batches_data)):
        optimizer.zero_grad()
        encoder.zero_grad()
        decoder.zero_grad()
        b_data = torch.from_numpy(batches_data[it]).float().cuda()
        b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
        loss_fit = miwae_loss(iota_x = b_data,mask = b_mask)
        loss = loss_fit['neg_bound']
        
        batches_loss = np.append(batches_loss, loss.cpu().data.numpy())
        loss.backward()
        
        loss_fit.pop("neg_bound")  # remove loss to not save computational graph associated with it
        loss_fits = np.append(loss_fits, {'loss_fit': loss_fit, 'obs_ids': splits[it]})
        optimizer.step()
      time_train=np.append(time_train,time.time()-t0_train)
      # loss_fit=miwae_loss(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda())
      # MIWAE_LB=(-np.log(K)-loss_fit['neg_bound'].cpu().data.numpy())
      total_loss = -np.sum(batches_loss)
      MIWAE_LB = total_loss / (K*n) - np.log(K)
      MIWAE_LB_epoch = np.append(MIWAE_LB_epoch,MIWAE_LB)
      if ep % 100 == 1:
        print('Epoch %g' %ep)
        print('MIWAE likelihood bound  %g' %MIWAE_LB) # Gradient step
        # if trace:
        #   print(loss_fit['params_x']['mean'][trace_ids])
        #   print(loss_fit['params_x']['sd'][trace_ids])
        ### Now we do the imputation
        t0_impute=time.time()
        
        # xhat_fit=miwae_impute(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda(),L=L)
        # xhat[~mask] = xhat_fit['xm'].cpu().data.numpy()[~mask]
        # #xhat = xhat_fit['xm'].cpu().data.numpy()  # observed values are not imputed
        batches_data = np.array_split(xhat_0, n/impute_bs)
        batches_mask = np.array_split(mask0, n/impute_bs)
        splits = np.array_split(range(n),n/impute_bs)
        xhat_fits=[]
        for it in range(len(batches_data)):
          b_data = torch.from_numpy(batches_data[it]).float().cuda()
          b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
          xhat_fit=miwae_impute(iota_x = b_data, mask = b_mask, L=L)
          xhat_fits = np.append(xhat_fits, {'xhat_fit': xhat_fit, 'obs_ids': splits[it]})
          #print(b_data[:4]); print(xhat_0[:4]); print(b_mask[:4]); print(mask[:4])
          b_xhat = xhat[splits[it],:]
          #b_xhat[batches_mask[it]] = np.mean(params_x['mean'].reshape([niw,-1]),axis=0).reshape([n,p])[splits[it],:][batches_mask[it]]   #  .cpu().data.numpy()[batches_mask[it]]  # keep observed data as truth
          b_xhat[~batches_mask[it]] = xhat_fit['xm'].cpu().data.numpy()[~batches_mask[it]] # just missing imputed
          xhat[splits[it],:] = b_xhat
        time_impute=np.append(time_impute,time.time()-t0_impute)
        err = mse(xhat,xfull,mask)
        mse_train_miss = np.append(mse_train_miss,np.array([err['miss']]),axis=0)
        mse_train_obs = np.append(mse_train_obs,np.array([err['obs']]),axis=0)
        
        zgivenx_flat = xhat_fit['zgivenx_flat'].cpu().data.numpy()   # L samples*batch_size x d (d: latent dimension)
        imp_weights=xhat_fit['imp_weights'].cpu().data.numpy()
        
        print('Observed MSE  %g' %err['obs'])     # observed values are not imputed
        print('Missing MSE  %g' %err['miss'])
        print('-----')
    saved_model={'encoder': encoder, 'decoder': decoder}
    mse_train={'miss':mse_train_miss,'obs':mse_train_obs}
    train_params = {'h':h, 'sigma':sigma, 'bs':bs, 'n_epochs':n_epochs, 'lr':lr, 'niw':niw, 'dim_z':dim_z, 'L':L, 'dec_distrib':dec_distrib, 'n_hidden_layers': n_hidden_layers}
    return {'train_params':train_params,'loss_fits':loss_fits,'xhat_fits':xhat_fits,'saved_model': saved_model,'zgivenx_flat': zgivenx_flat,'MIWAE_LB_epoch': MIWAE_LB_epoch,'time_train': time_train,'time_impute': time_impute,'imp_weights': imp_weights,'MSE': mse_train, 'xhat': xhat, 'mask': mask, 'norm_means':norm_means, 'norm_sds':norm_sds}
  else:
    # validating (hyperparameter values) or testing
    encoder=saved_model['encoder']
    decoder=saved_model['decoder']
    for ep in range(1,n_epochs):
      #perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
      #batches_data = np.array_split(xhat_0[perm,], n/bs)
      #batches_mask = np.array_split(mask[perm,], n/bs)
      #for it in range(len(batches_data)):
      #  optimizer.zero_grad()
      #  encoder.zero_grad()
      #  decoder_x.zero_grad()
      #  decoder_r.zero_grad()
      #  b_data = torch.from_numpy(batches_data[it]).float().cuda()
      #  b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
      #  loss = miwae_loss(iota_x = b_data,mask = b_mask)
      #  loss.backward()
      #  optimizer.step()
      #time_train=np.append(time_train,time.time()-t0_train)
      perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
      batches_data = np.array_split(xhat_0[perm,], n/bs)
      batches_mask = np.array_split(mask0[perm,], n/bs)
      splits = np.array_split(perm,n/bs)
      batches_loss = []
      encoder.zero_grad(); decoder.zero_grad()
      
      # loss_fit=miwae_loss(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda())
      # MIWAE_LB=(-np.log(K)-loss_fit['neg_bound'].cpu().data.numpy())
      # print('Epoch %g' %ep)
      # print('MIWAE likelihood bound  %g' %MIWAE_LB) # Gradient step      
      
      loss_fits = []
      for it in range(len(batches_data)):
        b_data = torch.from_numpy(batches_data[it]).float().cuda()
        b_mask = torch.from_numpy(batches_mask[it]).float().cuda()

        loss_fit = miwae_loss(iota_x = b_data, mask = b_mask)
        loss = loss_fit['neg_bound']
        batches_loss = np.append(batches_loss, loss.cpu().data.numpy())
        
        loss_fit.pop("neg_bound")
        loss_fits = np.append(loss_fits, {'loss_fit': loss_fit, 'obs_ids': splits[it]})
       
      total_loss = -np.sum(batches_loss)   # negative of the total loss (summed over K & bs)
      MIWAE_LB = total_loss / (K*n) - np.log(K)
        
      ### Now we do the imputation
      # xhat_fit=miwae_impute(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda(),L=L)
      # time_impute=np.append(time_impute,time.time()-t0_impute)
      # xhat[~mask] = xhat_fit['xm'].cpu().data.numpy()[~mask]
      # #xhat = xhat_fit['xm'].cpu().data.numpy()    # observed values are not imputed
      t0_impute=time.time()
      batches_data = np.array_split(xhat_0, n/impute_bs)
      batches_mask = np.array_split(mask0, n/impute_bs)
      splits = np.array_split(range(n),n/impute_bs)
      xhat_fits = []
      for it in range(len(batches_data)):
        b_data = torch.from_numpy(batches_data[it]).float().cuda()
        b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
        xhat_fit=miwae_impute(iota_x = b_data, mask = b_mask, L=L)
        xhat_fits = np.append(xhat_fits, {'xhat_fit': xhat_fit, 'obs_ids': splits[it]})
        b_xhat = xhat[splits[it],:]
        #b_xhat[batches_mask[it]] = torch.mean(loss_fit['params_x']['mean'].reshape([niw,-1]),axis=0).reshape([n,p])[splits[it],:].cpu().data.numpy()[batches_mask[it]]  # keep observed data as truth
        b_xhat[~batches_mask[it]] = xhat_fit['xm'].cpu().data.numpy()[~batches_mask[it]] # just missing imputed
        xhat[splits[it],:] = b_xhat
      #xhat_fit=nimiwae_impute(iota_xfull = cuda_xfull, iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda(),covar_miss = torch_covars_miss,L=L,temp=temp_min)
      time_impute=np.append(time_impute,time.time()-t0_impute)

      err = mse(xhat,xfull,mask)
      mse_train_miss = np.append(mse_train_miss,np.array([err['miss']]),axis=0)
      mse_train_obs = np.append(mse_train_obs,np.array([err['obs']]),axis=0)
      zgivenx_flat = xhat_fit['zgivenx_flat'].cpu().data.numpy()   # L samples*batch_size x d (d: latent dimension)
      imp_weights = xhat_fit['imp_weights'].cpu().data.numpy()
      print('Observed MSE  %g' %err['obs'])   # observed values are not imputed
      print('Missing MSE  %g' %err['miss'])
      print('-----')
    mse_test={'miss':err['miss'],'obs':err['obs']}
    saved_model={'encoder': encoder, 'decoder': decoder}
    return {'loss_fits':loss_fits,'xhat_fits':xhat_fits,'zgivenx_flat': zgivenx_flat,'saved_model': saved_model,'LB': MIWAE_LB,'time_impute': time_impute,'imp_weights': imp_weights,'MSE': mse_test, 'xhat': xhat, 'xfull': xfull, 'mask': mask, 'norm_means':norm_means, 'norm_sds':norm_sds}  

