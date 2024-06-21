
from copy import deepcopy
from torch.optim import Adam

import os
import os.path as osp
import pathlib
import pandas
import pickle

import torch
import numpy as np

from LFIS.util.util import print_status

from LFIS.LF.LF import LF_base

from time import time
torch.set_default_dtype(torch.float64)

def train_withoutweight(config):
    """
    Train Liouville flow without using IS(weight)

    """

    
    t_init,t_end = torch.tensor(0.0), torch.tensor(1.0)

    t = t_init

#########Setup basics ##############################
    LFmodel = LF_base(config)
    LFmodel.to(config.device)
    cfl = config.cfl
    dtmax = config.dtmax

####### Setup optimization ############
    lr = config.train.lr 
    optimizer = Adam(list(LFmodel.flow.parameters()), lr=lr)

    epoch = config.train.epoch
    patience = config.train.patience
    epoch_min = config.train.epoch_min
    epsilon = 1e-6
    threshold = config.train.threshold
    
    nsample = config.train.nsample
    nbatch = config.train.nbatch
    
#### Setting up output##########
    flowlist = []
    tlist = []
    losslist = []

    output = {'flow': None,'time':None, 'losslog': None}
##############################################
    x = LFmodel.sample_noweight( nsample, flowlist, tlist)
    w = torch.zeros(nsample).to(x.device)
    batch = np.arange(nsample)
    while t < t_end - epsilon:


        optimizer.param_groups[0]['lr'] = lr
        counter = 0
        bestLoss = 1e8

        LFmodel.flow.mean = torch.nn.Parameter(x.mean(axis=0).detach(), requires_grad= False )       
        RHSmean = LFmodel.comput_RHSmean(x, w, t)

#        if t > 0.8*t_end: LFmodel.flow.zerolayer()
        print(t, RHSmean)
        for i in range(epoch):
          np.random.shuffle(batch)
          xbatch = x[batch[:nbatch]]

          optimizer.zero_grad()

          loss, ploss = LFmodel.comput_eqloss(xbatch, t, RHSmean = RHSmean)
          loss.backward()

          if i%200 == 0: print_status(t, loss,  ploss)

          if t == 0: break
          if ploss < threshold:
              print_status(t, loss,  ploss)
              break
############ Check learning rate ##########
          if loss.detach().cpu().numpy() < bestLoss:
             bestLoss = loss.detach().cpu().numpy()
             counter = 0
          else:
            if i > epoch_min: 
                counter += 1
          if counter > patience:
              print('===Reducing LR====')
              optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2
              counter = 0
          if optimizer.param_groups[0]['lr'] < lr/40:
              print('====LR too small, stop traning ====')
              break
          optimizer.step()

##############################
        print('Complete Training Flow at time {:.4f}'.format(t))
        if cfl != None:
          with torch.no_grad():
            vel = LFmodel.flow(x).detach()
            velmax = vel.pow(2).sum(axis=1).max().cpu()**0.5
          
            dt = torch.min(torch.stack([cfl/velmax, dtmax]))
        else:
            dt = dtmax
        if t+dt >= 1: dt = 1-t

        x, w = samples_step(LFmodel, x, w, t, dt )
        w = torch.zeros(nsample).to(x.device)

        t = t+dt
        losslist.append(loss.detach())
        tlist.append(deepcopy(t))
        flowlist.append(deepcopy(LFmodel.flow))

    output['flow'] = flowlist
    output['time'] = tlist
    output['losslog'] = losslist
    return output


def train_resample(config): #LFmodel, lr = 0.01, cfl = None, dtmax = torch.tensor(0.01), epoch = 2000, nsample = 10000):

    """
    Train Liouville flow without using IS(weight) with resampling

    Resampling: generat new samples using learned flow for learning the flow at the next step


    """

    t_init,t_end = torch.tensor(0.0), torch.tensor(1.0)

    t = t_init

#########Setup basics ##############################                                                                                                    
    LFmodel = LF_base(config)
    LFmodel.to(config.device)
    cfl = config.cfl
    dtmax = config.dtmax

####### Setup optimization ############                                                                                                                 
    lr = config.train.lr
    optimizer = Adam(list(LFmodel.flow.parameters()), lr=lr)

    epoch = config.train.epoch
    patience = config.train.patience
    epoch_min = config.train.epoch_min
    epsilon = 1e-6
    threshold = config.train.threshold

    nsample = config.train.nsample
    nbatch = config.train.nbatch


#### Setting up output##########
    flowlist = []
    tlist = []
    losslist = []

    output = {'flow': None,'time':None, 'losslog': None}
###############################################
    while t < t_end - epsilon:
        optimizer.param_groups[0]['lr'] = lr
        counter = 0
        bestLoss = 1e8

        x = LFmodel.sample_noweight( nsample, flowlist, tlist)
        LFmodel.flow.mean = torch.nn.Parameter(x.mean(axis=0).detach(), requires_grad= False )       

        for i in range(epoch):
          if i%2 == 0:
              x = LFmodel.sample_noweight( nsample, flowlist, tlist)

          optimizer.zero_grad()
          loss, ploss = LFmodel.comput_eqloss(x, t)
 
          loss.backward()

          if i%200 == 0: print_status(t, loss,  ploss)

          if t == 0: break
          if ploss < 1e-3:
              print_status(t, loss,  ploss)
              break
############ Check learning rate ##########
          if loss.detach().cpu().numpy() < bestLoss:
             bestLoss = loss.detach().cpu().numpy()
             counter = 0
          else:
            if i > epoch_min: 
                counter += 1
          if counter > patience:
              print('===Reducing LR====')
              optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2
              counter = 0
          if optimizer.param_groups[0]['lr'] < lr/20:
              print('====LR too small, stop traning ====')
              break
          optimizer.step()

##############################
        print('Complete Training Flow at time {:.4f}'.format(t))
        if cfl != None:
          with torch.no_grad():
            vel = LFmodel.flow(x).detach()
            velmax = vel.pow(2).sum(axis=1).max().cpu()**0.5
          
            dt = torch.min(torch.stack([cfl/velmax, dtmax]))
        else:
            dt = dtmax
        if t+dt >= 1: dt = 1-t
        t = t+dt
        losslist.append(loss.detach())
        tlist.append(deepcopy(t))
        flowlist.append(deepcopy(LFmodel.flow))

    output['flow'] = flowlist
    output['time'] = tlist
    output['losslog'] = losslist
    return output

def train_withweight_resample(config):
    """
    Train Liouville flow using IS(weight) and resampling every few epoches 
    """

    t_init,t_end = torch.tensor(0.0), torch.tensor(1.0)

    t = t_init
#########Setup basics ##############################                                                                                                
                                                                                                                                                      
    LFmodel = LF_base(config)
    LFmodel.to(config.device)
    cfl = config.cfl
    dtmax = config.dtmax

####### Setup optimization ############                                                                                                              
                                                                                                                                                      
    lr = config.train.lr
    optimizer = Adam(list(LFmodel.flow.parameters()), lr=lr)

    epoch = config.train.epoch
    patience = config.train.patience
    epoch_min = config.train.epoch_min
    epsilon = 1e-6
    threshold = config.train.threshold

    nsample = config.train.nsample
    nbatch = config.train.nbatch

    
#### Setting up output##########
    flowlist = []
    tlist = []
    losslist = []

    output = {'flow': None,'time':None, 'losslog': None}

#################################
    x = LFmodel.sample_noweight( nsample, flowlist, tlist)
    w = torch.zeros(nsample).to(x.device)
    
    batch = np.arange(nsample)
    while t < t_end - epsilon:
        optimizer.param_groups[0]['lr'] = lr
        
        counter = 0
        bestLoss = 1e8

        LFmodel.flow.mean = torch.nn.Parameter(x.mean(axis=0).detach(), requires_grad= False )       
        RHSmean = LFmodel.comput_RHSmean(x, w, t)

        #if t > 0.8*t_end: LFmodel.flow.zerolayer()
        print(t, RHSmean)
        for i in range(epoch):

          if i%2 == 0:
              xbatch = LFmodel.sample_noweight( nbatch, flowlist, tlist)
          optimizer.zero_grad()
          loss, ploss = LFmodel.comput_eqloss(xbatch, t, RHSmean = RHSmean)
          loss.backward()

          if i%200 == 0: print_status(t, loss,  ploss)

          if t == 0: break
          if ploss < threshold:
              print_status(t, loss,  ploss)
              break
############ Check learning rate ##########
          if loss.detach().cpu().numpy() < bestLoss:
             bestLoss = loss.detach().cpu().numpy()
             counter = 0
          else:
            if i > epoch_min: 
                counter += 1
          if counter > patience:
              print('===Reducing LR====')
              optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2
              counter = 0
          if optimizer.param_groups[0]['lr'] < lr/40:
              print('====LR too small, stop traning ====')
              break
          optimizer.step()

##############################
        print('Complete Training Flow at time {:.4f}'.format(t))
        if cfl != None:
          with torch.no_grad():
            vel = LFmodel.flow(x).detach()
            velmax = vel.pow(2).sum(axis=1).max().cpu()**0.5
          
            dt = torch.min(torch.stack([cfl/velmax, dtmax]))
        else:
            dt = dtmax
        if t+dt >= 1: dt = 1-t

        x, w = samples_step_batch(LFmodel, x, w, t, dt, nsample, nbatch, RHSmean = RHSmean)

        t = t+dt
        losslist.append(loss.detach())
        tlist.append(deepcopy(t))
        flowlist.append(deepcopy(LFmodel.flow))

    output['flow'] = flowlist
    output['time'] = tlist
    output['losslog'] = losslist
    return output


def samples_step(LFmodel, x, w, t, dt):
    with torch.no_grad():
        dw, RHSmean,vel = LFmodel.comput_weight( LFmodel.flow, x , w, t)
    
        w = w + dw*dt
        x = x + vel*dt
    return x, w

def samples_step_batch(LFmodel, x, w, t, dt, nsample, nbatch, RHSmean=None):
    with torch.no_grad():
        batch = np.arange(nsample)
        batch = batch.reshape([-1,nbatch])
        batch_number = batch.shape[0]

        LHS = torch.zeros(nsample).to(x.device)
        RHS = torch.zeros(nsample).to(x.device)

        for n in range(batch_number):
            xbatch = x[batch[n]]
            LHSbatch, RHSbatch, vel = LFmodel.comput_weight_batch(LFmodel.flow, xbatch, t )

            LHS[batch[n]] = LHSbatch
            RHS[batch[n]] = RHSbatch.flatten()

            x[batch[n]] = xbatch + vel*dt

        dw = (LHS-(RHS-RHSmean))
        w = w + dw*dt
    return x, w
