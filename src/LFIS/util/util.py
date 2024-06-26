import os
import os.path as osp
import pathlib
from pathlib import Path
import pandas
import pickle

from box import ConfigBox
from box.exceptions import BoxValueError
import yaml

import torch
import numpy as np

from copy import deepcopy
torch.set_default_dtype(torch.float64)

def save_file(folder, output):
    cpath = Path(__file__).parents[3].resolve()
    path = os.path.join(cpath, 'output', folder)
    try:
        os.mkdir(path)
    except:
        print('Folder exist, overiding files')
    nflow = len(output['flow'])
    for i in range(nflow):
        torch.save(output['flow'][i].state_dict(),path+"/LF_"+str(i)+".params")
    np.savez(path+'/LF_timestep.npz', t = output['time'])

def load_file(folder, flow):
    cpath = Path(__file__).parents[3].resolve()
    path = os.path.join(cpath, 'output', folder)
    
    loaded = np.load(path+'/LF_timestep.npz')
    t = torch.tensor(loaded['t'])
    flowlist = []
    for i in range(len(t)):
        flow.load_state_dict(torch.load(path+"/LF_"+str(i)+".params"))
        flowlist.append(deepcopy(flow))
    output = {'flow':flowlist, 'time':t}
    return output
    
def save_stat(folder, logstat, filename = 'LF_stats.npz'):
    cpath = Path(__file__).parents[3].resolve()
    path = os.path.join(cpath, 'output', folder)
    #path = 'output/'+folder
    try:
        os.mkdir(path)
    except:
        print('Folder exist, overiding files')
    np.savez(path+'/'+filename, samples = logstat['samples']
             , weight = logstat['weight'], logzmean = logstat['logzmean'], logzstd = logstat['logzstd'], logzlist = logstat['logzlist'])
    


def print_status( t, loss, ploss ):
    tnp = t.cpu().numpy()
    print('time = {:.4f}, loss = {:.4f}, percetage = {:.4f}%'.format(tnp,loss.detach().cpu().numpy(),ploss.cpu().numpy()*100  ))

          

def run_stat(LFmodel, output, nsample=2000, nruns = 30, withweight=True):
    logzruns = []
    logstat = {'logzmean': None, 'logzstd':None, 'samples': None, 'weight': None, 'logzlist': None}
    for i in range(nruns):
        is_nan = True
        while is_nan:
            if withweight:
                x,w,logz = LFmodel.sample_weight_method2(nsample, output['flow'], output['time'])
            else:
                x,w,logz = LFmodel.sample_woweight_method2(nsample, output['flow'], output['time'])
            is_nan = torch.isnan(logz)
        logzruns.append(logz.cpu().numpy())

    logstat['logzmean'] = np.stack(logzruns).mean()
    logstat['logzstd'] = np.stack(logzruns).std()
    logstat['samples'] = x.cpu().numpy()
    logstat['weight'] = torch.exp(w).cpu().numpy()
    logstat['logzlist'] = np.stack(logzruns)
    return logstat

def run_stat_batch(LFmodel, output, nsample=2000, nbatch = 200, nruns = 100):
    logzruns = []
    logstat = {'logzmean': None, 'logzstd':None, 'samples': None, 'weight': None, 'logzlist': None}
    for i in range(nruns):
        x,w,logz = LFmodel.sample_weight_batch_method2(nsample,nbatch, output['flow'], output['time'])
        logzruns.append(logz.cpu().numpy())

    logstat['logzmean'] = np.stack(logzruns).mean()
    logstat['logzstd'] = np.stack(logzruns).std()
    logstat['samples'] = x.cpu().numpy()
    logstat['weight'] = torch.exp(w).cpu().numpy()
    logstat['logzlist'] = np.stack(logzruns)
    return logstat



def LGCP_prior():
    ndim = 1600

    file_path = osp.join(pathlib.Path(__file__).parents[1].resolve(), "dataset/df_pines.csv")
    df = pandas.read_csv(file_path)
    x, y = np.array(df["data_x"]), np.array(df["data_y"])

    M = int(ndim**0.5)

    xgrid = np.linspace(x.min(),x.max(),M+1)
    ygrid = np.linspace(y.min(),y.max(),M+1)

    n = np.zeros((M,M)).astype('int')
    yy,xx= np.meshgrid(np.arange(M), np.arange(M))
    xx = xx.astype('int')
    yy = yy.astype('int')

    for i in range(len(x)):

      x_ind = np.where( (x[i]>=xgrid[:-1])& (x[i]<=xgrid[1:]) )[0][0]
      y_ind = np.where( (y[i]>=ygrid[:-1])& (y[i]<=ygrid[1:]) )[0][0]

      n[x_ind,y_ind]+=1
    xx_ = xx.reshape((M*M,))
    yy_ = yy.reshape((M*M,))
    n_ = n.reshape((M*M,))


    alpha=torch.tensor(1/M/M)
    sigma2=1.91
    mu = torch.tensor((np.log(126)-sigma2/2) * np.ones((M*M)))
    beta = 1/33

    Knew = sigma2*np.exp(-np.sqrt((xx_.reshape([-1,1])-xx_.reshape([1,-1]))**2 + (yy_.reshape([-1,1])-yy_.reshape([1,-1]))**2)/M/beta)
    Knew = torch.tensor(Knew)
    return mu, Knew


def pad_with_const(x):
  extra = np.ones((x.shape[0], 1))
  return np.hstack([extra, x])

def standardize_and_pad(x):
  mean = np.mean(x, axis=0)
  std = np.std(x, axis=0)
  std[std == 0] = 1.
  x = (x - mean) / std
  return pad_with_const(x)


def load_data(name="sonar_full.pkl"):
  with open(name, mode="rb") as f:
    x, y = pickle.load(f)
  y = (y + 1) // 2
  x = standardize_and_pad(x)
  return x, y


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
