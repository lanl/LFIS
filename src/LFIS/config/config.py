import os
from pathlib import Path
import yaml
from box.exceptions import BoxValueError
from box import ConfigBox

import torch

from LFIS.density.density import (
    MG2D, MultiNormDist,
    Funnel, LGCP,
    LogisticRegression,
    VAE)
    
from LFIS.schedule.schedule import CosSchedule, LinearSchedule, QuadraticSchedule

from LFIS.model.model import NN_Base, NN_Skip, IndependentLinear

from ml_collections import config_dict as configdict

from LFIS.trainer.trainer import (
    train_withoutweight,
    train_resample,
    train_withweight_resample)

from LFIS.util.util import read_yaml, LGCP_prior
    
def get_baseconfig() -> configdict.ConfigDict:
    config = configdict.ConfigDict()

    config.device = 'cuda:1' ## 'cuda' or' cpu'
    config.task = 'train'
    config.problemtype = 'transform'
    
    config.ndim = 2
    config.nstep = 256
    config.dtmax = 1.0/config.nstep
    
    config.cfl = None
    config.schedule = CosSchedule()

    config.train = configdict.ConfigDict()
    config.train.method = train_withoutweight
    config.train.lr = 0.01
    config.train.patience= 200
    config.train.epoch_min = 0
    config.train.nsample = 600000
    config.train.nbatch = 20000
    config.train.threshold = 0.5e-3

    config.density = configdict.ConfigDict()

    return config

def setup_config(config:configdict.ConfigDict, case:str) -> configdict.ConfigDict:

#    path_to_yaml = os.path.join(Path(__file__).parent.resolve(),'configs',case+'.yaml')
    path_to_yaml = os.path.join("src/LFIS/configs",case+".yaml")
    cfgyaml = read_yaml(path_to_yaml)

    config.train.lr = cfgyaml.lr
    config.train.epoch = cfgyaml.epoch
    config.train.nsample = cfgyaml.nsample
    
    if case == 'MG2D':
        mu_prior = torch.tensor([0.0, 0.0])
        sigma2_prior = torch.eye(2)
        config.density.prior = MultiNormDist(config.ndim, mu_prior, sigma2_prior, device = config.device)

        config.density.target = MG2D(config.ndim, device=config.device, grid= 1, sigma2 = 0.3/25)

        config.nnmodel = NN_Base(config.ndim,64,2)

    return config

class get_configuration():
    def __init__(self):
        self.path = Path(__file__).parent.resolve()

    def setup_config(self, config:configdict.ConfigDict, case:str) -> configdict.ConfigDict:

        path_to_yaml = os.path.join(self.path,'config_file',case+'.yaml')
    
        cfgyaml = read_yaml(path_to_yaml)

        config.train.lr = cfgyaml.lr
        config.train.epoch = cfgyaml.epoch
        config.train.nsample = cfgyaml.nsample
    
        if case == 'MG2D':
            mu_prior = torch.tensor([0.0, 0.0])
            sigma2_prior = torch.eye(2)
            config.case = 'Mode-separated Gaussian Mixture'
            config.density.prior = MultiNormDist(config.ndim, mu_prior, sigma2_prior, device = config.device)
            
            config.density.target = MG2D(config.ndim, device=config.device, grid= 1, sigma2 = 0.3/25)

            config.nnmodel = NN_Base(config.ndim,64,2)

        elif case == 'funnel':
            
            config.ndim = 10
            mu_prior = torch.zeros(config.ndim)
            sigma2_prior = torch.eye(config.ndim)
            config.case = '10-Dimensional Funnel Distribution'
            config.density.prior = MultiNormDist(config.ndim, mu_prior, sigma2_prior, device = config.device)
            
            config.density.target = Funnel(config.ndim, device=config.device)

            config.nnmodel = NN_Base(config.ndim,64,2)
            
        elif case == 'LGCP':

            config.problemtype = 'bayes'
            config.ndim = 1600
            
            mu_prior, sigma2_prior = LGCP_prior()
            config.case = 'Log Gaussian Cox problem with Gaussian prior and Poisson process likelihood function'
            config.density.prior = MultiNormDist(config.ndim, mu_prior, sigma2_prior, device = config.device)
            
            config.density.target = LGCP(config.ndim, device=config.device)
            config.train.method = train_resample
            config.nnmodel = IndependentLinear(config.ndim)

        elif case == 'LogReg':

            config.problemtype = 'bayes'
            config.train.nbatch = cfgyaml.nbatch
            
            if cfgyaml.regcase == 'ionosphere':
                config.ndim = 35
            elif cfgyaml.regcase == 'sonar':
                config.ndim = 61
            else:
                print('Logistic regression problem does not exist!')
                exit()

            mu_prior = torch.zeros(config.ndim)
            sigma2_prior = torch.eye(config.ndim)                

            config.case = 'Logistic regression Bayesian: ' + cfgyaml.regcase
            config.density.prior = MultiNormDist(config.ndim, mu_prior, sigma2_prior, device = config.device)

            config.density.target = LogisticRegression(config.ndim, case = cfgyaml.regcase,  device=config.device)
            config.train.method = train_withweight_resample
            config.nnmodel = NN_Skip(config.ndim,64,2)

        elif case == 'VAE':

            config.problemtype = 'bayes'
            config.train.nbatch = cfgyaml.nbatch
            
            config.ndim = 30
           
            
            mu_prior = torch.zeros(config.ndim)
            sigma2_prior = torch.eye(config.ndim)                

            config.case = 'Latent space of VAE'
            config.density.prior = MultiNormDist(config.ndim, mu_prior, sigma2_prior, device = config.device)

            config.density.target = VAE(config.ndim,  device=config.device)
            config.train.method = train_withweight_resample
            config.nnmodel = NN_Base(config.ndim,64,2)

        else:
            print('No configurable case')
            exit()

            
        return config
        
if __name__ == '__main__':
    cfg = get_baseconfig()
    test_config = get_configuration()
    config = test_config.setup_config(cfg, 'MG2D')

    cfg = get_baseconfig()
    test_config = get_configuration()
    config = test_config.setup_config(cfg, 'funnel')

    cfg = get_baseconfig()
    test_config = get_configuration()
    config = test_config.setup_config(cfg, 'LGCP')

    cfg = get_baseconfig()
    test_config = get_configuration()
    config = test_config.setup_config(cfg, 'LogReg')

    cfg = get_baseconfig()
    test_config = get_configuration()
    config = test_config.setup_config(cfg, 'VAE')

    
