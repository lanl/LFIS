import os
from pathlib import Path
import yaml
from box.exceptions import BoxValueError
from box import ConfigBox

import torch

from density import MG2D, MultiNormDist

from schedule import CosSchedule, LinearSchedule, QuadraticSchedule

from model import NN_Base, NN_Skip

from LF import LF_base

from ml_collections import config_dict as configdict

from trainer import train_withoutweight

from util import read_yaml
# def read_yaml(path_to_yaml: Path) -> ConfigBox:
#     """reads yaml config file and returns
#     path_to_yaml (str): path like input

#     Returns:
#         ConfigBox: ConfigBox type
#     """
#     try:
#         with open(path_to_yaml) as yaml_file:
#             content = yaml.safe_load(yaml_file)
#             return ConfigBox(content)
#     except BoxValueError:
#         raise ValueError("yaml file is empty")
#     except Exception as e:
#         raise e
    
def get_baseconfig() -> configdict.ConfigDict:
    config = configdict.ConfigDict()

    config.device = 'cuda:0' ## 'cuda' or' cpu'
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

        path_to_yaml = os.path.join(self.path,'configs',case+'.yaml')
    
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
        
if __name__ == '__main__':
    cfg = get_baseconfig()
#    config = setup_config(cfg, 'MG2D')
    test_config = get_configuration()
    config = test_config.setup_config(cfg, 'MG2D')
# case_name = 'MG2D_128'
# device =  'cuda:1'
# ndim  = 2
# task = 'train'
# #################################                                                                                                      
# mu_prior = torch.tensor([0.0, 0.0])
# sigma2_prior = torch.eye(2)*5.5**2
# cfl = None
# steps = 128
# dtmax = torch.tensor(1.0/steps)

# ### Setting up the initial distribution and target distribution                                                                        
# prior = MultiNormDist(ndim,mu_prior, sigma2_prior, device = device )
# target = MG2D(2, device=device)


# ###### Setting up the schedule ##############                                                                                          
# tt = schedule.CosSchedule()

# ##### Setting up the NN model to parametrize the Liouville flow #####                                                                  
# flow_model  = model.NN_base(2,64,2)


#     return config
    
    
