import abc
import numpy as np

import torch
from torch import nn

from torch.autograd.functional import jacobian

import os.path as osp
import pathlib
import pandas

from LFIS.util.util import load_data
from LFIS.util.util_VAE import TorchConvDecoder

class Density_base(metaclass=abc.ABCMeta):
  """Abstract base class from which all log densities should inherit."""

  def __init__(self, ndim):
    self.ndim = ndim

  def __call__(self, x):
    output = self.evaluate_log_density(x)
    return output

  @abc.abstractmethod
  def evaluate_log_density(self, x): pass

  def comput_batch_jac(self, func, x):
    def _func_sum(x):
      return func(x).sum(dim=0)
    return jacobian(_func_sum, x,create_graph = False)

  def dlog_density(self,x):
    output = self.comput_batch_jac(self.evaluate_log_density, x)
    output = output[0]
    return output

class MultiNormDist(Density_base):
  def __init__(self, ndim, mu, sigma2, device = 'cpu'):
    super().__init__(ndim)

    assert ndim > 1
    
    self.device = device
    
    self.mu = mu.to(device)
    self.sigma2 = sigma2.to(device)

    self.det = torch.det(self.sigma2)
    self.sigmainv = torch.inverse(self.sigma2)   
    self.cholesky_gram = torch.linalg.cholesky(self.sigma2)

    self.const = torch.tensor(np.log(2*np.pi)*(-0.5*self.ndim)).to(device) - 0.5*torch.log(self.det)

  def evaluate_log_density(self,x):
    xvec = (x-self.mu.reshape([1,self.ndim])).reshape([-1,1,self.ndim])
    return (self.const+(-0.5*torch.matmul(torch.matmul(xvec,self.sigmainv),xvec.permute(0,2,1)))).reshape([-1,1])

  def dlog_density_fast(self,x):
    xvec = (x-self.mu.reshape([1,self.ndim])).reshape([-1,1,self.ndim])
    return (-torch.matmul(xvec,self.sigmainv)).reshape([-1,self.ndim])

  def sample(self,nsample):
    white = torch.randn(nsample, self.ndim).to(self.device)
    return torch.einsum("ij,bj->bi", self.cholesky_gram, white) + self.mu

class NormDist(Density_base):
  def __init__(self, mu, sigma2, device = 'cpu'):
    super().__init__(1)

    self.device = device
    
    self.mu = mu.to(device)
    self.sigma2 = sigma2.to(device)

    self.const = torch.tensor(np.log(2*np.pi)*(-0.5)).to(device) - 0.5*torch.log(self.sigma2)

  def evaluate_log_density(self,x):
    xvec = x - self.mu
    return self.const + (-0.5*xvec**2/self.sigma2)

  def sample(self,nsample):
    white = torch.randn(nsample, self.ndim).to(self.device)
    return white*(self.sigma2**0.5) + self.mu

class MG2D(Density_base):
  def __init__(self, ndim,  device = 'cpu', grid = 5, sigma2 = 0.3):
    super().__init__(ndim)
    assert ndim == 2
    self.center = torch.tensor([[0+grid*i,0+grid*j] for i in range(-1,2) for j in range(-1,2) ]).to(device)
    self.sigma2 = torch.eye(2).to(device)*sigma2
    self.det = torch.det(self.sigma2)
    self.sigmainv = torch.inverse(self.sigma2)
    self.device = device
    
    self.const = torch.tensor(np.log(2*np.pi)*(-0.5*self.ndim)).to(device) - 0.5*torch.log(self.det)
    self.const2 = torch.log(torch.tensor(9.0)).to(device)


  def evaluate_log_density(self,x):
    out =[]
    for i in range(len(self.center)):
      xvec = (x-self.center[i].reshape([1,self.ndim])).reshape([-1,1,self.ndim])
      out.append((self.const+(-0.5*torch.matmul(torch.matmul(xvec,self.sigmainv),xvec.permute(0,2,1)))).reshape([-1,1]))
    out = torch.stack(out)
    return torch.logsumexp(out,axis=0) - self.const2

class MG2DWeighted(Density_base):
  def __init__(self, ndim,  device = 'cpu', grid = 5, sigma2 = 0.3,
               weights = torch.tensor([0.025*9, 0.18*9, 0.025*9, 0.18*9, 0.18*9, 0.18*9, 0.025*9, 0.18*9, 0.025*9 ])):
    super().__init__(ndim)
    assert ndim == 2
    self.center = torch.tensor([[0+grid*i,0+grid*j] for i in range(-1,2) for j in range(-1,2) ]).to(device)
    self.sigma2 = torch.eye(2).to(device)*sigma2
    self.det = torch.det(self.sigma2)
    self.sigmainv = torch.inverse(self.sigma2)
    self.device = device
    
    self.const = torch.tensor(np.log(2*np.pi)*(-0.5*self.ndim)).to(device) - 0.5*torch.log(self.det)
    self.const2 = torch.log(torch.tensor(9.0)).to(device)

    self.weights = weights.to(self.device)
    self.logweights = torch.log(self.weights)


  def evaluate_log_density(self,x):
    out =[]
    for i in range(len(self.center)):
      xvec = (x-self.center[i].reshape([1,self.ndim])).reshape([-1,1,self.ndim])
      out.append((self.logweights[i] + self.const+(-0.5*torch.matmul(torch.matmul(xvec,self.sigmainv),xvec.permute(0,2,1)))).reshape([-1,1]))
    out = torch.stack(out)
    return torch.logsumexp(out,axis=0) - self.const2

class Funnel(Density_base):
  def __init__(self, ndim,  device = 'cpu'):
    super().__init__(ndim)
    self.sigma0 = torch.tensor(9.0).to(device)
    self.device = device
    self.ndim = ndim
    self.iden = torch.eye(self.ndim-1).to(self.device)
    self.piconst = torch.log(torch.tensor(2*torch.pi).to(self.device))
    
  def evaluate_log_density(self,x):
    sigma1_9 = torch.exp(x[:,:1]).reshape([-1,1,1])*self.iden#torch.eye(self.ndim-1).to(self.device)
    inv1_9 = torch.exp(-x[:,:1]).reshape([-1,1,1])*self.iden#torch.eye(self.ndim-1).to(self.device)

    xvec = x[:,1:].reshape([-1,1,self.ndim-1])
    return  (-0.5*(self.piconst+ torch.log(self.sigma0))+(-0.5*x[:,:1]**2/self.sigma0)-(self.ndim-1)/2*self.piconst-(self.ndim-1)*0.5*x[:,:1]+ (-0.5*torch.matmul(torch.matmul(xvec,inv1_9),xvec.permute(0,2,1))).reshape([-1,1])).reshape([-1,1])

  def evaluate_log_density_fast(self,x):
    #sigma1_9 = torch.exp(x[:,:1]).reshape([-1,1,1])*self.iden#torch.eye(self.ndim-1).to(self.device)
    #inv1_9 = torch.exp(-x[:,:1]).reshape([-1,1,1])*self.iden#torch.eye(self.ndim-1).to(self.device)

    xvec = x[:,1:].reshape([-1,1,self.ndim-1])
    return  (-0.5*(2*torch.pi*self.sigma0)+(-0.5*x[:,:1]**2/self.sigma0)-(self.ndim-1)/2*self.piconst-(self.ndim-1)*0.5*x[:,:1]  -0.5*(x[:,1:]**2*torch.exp(-x[:,:1])).sum(axis=1).reshape([-1,1]) ).reshape([-1,1])

  def dlog_density_fast(self,x):
    dx0 = -x[:,:1]/self.sigma0 -(self.ndim-1)*0.5 + 0.5*(x[:,1:]**2*torch.exp(-x[:,:1])).sum(axis=1).reshape([-1,1])
    dx1 = -x[:,1:]*torch.exp(-x[:,:1])
    return torch.cat([dx0, dx1], axis= 1)

class LGCP(Density_base):
  def __init__(self, ndim,  device = 'cpu'):
    super().__init__(ndim)
    self.ndim = ndim
    self.device = device
    
    self.file_path = osp.join(pathlib.Path(__file__).parents[1].resolve(), "dataset/df_pines.csv")
    df = pandas.read_csv(self.file_path)
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
        
        
    self.alpha=torch.tensor(1/M/M).to(device)
    sigma2=1.91
    self.mu = torch.tensor((np.log(126)-sigma2/2) * np.ones((M*M))).to(device)
    beta = 1/33

    self.Knew = sigma2*np.exp(-np.sqrt((xx_.reshape([-1,1])-xx_.reshape([1,-1]))**2 + (yy_.reshape([-1,1])-yy_.reshape([1,-1]))**2)/M/beta)
    self.Kinv = torch.tensor(np.linalg.inv(self.Knew)).to(device)

    self.kcons = torch.tensor(np.log(2*np.pi)*(-ndim/2) - 0.5* np.log(np.linalg.det(self.Knew))).to(device)
        
    self.yval = torch.tensor(n_.reshape([1, M*M])).to(device)

    
  def evaluate_log_density(self,x): #### Bayes formulation
    out =  (x*self.yval).sum(axis=1) - self.alpha* torch.sum(torch.exp(x), axis=1)
    return out.reshape([-1,1])



class LogisticRegression(Density_base):
  def __init__(self, ndim, case = 'ionosphere',  device = 'cpu'):
    super().__init__(ndim)
    self.ndim = ndim
    self.device = device
    if case == 'ionosphere':
      self.file_path = osp.join(pathlib.Path(__file__).parents[1].resolve(), "dataset/ionosphere_full.pkl")
    elif case == 'sonar':
      self.file_path = osp.join(pathlib.Path(__file__).parents[1].resolve(), "dataset/sonar_full.pkl")
      
    x , y = load_data(name = self.file_path)
        
    self.datax = torch.tensor(x).to(self.device)
    self.datay = torch.tensor(y).to(self.device)

  def evaluate_log_density(self,x): #### Bayes formulation
    p = torch.sigmoid((x.reshape([-1,1,self.ndim])*self.datax).sum(axis=-1))
    out = (self.datay*torch.log(p+10e-18) + (1-self.datay)*torch.log(1-p+10e-18)).sum(axis=-1)
    return out.reshape([-1,1])
#    return self.binary_cross_entropy_from_logits(logits)

  def binary_cross_entropy_from_logits(self, logits):

    labels = self.datay
    max_logits_zero = nn.ReLU()(logits)
    negative_abs_logits = -torch.abs(logits)
    terms = max_logits_zero - logits*labels + nn.Softplus()(negative_abs_logits)
    return - torch.sum(terms, axis=-1)


class VAE(Density_base):
  def __init__(self, ndim,  device = 'cpu'):
    super().__init__(ndim)
    self.ndim = ndim
    self.device = device

    self.VAE = TorchConvDecoder(device = self.device).to(self.device)

  def evaluate_log_density(self,x): #### Bayes formulation                                                                                            
    out = self.VAE.logL(x)
    return out.reshape([-1,1])

   
