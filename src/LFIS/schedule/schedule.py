import abc

import numpy as np

import torch

from torch.autograd.functional import jacobian


class Schedule_base(metaclass=abc.ABCMeta):
  """Abstract base class from which all log densities should inherit."""

  def __init__(self):pass

  def __call__(self, x):
    outf = self.tt(x)
    doutf = self.dtt(x)
    
    return outf, doutf

  @abc.abstractmethod
  def tt(self, x): pass

  @abc.abstractmethod
  def dtt(self, x): pass
  ## dt should be the equal to dt/dx


class CosSchedule(Schedule_base):
  def __init__(self):
    super().__init__()
  def tt(self,t):
    out = 0.5*(1.0-torch.cos(torch.pi*t))
    return out

  def dtt(self,t):
    out = (0.5*torch.pi*torch.sin(torch.pi*t))
    return out

class LinearSchedule(Schedule_base):
  def __init__(self):
    super().__init__()
  def tt(self,t):
    out = t 
    return out

  def dtt(self,t):
    out = torch.tensor(1.0)
    return out

class QuadraticSchedule(Schedule_base):
  def __init__(self):
    super().__init__()
  def tt(self,t):
    out = t*t 
    return out

  def dtt(self,t):
    out = 2*torch.tensor(t)
    return out

class BridgeQuadraticSchedule(Schedule_base):
  def __init__(self):
    super().__init__()
  def tt(self,t):
    out = -4*t*(t-1)   
    return out

  def dtt(self,t):
    out = -8*t +4
    return out

