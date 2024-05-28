
from copy import deepcopy
import torch
from torch import nn

import os
import os.path as osp
import pathlib
import pandas
import pickle

import torch
import numpy as np


torch.set_default_dtype(torch.float32)

class TorchConvDecoder(nn.Module):
  """A residual network decoder with logit outputs."""

  def __init__(self, device = 'cpu'):
    super().__init__()

    self.device = device
    self.ckpt_filename = osp.join(pathlib.Path(__file__).parents[1].resolve(), "dataset/vae.pickle")

    self._vae_params = self._get_vae_params(self.ckpt_filename)

    self.testimage_filename = osp.join(pathlib.Path(__file__).parents[1].resolve(), "dataset/VAE_testimage.npz")
    loaddata = np.load(self.testimage_filename)
    test_image = loaddata['image']

    
    self.deconv_a = nn.ConvTranspose2d(32,64, kernel_size=(3, 3), stride=(2, 2))
    self.deconv_b = nn.ConvTranspose2d(64,32, kernel_size=(3, 3), stride=(2, 2))
    self.deconv_c = nn.ConvTranspose2d(32, 1,  kernel_size=(3, 3), stride=(1, 1))
    self.act = nn.ReLU()
    linear_features = 7 * 7 * 32
    self.linear = nn.Linear(30, linear_features)
    self.load_param()
    self.test_image = torch.tensor(test_image,dtype=torch.float32).to(self.device)
      
  def _get_vae_params(self, ckpt_filename):
    with open(ckpt_filename, "rb") as f:
      vae_params = pickle.load(f)
    return vae_params
  def load_param(self):
    self.linear.weight = nn.Parameter(torch.tensor(self._vae_params['conv_vae/~/conv_decoder/linear']['w']).permute(1,0), requires_grad=False)
    self.linear.bias = nn.Parameter(torch.tensor(self._vae_params['conv_vae/~/conv_decoder/linear']['b']), requires_grad=False)
    self.deconv_a.weight = nn.Parameter(torch.flip(torch.tensor(self._vae_params['conv_vae/~/conv_decoder/conv2_d_transpose']['w']).permute(3,2,0,1), [2,3])
                                        , requires_grad=False)
    self.deconv_a.bias = nn.Parameter(torch.tensor(self._vae_params['conv_vae/~/conv_decoder/conv2_d_transpose']['b'])
                                        , requires_grad=False)

    self.deconv_b.weight = nn.Parameter(torch.flip(torch.tensor(self._vae_params['conv_vae/~/conv_decoder/conv2_d_transpose_1']['w']).permute(3,2,0,1), [2,3])
                                        , requires_grad=False)
    self.deconv_b.bias = nn.Parameter(torch.tensor(self._vae_params['conv_vae/~/conv_decoder/conv2_d_transpose_1']['b'])
                                        , requires_grad=False)

    self.deconv_c.weight = nn.Parameter(torch.flip(torch.tensor(self._vae_params['conv_vae/~/conv_decoder/conv2_d_transpose_2']['w']).permute(3,2,0,1), [2,3])
                                        , requires_grad=False)
    self.deconv_c.bias = nn.Parameter(torch.tensor(self._vae_params['conv_vae/~/conv_decoder/conv2_d_transpose_2']['b'])
                                        , requires_grad=False)
  def forward(self, z):
    progress = self.linear(z)
    nn.LayerNorm(progress.shape).to(self.device)(progress)
    out = torch.reshape(progress, (-1, 7, 7, 32)).permute(0,3,1,2)

    out = self.act(self.deconv_a(out))
    out = out[:,:,:-1,:-1]
    out = self.act(self.deconv_b(out))
    out = out[:,:,:-1,:-1]
    out = self.deconv_c(out)
    out = out[:,:,1:-1,1:-1]
    return out.permute([0,2,3,1])

  def binary_cross_entropy_from_logits(self, logits):
    """Numerically stable implementation of binary cross entropy with logits."""
    labels = self.test_image
    max_logits_zero = nn.ReLU()(logits)
    negative_abs_logits = -torch.abs(logits)
    terms = max_logits_zero - logits*labels + nn.Softplus()(negative_abs_logits)
    return torch.logsumexp(torch.sum(-terms, axis=(2,3,4)), axis=0)

  def logL(self, z):
    logits = self.forward(z)
    loglike =  self.binary_cross_entropy_from_logits(logits)
    return loglike

  def comput_batch_jac_nograph(self, func, x):
    def _func_sum(x):
        return func(x).sum(dim=0)
    return jacobian(_func_sum, x,create_graph =False)

  def dlogL(self, z):
    out = self.comput_batch_jac_nograph(self.logL, z)
    return out
