import torch
from torch import nn
from torch.autograd.functional import jacobian

class NN_Base(nn.Module):
    def __init__(self,ndim, nhidden,nlayer):
        super().__init__()
        self.ndim = ndim

        self.layers = nn.Sequential(
            nn.Linear(ndim, nhidden),
            *[
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(nhidden, nhidden),
                )
                for _ in range(nlayer)
            ],
            nn.SiLU(),
            nn.Linear(nhidden, ndim)
        )

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.00)
        self.mean = nn.Parameter(torch.zeros(ndim), requires_grad = False ) 
    def forward(self,x):
        return self.layers(x - self.mean)

    def divergence(self,x):
        # compute the divergence of the flow field
        def _func_sum(x):
            return self.forward(x).sum(dim=0) # sum over the batches
        out=jacobian(_func_sum, x,create_graph =True,vectorize=True)
        div = out.diagonal(offset=0, dim1=-1, dim2=-3).sum(-1)
        return div

    def divergence_nograph(self,x):
        # compute the divergence of the flow field
        def _func_sum(x):
            return self.forward(x).sum(dim=0) # sum over the batches
        out=jacobian(_func_sum, x,create_graph =False,vectorize=True)
        div = out.diagonal(offset=0, dim1=-1, dim2=-3).sum(-1)
        return div

    def zerolayer(self):
        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.0)

class NN_Skip(nn.Module):
    def __init__(self,ndim, nhidden,nlayer):
        super().__init__()
        self.ndim = ndim

        self.layers = nn.Sequential( nn.Sequential(
            nn.Linear(ndim, nhidden),nn.SiLU(),),
            *[
                nn.Sequential( nn.Linear(nhidden + ndim, nhidden),
                               nn.SiLU(), )
                for _ in range(nlayer)
            ],
            nn.Linear(nhidden + ndim, ndim)
        )

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.00)
        self.mean = nn.Parameter(torch.zeros(ndim), requires_grad = False ) 
    def forward(self,x):
        x0 = x - self.mean
        out  = self.layers[0](x0)
        for  i in range(1,len(self.layers)):
            out = torch.cat([out, x0], axis = 1)
            out = self.layers[i](out)
        
        return out

    def divergence(self,x):
        # compute the divergence of the flow field
        def _func_sum(x):
            return self.forward(x).sum(dim=0) # sum over the batches
        out=jacobian(_func_sum, x,create_graph =True,vectorize=True)
        div = out.diagonal(offset=0, dim1=-1, dim2=-3).sum(-1)
        return div

    def divergence_nograph(self,x):
        # compute the divergence of the flow field
        def _func_sum(x):
            return self.forward(x).sum(dim=0) # sum over the batches
        out=jacobian(_func_sum, x,create_graph =False,vectorize=True)
        div = out.diagonal(offset=0, dim1=-1, dim2=-3).sum(-1)
        return div

    def zerolayer(self):
        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.0)

class IndependentLinear(nn.Module):
    def __init__(self,ndim):
        super().__init__()
        self.ndim = ndim

        self.l1_w = torch.nn.parameter.Parameter(torch.zeros((ndim,)), requires_grad=True)
        self.l1_b = torch.nn.parameter.Parameter(torch.zeros((ndim,)), requires_grad=True)

        self.mean = nn.Parameter(torch.zeros(ndim), requires_grad = False ) 
    def forward(self,x):
        out = self.l1_w*(x-self.mean) + self.l1_b
        return out

    def divergence(self,x):
        # compute the divergence of the flow field
        return torch.sum(self.l1_w)

    def divergence_nograph(self,x):
        # compute the divergence of the flow field
        return torch.sum(self.l1_w)

class Linear(nn.Module):
    def __init__(self,ndim):
        super().__init__()
        self.ndim = ndim
        self.l1 = nn.Linear(ndim, ndim)
        self.mean = nn.Parameter(torch.zeros(ndim), requires_grad = False ) 
        self.l1.weight.data = self.l1.weight.data*0.001
        self.l1.bias.data = self.l1.bias.data*0.001
    def forward(self,x):
        out = self.l1(x)
        return out

    def divergence(self,x):
        # compute the divergence of the flow field
        def _func_sum(x):
            return self.forward(x).sum(dim=0) # sum over the batches
        out=jacobian(_func_sum, x,create_graph =True,vectorize=True)
        div = out.diagonal(offset=0, dim1=-1, dim2=-3).sum(-1)
        return div
