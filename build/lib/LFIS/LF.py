import torch
from torch import nn
import numpy as np


class LF_base(nn.Module):
    def __init__(self,config): #
#    ndim, flow, schedule,  target, prior, setup = 'transform'): 
        super().__init__()
        self.ndim = config.ndim
        self.flow = config.nnmodel
        self.schedule = config.schedule
        self.target = config.density.target
        self.prior = config.density.prior

        self.setup = config.problemtype


    def comput_SandRHS(self,x, t):
        outf, doutf = self.schedule(t)

        logtarget = self.target.evaluate_log_density(x)
        dlogtarget = self.target.dlog_density(x)
        dlogprior = self.prior.dlog_density(x)
        
        if self.setup == 'transform':
            logprior = self.prior.evaluate_log_density(x)
            Score = outf * dlogtarget + (1-outf) * dlogprior
            RHS = - (doutf * logtarget  - doutf * logprior)
            
        elif self.setup == 'bayes':
            Score = outf * dlogtarget +  dlogprior
            RHS = - (doutf * logtarget)

        return Score, RHS

    def comput_SandRHS_test(self,x, t):
        outf, doutf = self.schedule(t)

        logtarget = self.target.evaluate_log_density_fast(x)
        dlogtarget = self.target.dlog_density_fast(x)
        dlogprior = self.prior.dlog_density_fast(x)
        
        logprior = self.prior.evaluate_log_density(x)
        Score = outf * dlogtarget + (1-outf) * dlogprior
        RHS = - (doutf * logtarget  - doutf * logprior)

        return Score, RHS

    def comput_RHSmean(self,x, w, t):
        outf, doutf = self.schedule(t)

        logtarget = self.target.evaluate_log_density(x)
        
        if self.setup == 'transform':
            logprior = self.prior.evaluate_log_density(x)
            RHS = - (doutf * logtarget  - doutf * logprior)
        elif self.setup == 'bayes':
            RHS = - (doutf * logtarget)
        RHSmean = (RHS.flatten()*torch.exp(w)/torch.exp(w).mean()).mean()
        return  RHSmean

    def comput_eqloss(self, x , t, RHSmean = None):
        
        Score, RHS = self.comput_SandRHS(x,t)
        div = self.flow.divergence(x)
        vel = self.flow(x)

        if RHSmean == None:
            RHSmean = RHS.mean()
        LHS = div + (Score*vel).sum(axis=1)
        
        eqloss =  (LHS - (RHS.flatten()-RHSmean))

        eqloss = torch.nan_to_num(eqloss, posinf=1.0, neginf = -1.0)
        eqloss = eqloss.pow(2).mean()
        
        return eqloss, (eqloss/torch.nan_to_num(RHS).var()).detach()

    
    def sample_noweight(self, nsample, modellist, tlist):
        ## This function resample using trained network
        assert len(modellist) == len(tlist)
        x = self.prior.sample(nsample)
       
        t = 0.0
        with torch.no_grad():
            for i in range(len(modellist)):
               
                vel = modellist[i](x)
                x = x + vel*(tlist[i]-t)
                t = tlist[i]
                
        return x

    def comput_weight(self, flow, x , w, t):
        with torch.no_grad():
            Score, RHS = self.comput_SandRHS(x,t)
            div = flow.divergence_nograph(x)
            vel = flow(x)
            LHS = div + (Score*vel).sum(axis=1)

            RHSmean = ((torch.exp(w)/torch.exp(w).mean())*RHS.flatten()).mean()
            dw =  (LHS - (RHS.flatten()-RHSmean))
        return dw, RHSmean, vel

    def comput_weight_method2(self, flow, x , w, t):


        with torch.no_grad():
            Score, RHS = self.comput_SandRHS(x,t)
            div = flow.divergence_nograph(x)
            vel = flow(x)
            LHS = div + (Score*vel).sum(axis=1)

            RHSmean = ((torch.exp(w)/torch.exp(w).mean())*RHS.flatten()).mean()

            dw2  = (LHS - (RHS.flatten()))

            dw =  (LHS - (RHS.flatten()-RHSmean))
        return dw, RHSmean, vel,dw2


    def comput_weight_batch(self, flow, x , t):
        Score, RHS = self.comput_SandRHS(x,t)

        with torch.no_grad():
 #Modefied for Alanine problem           #Score, RHS = self.comput_SandRHS(x,t)
            div = flow.divergence(x)
            vel = flow(x)
        
            LHS = div + (Score*vel).sum(axis=1)

        return LHS, RHS, vel

    def sample_weight(self, nsample, modellist, tlist):
        ## This function resample using trained network
        assert len(modellist) == len(tlist)
        x = self.prior.sample(nsample)

        w = torch.zeros(nsample).to(x.device)
        logz = 0
        t = torch.tensor(0.0).to(x.device)
        
        with torch.no_grad():
            for i in range(len(modellist)):
                
                dw, RHSmean, vel = self.comput_weight(modellist[i], x,w, t )
                dlogz = -RHSmean

                w = w + dw*(tlist[i]-t)
                x = x + vel*(tlist[i]-t)
                
                logz = logz + dlogz *(tlist[i]-t)
                    
                t = tlist[i]
        return x, w, logz

    def sample_weight_method2(self, nsample, modellist, tlist):
        ## This function resample using trained network
        assert len(modellist) == len(tlist)
        x = self.prior.sample(nsample)

        w = torch.zeros(nsample).to(x.device)
        w2 = torch.zeros(nsample).to(x.device)

        logz = 0
        t = torch.tensor(0.0).to(x.device)
        
        with torch.no_grad():
            for ii in range(1):
                for i in range(len(modellist)):
                
                    dw, RHSmean, vel, dw2 = self.comput_weight_method2(modellist[i], x,w,t )
                    dlogz = -RHSmean

                    w = w + dw*(tlist[i]-t)
                    w2 = w2 + dw2*(tlist[i]-t)

                    x = x + vel*(tlist[i]-t)
                
                    logz = logz + dlogz *(tlist[i]-t)
                
                    t = tlist[i]
            w2_2 = w2 - w2.mean()
            logz = torch.log(torch.mean(torch.exp(w2_2))) + w2.mean()
        return x, w, logz


    def sample_woweight(self, nsample, modellist, tlist):
        ## This function resample using trained network
        assert len(modellist) == len(tlist)
        x = self.prior.sample(nsample)
        w = torch.zeros(nsample).to(x.device)
        
        logz = 0
        t = torch.tensor(0.0).to(x.device)
        
        with torch.no_grad():
            for i in range(len(modellist)):
                
                dw, RHSmean, vel = self.comput_weight(modellist[i], x,w, t )
                dlogz = -RHSmean

                x = x + vel*(tlist[i]-t)
                
                logz = logz + dlogz *(tlist[i]-t)
                    
                t = tlist[i]
        return x, w, logz


    def sample_weight_batch(self, nsample,nbatch, modellist, tlist):
        ## This function resample using trained network
        assert len(modellist) == len(tlist)
        x = self.prior.sample(nsample)

        w = torch.zeros(nsample).to(x.device)
        
        batch = np.arange(nsample)
        batch = batch.reshape([-1,nbatch])
        batch_number = batch.shape[0]

        LHS = torch.zeros(nsample, len(modellist)).to(x.device)
        RHS = torch.zeros(nsample, len(modellist)).to(x.device)
    
        logz = 0
        
        with torch.no_grad():
            for n in range(batch_number):
                t = torch.tensor(0.0).to(x.device)
                xbatch = x[batch[n]]
                
                for i in range(len(modellist)):

                    LHSbatch, RHSbatch, vel = self.comput_weight_batch(modellist[i], xbatch, t )

                    LHS[batch[n], i] = LHSbatch
                    RHS[batch[n], i] = RHSbatch.flatten()

                    xbatch = xbatch + vel*(tlist[i]-t)
                    
                    t = tlist[i]
                x[batch[n]] = xbatch
            t = 0
            for i in range(len(modellist)):
                RHSmean = ((torch.exp(w)/torch.exp(w).mean())*RHS[:,i].flatten()).mean()
                dw = (LHS[:,i]-(RHS[:,i]-RHSmean))

                dlogz = -RHSmean
                w = w + dw*(tlist[i]-t)

                logz = logz + dlogz *(tlist[i]-t)
                t = tlist[i]
        return x, w, logz

    def sample_weight_batch_method2(self, nsample,nbatch, modellist, tlist):
        ## This function resample using trained network
        assert len(modellist) == len(tlist)
        x = self.prior.sample(nsample)

        w = torch.zeros(nsample).to(x.device)
        w2 = torch.zeros(nsample).to(x.device)
        
        batch = np.arange(nsample)
        batch = batch.reshape([-1,nbatch])
        batch_number = batch.shape[0]

        LHS = torch.zeros(nsample, len(modellist)).to(x.device)
        RHS = torch.zeros(nsample, len(modellist)).to(x.device)
    
        logz = 0
        
        with torch.no_grad():
            for n in range(batch_number):
                t = torch.tensor(0.0).to(x.device)
                xbatch = x[batch[n]]
                
                for i in range(len(modellist)):

                    LHSbatch, RHSbatch, vel = self.comput_weight_batch(modellist[i], xbatch, t )

                    LHS[batch[n], i] = LHSbatch
                    RHS[batch[n], i] = RHSbatch.flatten()

                    xbatch = xbatch + vel*(tlist[i]-t)
                    
                    t = tlist[i]
                x[batch[n]] = xbatch
            t = 0
            for i in range(len(modellist)):
                RHSmean = ((torch.exp(w)/torch.exp(w).mean())*RHS[:,i].flatten()).mean()
                dw = (LHS[:,i]-(RHS[:,i]-RHSmean))
                dw2 = (LHS[:,i]-(RHS[:,i]))

                dlogz = -RHSmean
                w = w + dw*(tlist[i]-t)
                w2 = w2 + dw2*(tlist[i]-t)

                logz = logz + dlogz *(tlist[i]-t)
                t = tlist[i]
            w2_2 =  w2 - w2.mean()
            print(w2.mean())
            logz = torch.log(torch.mean(torch.exp(w2_2))) + w2.mean()            
        return x, w, logz

