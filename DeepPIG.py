import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from scipy.stats import ttest_rel
import math
from copy import deepcopy

# EarlyStopping imported from pytorchtools
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, by='loss'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.by = by
    def __call__(self, val_loss, model):
        if self.by == 'loss':
            score = val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
        
            if score > self.best_score + self.delta:
                self.counter += 1
                # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        if self.by == 'score':
            score = val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
        
            if score < self.best_score + self.delta:
                self.counter += 1
                # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss




# Mean Squared Logarithmic Error, implemeted in Keras version.
class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()        
    def forward(self, pred, actual):
        pred = torch.maximum(pred,torch.ones_like(pred,requires_grad=False)*1e-7)
        actual = torch.maximum(actual,torch.ones_like(pred,requires_grad=False)*1e-7)        
        return torch.mean(torch.square(torch.log(actual + 1.) - torch.log(pred + 1.)))



class PairwiseGateLayer(nn.Module):
    def __init__(self, input_dim):
        super(PairwiseGateLayer, self).__init__()

        self.feat_dim = input_dim // 2


        self.filter_weight = nn.Parameter(torch.rand(input_dim), requires_grad=True)
        nn.init.uniform_(self.filter_weight.data, -0.05,0.05)
        self.device = self.filter_weight.device
        self.mu = nn.Parameter((1+1e-6)*torch.ones(input_dim, ), requires_grad=False)
        self.noise = torch.randn(self.mu.size())
        self.stochastic_gate = nn.Parameter(0.5*torch.ones(input_dim, ))
        self.epoch = 0

    def hard_sigmoid(self, x):
        return torch.clamp(x, 0.0, 1.0)
    def regularizer(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 
    
    def forward(self, X):
        if self.training:
            self.noise = self.noise.to(self.filter_weight.device)
            z = self.mu + self.noise.normal_()
        if not self.training:
            z = self.mu
        stochastic_gate = self.hard_sigmoid(z)
        
        
        elm_mul = X * stochastic_gate * self.filter_weight
        output = elm_mul[:, 0:self.feat_dim] + elm_mul[:, self.feat_dim:]
        return output


class DeepPIG(nn.Module):
    def __init__(self, n_input, print_interval=1000):
        super(DeepPIG, self).__init__()
        self.n_input = n_input
        self.pairwise_coupling = PairwiseGateLayer(n_input)

        self.FC1 = nn.Linear(self.pairwise_coupling.feat_dim,
                             self.pairwise_coupling.feat_dim)

        self.FC2 = nn.Linear(self.pairwise_coupling.feat_dim, 1)

        nn.init.constant_(self.FC1.bias.data, 0)
        nn.init.constant_(self.FC2.bias.data, 0)

        self.ELU = nn.ELU()

        self.device = self.FC1.weight.device
        self.print_interval = print_interval
        

    def forward(self, X):
        output = self.pairwise_coupling.forward(X)
        output = self.FC1(output)
        output = self.ELU(output)
        output = self.FC2(output)
        return output

    def fit(self, X, y, y_design, es,batch_size=32,max_epoch=300,transfer_min_epoch=0,K=10,lr=0.0001, l1=0.001,lambda_stg=0.01,patience=30):
        self.device = self.FC1.weight.device
        mu_trace = {}

        p = self.pairwise_coupling.feat_dim
        X_origin, X_ko = X[:,:p],X[:,p:]

        X_origin_only = np.concatenate((X_origin, np.zeros_like(X_ko)), axis=1)
        X_origin_only = torch.Tensor(X_origin_only).to(self.device)


        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)
        train_idx, val_idx, y_train, y_val = train_test_split(np.arange(X.shape[0]),y,test_size=0.1)
        
        ds_train_origin_only = TensorDataset(X_origin_only[train_idx], y_train)
        dl_train_origin_only = DataLoader(ds_train_origin_only, batch_size=batch_size)

        ds_train = TensorDataset(X[train_idx], y_train)
        dl_train = DataLoader(ds_train, batch_size=batch_size)

        X_val = X[val_idx]
        if y_design == "nonlinear":
            criterion = MSLELoss()
        if y_design == "linear":
            criterion = nn.MSELoss()
        if y_design == "clf":
            criterion = nn.BCEWithLogitsLoss()

        optimizer_adam = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.)
        early_stopping = EarlyStopping(patience=patience, path=es, verbose=False)

        
        transfer = False
        loss_min = np.inf
        counter = 0
        epoch = 0
        model_log = {}
        optim_log = {}
        model_log[0] = deepcopy(self.state_dict())
        optim_log[0] = deepcopy(optimizer_adam.state_dict())

        while True:
            if epoch == max_epoch:
                print("MAX EPOCH REACHED!")
                break
            
            epoch += 1
            
            self.train()
            #Training Phase 1
            if not transfer:
                for batch_idx, samples in enumerate(dl_train_origin_only):
                    x_train_batch, y_train_batch = samples

                    loss = 0
                    for k in range(K):
                        output = self.forward(x_train_batch)
                        loss += criterion(output, y_train_batch)
                        

                    loss = loss / K
                    reg_loss = 0.0
                    for param in [self.pairwise_coupling.filter_weight[:p],self.FC1.weight.data]:
                        reg_loss += torch.norm(param, p=1)
                    
                    mu = self.pairwise_coupling.mu.data[:p]
                    mu_tilde = self.pairwise_coupling.mu.data[p:]
                    z = self.pairwise_coupling.filter_weight.data[:p]
                    z_tilde = self.pairwise_coupling.filter_weight.data[p:]
                    w = (self.FC2.weight.data @ self.FC1.weight.data).reshape(p,)

                    loss += l1 * reg_loss

                    optimizer_adam.zero_grad()
                    loss.backward()
                    optimizer_adam.step()

                    model_log[epoch] = deepcopy(self.state_dict())
                    optim_log[epoch] = deepcopy(optimizer_adam.state_dict())

            # Training Phase 2
            if transfer:
                for batch_idx, samples in enumerate(dl_train):
                    x_train_batch, y_train_batch = samples

                    loss = 0
                    for k in range(K):
                        output = self.forward(x_train_batch)
                        loss += criterion(output, y_train_batch)                       

                    loss = loss / K
                    reg_loss = 0.0
                    for param in [self.pairwise_coupling.mu.data[:p], self.pairwise_coupling.filter_weight[:p], self.FC1.weight.data]:
                        reg_loss += torch.norm(param, p=1)  


                    loss += l1 * reg_loss 
                    stg_reg_loss = torch.mean(self.pairwise_coupling.regularizer((self.pairwise_coupling.mu.data[:p] + 0.5)/0.5))
                    
                    loss += lambda_stg*stg_reg_loss
                    
                    optimizer_adam.zero_grad()
                    loss.backward()
                    optimizer_adam.step()

            self.eval()

            

            mu = self.pairwise_coupling.mu.data.detach().cpu().numpy()[:p]
            mu_tilde = self.pairwise_coupling.mu.data.detach().cpu().numpy()[p:]
            z = self.pairwise_coupling.filter_weight.data.detach().cpu().numpy()[:p]
            z_tilde = self.pairwise_coupling.filter_weight.data.detach().cpu().numpy()[p:]
            w = (self.FC2.weight.data @ self.FC1.weight.data).reshape(p,).detach().cpu().numpy()

            output = self.forward(X[train_idx])
            train_loss = criterion(output, y_train) 
            y_pred = self.forward(X_val)
            val_loss = criterion(y_pred, y_val) 

            if not transfer:
                if val_loss < loss_min:
                    loss_min = val_loss
                    optimal_epoch = epoch
                    counter = 0
                else:
                    counter += 1

            if transfer:
                early_stopping(val_loss.item(),self)


            # Weight Transfer 
            if not transfer and (counter == 10 and epoch > transfer_min_epoch):
                transfer = True
                print("Transfer at epoch", int(optimal_epoch *0.5))
                epoch = int(optimal_epoch *0.5)
                with torch.no_grad():
                    self.load_state_dict(model_log[int(optimal_epoch *0.5) ])
                    optimizer_adam.load_state_dict(optim_log[int(optimal_epoch *0.5)])
                    z_abs = np.abs(self.pairwise_coupling.filter_weight.detach().cpu().numpy()[:p])
                    self.pairwise_coupling.filter_weight.data[p:] = \
                                                        self.pairwise_coupling.filter_weight.data[:p]
                    z_prob = (z_abs - np.min(z_abs)) / (np.max(z_abs) - np.min(z_abs))
                    self.pairwise_coupling.mu.data[:p] = torch.tensor(z_prob, device=self.device)
                    self.pairwise_coupling.mu.data[p:] = torch.tensor(z_prob, device=self.device)
                    self.pairwise_coupling.mu.requires_grad = True

                    early_stopping.early_stop = False
                    early_stopping.counter = 0
            
            W = mu * (z*w)**2
            W_tilde = mu_tilde * (z_tilde*w)**2
            score = W-W_tilde

            mask = score - np.mean(score) < 2*np.std(score)
            ttest_p = ttest_rel(W[mask],W_tilde[mask], alternative='greater')[1]

            
            if transfer and early_stopping.early_stop and ttest_p > 0.05:
                print("EarlyStopping!", epoch, "ttest_p",ttest_p)
                break


            if (epoch+1) % self.print_interval == 0:
                print("Epoch : ", epoch+1, "Train Loss : ",train_loss.item(),"Val Loss : ", val_loss.item())


    def knockoff_stats(self):
        w = self.FC2.weight @ self.FC1.weight
        w = w.reshape(self.pairwise_coupling.feat_dim, )
        z = self.pairwise_coupling.filter_weight[:self.pairwise_coupling.feat_dim]
        z_tilde = self.pairwise_coupling.filter_weight[self.pairwise_coupling.feat_dim:]
        mu = self.pairwise_coupling.mu.data[:self.pairwise_coupling.feat_dim]
        mu_tilde = self.pairwise_coupling.mu.data[self.pairwise_coupling.feat_dim:]

        W = mu * (w * z) ** 2 - mu_tilde * (w * z_tilde) ** 2
        return W.detach().cpu().numpy()


    def knockoff_select(self, q=0.2, ko_plus=True):
        W = self.knockoff_stats()
        p = len(W)
        t = np.sort(np.concatenate(([0], abs(W))))
        if ko_plus:
            ratio = [(1 + sum(W <= -tt)) / max(1, sum(W >= tt)) for tt in t[:p]]
        else:
            ratio = [sum(W <= -tt) / max(1, sum(W >= tt)) for tt in t[:p]]
        ind = np.where(np.array(ratio) <= q)[0]
        if len(ind) == 0:
            T = float('inf')
        else:
            T = t[ind[0]]

        return np.where(W >= T)[0]
    
    

    def fdp(self, S, beta_true):
        return sum(beta_true[S] == 0) / max(1, len(S))

    def pow(self, S, beta_true):
        return sum(beta_true[S] != 0) / sum(beta_true != 0)