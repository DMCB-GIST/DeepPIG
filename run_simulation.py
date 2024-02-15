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
from DeepPIG import DeepPIG


try:
    os.makedirs("./simulation_data/checkpoint")
    
except FileExistsError:
    pass

parser = argparse.ArgumentParser()

parser.add_argument('--x_design', '-xd', type=str, default='linear')
parser.add_argument('--y_design', '-yd', type=str, default='linear')
parser.add_argument("--amplitude","-a", action="store", type=float,
                  dest="amplitude", help="signal amplitude", default=1)
parser.add_argument("--gpu","-g", action="store", type=int,
                  dest="gpu", help="cuda", default=0)
parser.add_argument("--lr","-lr", action="store", type=float,
                  dest="lr", help="learning_rate", default=0.01)
parser.add_argument("--l1","-l1", action="store", type=float,
                  dest="l1", help="l1 coeff.", default=0.0001)
config = parser.parse_args()



x_design = config.x_design
y_design = config.y_design
amplitude = config.amplitude
gpu = config.gpu
lr = config.lr
l1 = config.l1

batch_size = 32
q = 0.2 # target FDR level
it = 30 # number of iterations
patience = 30
mlp_epoch = 300 # number of mlp training epochs
mlp_verb = 0 # verbose level of mlp

cuda = True if torch.cuda.is_available() else False
if cuda:
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')


input_data = './simulation_data/'+x_design

output_name = "./"+ x_design + "_" + y_design + "_" + str(amplitude)


result = np.repeat(np.repeat([[0]], 6, 0), it + 2, axis=1)
r_est = np.zeros(it, dtype=int)
colnames = ['mean', 'sd'] + [str(obj) for obj in range(1, it + 1)]
result = pd.DataFrame(result, index=['FDR', 'Power', 'FDR+', 'Power+',"selected","selected+"], columns=colnames)

train_errors, val_errors = [], []

for i in range(it):
    iter_id = i
    print("Run",iter_id)
    Xnew, true_beta, epsilon = torch.load(input_data+"_"+str(iter_id))

    n = Xnew.shape[0]
    p = int(Xnew.shape[1]/2)
    X = Xnew[:,:p]
    if y_design == "linear":
        y = X @ (amplitude*true_beta) + epsilon
    if y_design == "nonlinear":
        y = np.sin(X @ (amplitude*true_beta)) * np.exp(X @ (amplitude*true_beta)) + epsilon

    # compute knockoff statistics
    dp = DeepPIG(Xnew.shape[1], print_interval=50).to(device)
    dp.fit(Xnew,y, max_epoch=mlp_epoch,lr=lr,l1=l1, y_design = y_design, patience=patience,es=output_name+"_"+str(i)+'.pt')


    # feature selection

    dp.eval()
    selected = dp.knockoff_select(ko_plus=False)
    selected_plus = dp.knockoff_select(ko_plus=True)
    print("# KO : ", len(selected), "# KO+ : ", len(selected_plus) )
    print("FDR+ :", dp.fdp(selected_plus, true_beta), "Pow+ :", dp.pow(selected_plus, true_beta))
    

    result.iloc[0, i + 2] = dp.fdp(selected, true_beta)
    result.iloc[1, i + 2] = dp.pow(selected, true_beta)
    result.iloc[2, i + 2] = dp.fdp(selected_plus, true_beta)
    result.iloc[3, i + 2] = dp.pow(selected_plus, true_beta)
    result.iloc[4, i + 2] = len(selected)
    result.iloc[5, i + 2] = len(selected_plus)

    
    # back-up
    result[result.columns[0]] = np.mean(result.iloc[:, 2:3+i], axis=1)
    result[result.columns[1]] = np.std(result.iloc[:, 2:3+i], axis=1, ddof=1)
    result.to_csv('./'+output_name, index=True, header=True, sep=',')

