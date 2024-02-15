from DeepPIG import DeepPIG
import numpy as np
import pandas as pd
import argparse
import torch
import os
import time
from torch_geometric.data import Data
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score
import keras
from keras import backend as K 
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Layer

from numba import cuda
import warnings
warnings.filterwarnings("ignore")


# DeepPINK layer
class PairwiseConnected(Layer):
    def __init__(self, **kwargs):
        super(PairwiseConnected, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0
        self.feat_dim = input_shape[-1] // 2
        self.w = self.add_weight(name='weight', shape=(input_shape[-1],),
                                 initializer='uniform', trainable=True)
        super(PairwiseConnected, self).build(input_shape)

    def call(self, x):
        elm_mul = x * self.w
        output = elm_mul[:, 0:self.feat_dim] + elm_mul[:, self.feat_dim:]

        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.feat_dim
        return tuple(output_shape)


parser = argparse.ArgumentParser()
parser.add_argument("--dimension","-d", action="store", type=int, dest="d",)
parser.add_argument("--output","-o", action="store", type=str, dest="o",)
parser.add_argument("--epoch","-e", action="store", type=int,
                  dest="epoch", help="epoch", default=500)


config = parser.parse_args()

X_path = "/NAS_Storage3/euiyoung/Graph_knockoff/RealDataAnalysis/TCGA/"+config.o+"/expr_prognostic_DC_final.tsv"
y_path = "/NAS_Storage3/euiyoung/Graph_knockoff/RealDataAnalysis/TCGA/"+config.o+"/y_prognostic_DC_final.tsv"

print(X_path, y_path)
output_path = config.o

cuda_availabe = True if torch.cuda.is_available() else False
if cuda_availabe:
    device = torch.device(f'cuda:{1}' if torch.cuda.is_available() else 'cpu')


bs = 32
d = config.d
aut_epoch = 500
aut_loss = 'mean_squared_error'
aut_verb = 0
dnn_epoch = config.epoch
dnn_loss = 'binary_crossentropy'
dnn_verb = 0
aut_met = 'relu'
dnn_met = 'elu'
q = 0.2
nrep = 100
l1 = 1e-5 # l1 regularization factor in mlp
lr = 0.005 # learning rate for mlp training

X = np.array(pd.read_table(X_path,nrows=d, header=None)).transpose()


sc = StandardScaler()
X = sc.fit_transform(X)
n = X.shape[0]
p = X.shape[1]
print(X.shape)
y = np.array(pd.read_table(y_path)["LTS"].tolist())
y = np.expand_dims(y,axis=1)

sel_count_pink, sel_count_pig = [], []
result_pink, result_pig = pd.DataFrame(np.zeros(p), index=np.arange(p), columns=['Frequency']), pd.DataFrame(np.zeros(p), index=np.arange(p), columns=['Frequency'])


mat_selected = np.zeros([nrep, d])
mat_selected_plus = np.zeros([nrep, d])

n_train, n_test = int(X.shape[0]*0.8), X.shape[0] - int(X.shape[0]*0.8)

indmat = np.zeros([n_train, nrep])

yhat_train_pink = np.zeros([nrep, n_train])
yhat_test_pink = np.zeros([nrep, n_test])
pe_test_pink = [np.nan]*nrep
pe_train_pink = [np.nan]*nrep
auc_train_pink = [np.nan]*nrep
auc_test_pink = [np.nan]*nrep
mat_selected_pink = np.zeros([nrep, d])

yhat_train_pig = np.zeros([nrep, n_train])
yhat_test_pig = np.zeros([nrep, n_test])
pe_test_pig = [np.nan]*nrep
pe_train_pig = [np.nan]*nrep
auc_train_pig = [np.nan]*nrep
auc_test_pig = [np.nan]*nrep
mat_selected_pig = np.zeros([nrep, d])

yhat_test_all = np.zeros([nrep, n_test])
pe_test_all = [np.nan]*nrep
auc_test_all = [np.nan]*nrep

yhat_test_random = np.zeros([nrep, n_test])
pe_test_random = [np.nan]*nrep
auc_test_random = [np.nan]*nrep

dp_name = str(int(d))+"_pink"
pig_name = str(int(d))+"_PIG"
all_name = str(int(d))+"_all"
random_name = str(int(d))+"_random"



for i in range(nrep):
    print("Run",i)
    r_hat = 3
    autoencoder = Sequential()
    autoencoder.add(Dense(d, activation=aut_met, use_bias=True, input_shape=(d,)))
    autoencoder.add(Dense(r_hat, activation=aut_met, use_bias=True))
    autoencoder.add(Dense(d, activation=aut_met, use_bias=True))
    autoencoder.compile(loss=aut_loss, optimizer=keras.optimizers.Adam())
    autoencoder.fit(X, X, epochs=aut_epoch, batch_size=bs, verbose=aut_verb)
    C = autoencoder.predict(X)
    E = X - C
    sigma = np.sqrt(np.sum(E ** 2) / (n * d))
    X_ko = C + sigma * np.random.randn(n, d)
    Xnew = np.hstack((X, X_ko))

    print(Xnew.shape)
    train_idx, test_idx, _,_ = train_test_split(np.arange(Xnew.shape[0]),y, test_size=0.2,stratify=y)

    Xnew_train, y_train = Xnew[train_idx], y[train_idx]
    Xnew_test, y_test = Xnew[test_idx], y[test_idx]
    X,Xk = Xnew[:,:p], Xnew[:,p:]

    es = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
    dp = Sequential()
    dp.add(PairwiseConnected(input_shape=(2 * d,)))
    dp.add(Dense(d, activation=dnn_met,
                    kernel_regularizer=keras.regularizers.l1_l2(l1=1e-6, l2=1e-6)))
    dp.add(Dropout(0.4)) # play with this number, such as 0.4, 0.6, 0.7
    dp.add(Dense(1, activation='elu',
                kernel_regularizer=keras.regularizers.l1_l2(l1=1e-6, l2=1e-6)))
    dp.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer=keras.optimizers.Adam(learning_rate=0.001))
    dp.fit(Xnew_train, y_train, epochs=dnn_epoch, batch_size=bs, verbose=dnn_verb, validation_split=0.2, callbacks=[es])
    
    weights = dp.get_weights()
    w3 = np.matmul(weights[1], weights[2]).reshape(d,)
    w1 = np.multiply(weights[0][:d], w3)
    w2 = np.multiply(weights[0][d:], w3)
    W = w1**2 - w2**2

    t = np.sort(np.concatenate(([0], abs(W))))
    
    ratio = [float(sum(W <= -tt)) / float(max(1, sum(W >= tt))) for tt in t[:d]]
    ind = np.where(np.array(ratio) <= q)[0]
    if len(ind) == 0:
        T = float('inf')
    else:
        T = t[ind[0]]
    
    selected_pink = np.where(W >= T)[0]

    train_LTS, train_nonLTS = np.where(y_train==1)[0], np.where(y_train==0)[0]


    print("DP selected :",selected_pink)
    result_pink.iloc[selected_pink] += 1
    mat_selected_pink[i,selected_pink] = 1
    sel_count_pink.append(len(selected_pink))

    

    yhat_train_pink[i,:] = [1 if a > 0.5 else 0 for a in dp.predict(Xnew_train).flatten()]
    pe_train_pink[i] = np.mean([1 if a > 0.5 else 0 for a in dp.predict(Xnew_train).flatten()] != y_train.flatten())

    #### DeepPIG ####
    pig = DeepPIG(Xnew.shape[1], print_interval=100).to(device)
    pig.fit(Xnew_train, y_train, max_epoch=dnn_epoch,transfer_min_epoch=30,lr=lr,l1=l1,patience=30,y_design='clf',es='./real_data/survival/'+output_path+'/checkpoint/'+pig_name+"_"+str(i)+'.pt')

    
    selected_pig = pig.knockoff_select(ko_plus=False,q=q)

    print("PIG selected :",selected_pig)

    result_pig.iloc[selected_pig] += 1
    mat_selected_pig[i,selected_pig] = 1
    sel_count_pig.append(len(selected_pig))

    yhat_train_pig[i,:] = [1 if a > 0.5 else 0 for a in pig.forward(torch.Tensor(Xnew_train).to(device)).flatten()]
    pe_train_pig[i] = np.mean(yhat_train_pig[i,:] != y_train.flatten())


    
    ## refit to calculate PE (DP) ##
    if len(selected_pink) == 0:
        pe_test_pink[i] = np.nan

    else:
        print("DP_refit")
        s = len(selected_pink)
        es = EarlyStopping(monitor='val_loss', patience=30, verbose=2,restore_best_weights=False)
        mrefit = Sequential()
        mrefit.add(Dense(s, input_dim=s, activation='relu'))
        mrefit.add(Dense(s, input_dim=s, activation='relu'))
        mrefit.add(Dense(1, activation='sigmoid'))
        mrefit.compile(loss=dnn_loss, optimizer=keras.optimizers.Adam())
        mrefit.fit(X[train_idx,:][:,selected_pink], y_train, epochs=600, batch_size=bs, verbose=dnn_verb, validation_split=0.2, callbacks=[es])
        
        

        
        yhat_test_pink[i,:] = [1 if a > 0.5 else 0 for a in mrefit.predict(X[test_idx,:][:,selected_pink]).flatten()]
        pe_test_pink[i] = np.mean(yhat_test_pink[i,:] != y_test.flatten())
        auc_test_pink[i] = roc_auc_score(y[test_idx], mrefit.predict(X[test_idx,:][:,selected_pink]).flatten())
    ###########################
    ## refit to calculate PE (PIG)##
    if len(selected_pig) == 0:

        pe_test_pig[i] = np.nan
    else:
        print("PIG_refit")
        s = len(selected_pig)
        es = EarlyStopping(monitor='val_loss', patience=30, verbose=2,restore_best_weights=False)
        mrefit_pig = Sequential()
        mrefit_pig.add(Dense(s, input_dim=s, activation='relu'))
        mrefit_pig.add(Dense(s, input_dim=s, activation='relu'))
        mrefit_pig.add(Dense(1, activation='sigmoid'))
        mrefit_pig.compile(loss=dnn_loss, optimizer=keras.optimizers.Adam(learning_rate=0.003))
        mrefit_pig.fit(X[train_idx,:][:,selected_pig], y_train, epochs=600, batch_size=bs, verbose=dnn_verb, validation_split=0.2, callbacks=[es])

        yhat_test_pig[i,:] = [1 if a > 0.5 else 0 for a in mrefit_pig.predict(X[test_idx,:][:,selected_pig]).flatten()]
        pe_test_pig[i] = np.mean(yhat_test_pig[i,:] != y_test.flatten())

        auc_test_pig[i] = roc_auc_score(y[test_idx], mrefit_pig.predict(X[test_idx,:][:,selected_pig]).flatten())
    ###########################
    ## refit to calculate PE (all)##
    print("All_refit")
    mrefit_all = Sequential()
    es = EarlyStopping(monitor='val_loss', patience=30, verbose=2,restore_best_weights=False)
    mrefit_all.add(Dense(d, input_dim=d, activation='relu'))
    mrefit_all.add(Dense(d, input_dim=d, activation='relu'))
    mrefit_all.add(Dense(1, activation='sigmoid'))
    mrefit_all.compile(loss=dnn_loss, optimizer=keras.optimizers.Adam(learning_rate=0.005))
    mrefit_all.fit(X[train_idx,:], y_train, epochs=600, batch_size=bs, verbose=dnn_verb, validation_split=0.2, callbacks=[es])
    

    yhat_test_all[i,:] = [1 if a > 0.5 else 0 for a in mrefit_all.predict(X[test_idx,:]).flatten()]
    pe_test_all[i] = np.mean(yhat_test_all[i,:] != y_test.flatten())
    auc_test_all[i] = roc_auc_score(y[test_idx], mrefit_all.predict(X[test_idx,:]).flatten())
    ###########################

    ###########################   
        ## refit to calculate PE (random)##
    if len(selected_pig) == 0:

        pe_test_random[i] = np.nan

    else:
        print("random_refit")
        random_feature = np.random.choice(d,len(selected_pig), replace=False)
        mrefit_random = Sequential()
        es = EarlyStopping(monitor='val_loss', patience=30, verbose=2,restore_best_weights=False)
        mrefit_random.add(Dense(s, input_dim=s, activation='relu'))
        mrefit_random.add(Dense(s, input_dim=s, activation='relu'))
        mrefit_random.add(Dense(1, activation='sigmoid'))
        mrefit_random.compile(loss=dnn_loss, optimizer=keras.optimizers.Adam())
        mrefit_random.fit(X[train_idx,:][:,random_feature], y_train, epochs=600, batch_size=bs, verbose=dnn_verb, validation_split=0.2, callbacks=[es])
        
        yhat_test_random[i,:] = [1 if a > 0.5 else 0 for a in mrefit_random.predict(X[test_idx,:][:,random_feature]).flatten()]
        # print("Random",yhat_test_random[i,:])
        pe_test_random[i] = np.mean(yhat_test_random[i,:] != y_test.flatten())
        auc_test_random[i] = roc_auc_score(y[test_idx], mrefit_random.predict(X[test_idx,:][:,random_feature]).flatten())
    ###########################
    print("AVG.so_far (DP):",
            "TestError:",np.round(np.nanmean(pe_test_pink[:i+1]),4),
            "AUC:",np.round(np.nanmean(auc_test_pink[:i+1]),4),
            "#.sel:",np.round(np.nanmean(sel_count_pink),4),
            "Unselected",len(np.where(np.array(sel_count_pink)==0)[0]),"/",i+1)
    print("AVG.so_far (PIG):",
            "TestError:",np.round(np.nanmean(pe_test_pig[:i+1]),4),
            "AUC:",np.round(np.nanmean(auc_test_pig[:i+1]),4),
            "#.sel:",np.round(np.nanmean(sel_count_pig),4),
            "Unselected",len(np.where(np.array(sel_count_pig)==0)[0]),"/",i+1)
    
    print("AVG.so_far (all)",np.nanmean(pe_test_all[:i+1]),
            "AUC:",np.nanmean(auc_test_all[:i+1]))
    print("AVG.so_far (random)",np.nanmean(pe_test_random[:i+1]),
            "AUC:",np.nanmean(auc_test_random[:i+1]))



os.chdir('./real_data/survival/'+output_path)
pd.DataFrame(mat_selected_pink).to_csv(dp_name+'_selected.csv', index=True, header=True, sep=',')
pd.DataFrame(pe_train_pink).to_csv('real_pe_train_' + dp_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(pe_test_pink).to_csv('real_pe_test_' + dp_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(auc_train_pink).to_csv('real_auc_train_' + dp_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(auc_test_pink).to_csv('real_auc_test_' + dp_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_test_pink).to_csv('real_yhat_test_' + dp_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_train_pink).to_csv('real_yhat_train_' + dp_name+".csv", index=True, header=True, sep=',')

print("#### DeepPINK ####")
print('Training average PE: ',np.mean(pe_train_pink),'Test average PE: ' ,np.nanmean(pe_test_pink),'Test average AUC: ',np.nanmean(auc_test_pink[:i+1]))
print("Max.",np.max(result_pink['Frequency']), "Avg.",np.mean(sel_count_pink), "Redundant iter:", nrep - len(np.nonzero(sel_count_pink)[0]))

pd.DataFrame(mat_selected_pig).to_csv(pig_name+'_selected.csv', index=True, header=True, sep=',')
pd.DataFrame(pe_train_pig).to_csv('real_pe_train_' + pig_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(pe_test_pig).to_csv('real_pe_test_' + pig_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(auc_train_pig).to_csv('real_auc_train_' + pig_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(auc_test_pig).to_csv('real_auc_test_' + pig_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_test_pig).to_csv('real_yhat_test_' + pig_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_train_pig).to_csv('real_yhat_train_' + pig_name+".csv", index=True, header=True, sep=',')

print("#### PIGnet ####")
print('Training average PE: ',np.mean(pe_train_pig),'Test average PE: ' ,np.nanmean(pe_test_pig),'Test average AUC: ',np.nanmean(auc_test_pig[:i+1]))
print("Max.",np.max(result_pig['Frequency']), "Avg.",np.mean(sel_count_pig), "Redundant iter:", nrep - len(np.nonzero(sel_count_pig)[0]))

pd.DataFrame(pe_test_all).to_csv('real_pe_test_' + all_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(auc_test_all).to_csv('real_auc_test_' + all_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_test_all).to_csv('real_yhat_test_' + all_name+".csv", index=True, header=True, sep=',')

print("#### All ####")
print('Test average PE: ' ,np.nanmean(pe_test_all),'Test average AUC: ',np.nanmean(auc_test_all))


pd.DataFrame(pe_test_random).to_csv('real_pe_test_' + random_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(auc_test_random).to_csv('real_auc_test_' + random_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_test_random).to_csv('real_yhat_test_' + random_name+".csv", index=True, header=True, sep=',')


print("#### Random ####") 
print('Test average PE: ' ,np.nanmean(pe_test_random),'Test average AUC: ',np.nanmean(auc_test_random))
