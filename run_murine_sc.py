import torch
from DeepPIG import DeepPIG
import pandas as pd
import os
import random
import tensorflow as tf
import numpy as np
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.initializers import RandomUniform 
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Layer

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



cuda = True if torch.cuda.is_available() else False
if cuda:
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

os.chdir('./real_data/murine_sc')
ds = 'rna1'
d = 200

X0 = np.genfromtxt(ds + '.csv', delimiter=',', skip_header=1)
y = X0[:, 13777]
X = X0[:,0: 13777]
y = np.expand_dims(y,axis=1)


indmat_dist = np.genfromtxt('indmat_dist_p50.csv', delimiter=',', skip_header=1)
screen_top = np.genfromtxt('top500_p50.csv', delimiter=',', skip_header=1) - 1


# center_scale data
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0, ddof=1)
n = X.shape[0]
p = X.shape[1]


bs = 15

aut_epoch = 300
aut_loss = 'mean_squared_error'
aut_verb = 0
dnn_epoch = 200
dnn_loss = 'binary_crossentropy'
dnn_verb = 0
aut_met = 'relu'
dnn_met = 'elu'
q = 0.2
nrep = 100
l1 = 1e-5 # l1 regularization factor in mlp
lr = 0.005 # learning rate for mlp training

mat_selected = np.zeros([nrep, d])
mat_selected_plus = np.zeros([nrep, d])

n_train = 228
n_test = 57
indmat = np.zeros([n_train, nrep])

yhat_train_pink = np.zeros([nrep, n_train])
yhat_test_pink = np.zeros([nrep, n_test])
pe_test_pink = [0.]*nrep
pe_train_pink = [0.]*nrep
mat_selected_pink = np.zeros([nrep, d])

yhat_train_pig = np.zeros([nrep, n_train])
yhat_test_pig = np.zeros([nrep, n_test])
pe_test_pig = [0.]*nrep
pe_train_pig = [0.]*nrep
mat_selected_pig = np.zeros([nrep, d])

yhat_test_all = np.zeros([nrep, n_test])
pe_test_all = [0.]*nrep


yhat_test_random = np.zeros([nrep, n_test])
pe_test_random = [0.]*nrep



pink_name = str(int(d))+"_PINK"
pig_name = str(int(d))+"_PIG"
all_name = str(int(d))+"_all"
random_name = str(int(d))+"_random"

result_pink, result_pig = pd.DataFrame(np.zeros(p), index=np.arange(p), columns=['Frequency']), pd.DataFrame(np.zeros(p), index=np.arange(p), columns=['Frequency'])
sel_count_pink, sel_count_pig = [], []

for i in range(nrep):
    print("########## Run",i+1)
    ind_col = screen_top[range(d),i].astype(int)
    X1 = X[:,ind_col]
    ## autoencoder ##
    r_hat = 3
    autoencoder = Sequential()
    autoencoder.add(Dense(d, activation=aut_met, use_bias=True, input_shape=(d,)))
    autoencoder.add(Dense(r_hat, activation=aut_met, use_bias=True))
    autoencoder.add(Dense(d, activation=aut_met, use_bias=True))
    autoencoder.compile(loss=aut_loss, optimizer=keras.optimizers.Adam())
    autoencoder.fit(X1, X1, epochs=aut_epoch, batch_size=bs, verbose=aut_verb)
    C = autoencoder.predict(X1)
    E = X1 - C
    sigma = np.sqrt(np.sum(E ** 2) / (n * d))
    X1_ko = C + sigma * np.random.randn(n, d)
    Xnew = np.hstack((X1, X1_ko))
    #################

    random.seed(58*i)
    ######### load data #########    
    ind_dist = indmat_dist[i,:] - 1
    ind_dnn = list(set(np.arange(n)) - set(ind_dist))
    
    ind_train = np.random.choice(ind_dnn, n_train, False)
    ind_test = list(set(ind_dnn) - set(ind_train))
    
    Xnew_train = Xnew[ind_train,:]
    Xnew_test = Xnew[ind_test,:]
    y_train = y[ind_train]
    y_test = y[ind_test]
    
    indmat[:,i] = ind_train

    #### DeepPINK ####
    es = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
    dp = Sequential()
    dp.add(PairwiseConnected(input_shape=(2 * d,)))
    dp.add(Dense(d, activation=dnn_met,
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
    dp.add(Dense(1, activation='relu',
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
    dp.compile(loss=dnn_loss, optimizer=keras.optimizers.Adam())
    dp.fit(Xnew_train, y_train, epochs=dnn_epoch, batch_size=bs, verbose=dnn_verb)
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


    print("DP selected :",selected_pink)
    result_pink.iloc[selected_pink] += 1
    mat_selected_pink[i,selected_pink] = 1
    sel_count_pink.append(len(selected_pink)) 

    yhat_train_pink[i,:] = [1 if a > 0.5 else 0 for a in dp.predict(Xnew_train).flatten()]
    pe_train_pink[i] = np.mean(yhat_train_pink[i,:] != y_train.flatten())


    #### PIGNET ####
    pig = DeepPIG(Xnew.shape[1], print_interval=100).to(device)
    pig.fit(Xnew_train, y_train, max_epoch=dnn_epoch,lr=lr,l1=l1, patience=30,y_design='clf',es='./checkpoint/'+pig_name+"_"+str(i)+'.pt')

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
        s = len(selected_pink)
        mrefit = Sequential()
        es = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
        mrefit.add(Dense(s, input_dim=s, activation='relu',kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
        mrefit.add(Dense(1, activation='sigmoid',kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
        mrefit.compile(loss=dnn_loss, optimizer=keras.optimizers.Adam(learning_rate=0.005))
        mrefit.fit(X1[ind_train,:][:,selected_pink], y[ind_train], epochs=300, batch_size=bs, verbose=dnn_verb, validation_split=0.1,callbacks=[es])
        
        yhat_test_pink[i,:] = [1 if a > 0.5 else 0 for a in mrefit.predict(X1[ind_test,:][:,selected_pink]).flatten()]
        pe_test_pink[i] = np.mean(yhat_test_pink[i,:] != y_test.flatten())
    print("PE_TEST(DP):",pe_test_pink[i] )
    ###########################
    ## refit to calculate PE (PIG)##
    if len(selected_pig) == 0:
        pe_test_pig[i] = np.nan
    else:
        s = len(selected_pig)
        mrefit_pig = Sequential()
        es = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
        mrefit_pig.add(Dense(s, input_dim=s, activation='relu',kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
        mrefit_pig.add(Dense(1, activation='sigmoid',kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
        mrefit_pig.compile(loss=dnn_loss, optimizer=keras.optimizers.Adam(learning_rate=0.005))
        mrefit_pig.fit(X1[ind_train,:][:,selected_pig], y[ind_train], epochs=300, batch_size=bs, verbose=dnn_verb, validation_split=0.1,callbacks=[es])
        
        yhat_test_pig[i,:] = [1 if a > 0.5 else 0 for a in mrefit_pig.predict(X1[ind_test,:][:,selected_pig]).flatten()]
        pe_test_pig[i] = np.mean(yhat_test_pig[i,:] != y_test.flatten())
    print("PE_TEST(PIG):",pe_test_pig[i])
    ###########################
    ## refit to calculate PE (all)##
    mrefit_all = Sequential()
    es = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
    mrefit_all.add(Dense(d, input_dim=d, activation='relu',kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
    mrefit_all.add(Dense(1, activation='sigmoid',kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
    mrefit_all.compile(loss=dnn_loss, optimizer=keras.optimizers.Adam(learning_rate=0.005))
    mrefit_all.fit(X1[ind_train,:], y[ind_train], epochs=300, batch_size=bs, verbose=dnn_verb, validation_split=0.1,callbacks=[es])
    
    yhat_test_all[i,:] = [1 if a > 0.5 else 0 for a in mrefit_all.predict(X1[ind_test,:]).flatten()]
    pe_test_all[i] = np.mean(yhat_test_all[i,:] != y_test.flatten())
    ###########################

    ###########################   
    ## refit to calculate PE (random)##
    if len(selected_pig) == 0:
        pe_test_random[i] = np.nan

    else:
        random_feature = np.random.choice(d,len(selected_pig), replace=False)
        mrefit_random = Sequential()
        es = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
        mrefit_random.add(Dense(s, input_dim=s, activation='relu',kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
        mrefit_random.add(Dense(1, activation='sigmoid',kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)))
        mrefit_random.compile(loss=dnn_loss, optimizer=keras.optimizers.Adam(learning_rate=0.005))
        mrefit_random.fit(X1[ind_train,:][:,random_feature], y[ind_train], epochs=300, batch_size=bs, verbose=dnn_verb, validation_split=0.1,callbacks=[es])
        
        yhat_test_random[i,:] = [1 if a > 0.5 else 0 for a in mrefit_random.predict(X1[ind_test,:][:,random_feature]).flatten()]
        pe_test_random[i] = np.mean(yhat_test_random[i,:] != y_test.flatten())
    ###########################
    print("AVG.so_far (DP):","TestError:",np.nanmean(pe_test_pink[:i+1]),"#.sel:",np.mean(sel_count_pink),"Unselected",len(np.where(sel_count_pink==0)[0]))
    print("AVG.so_far (PIG):","TestError:",np.nanmean(pe_test_pig[:i+1]),"#.sel:",np.mean(sel_count_pig),"Unselected",len(np.where(sel_count_pig==0)[0]))
    print("AVG.so_far (all)",np.nanmean(pe_test_all[:i+1]))
    print("AVG.so_far (random)",np.nanmean(pe_test_random[:i+1]))



pd.DataFrame(mat_selected_pink).to_csv(pink_name+'_selected.csv', index=True, header=True, sep=',')
pd.DataFrame(pe_train_pink).to_csv('real_pe_train_' + pink_name +".csv", index=True, header=True, sep=',')
pd.DataFrame(pe_test_pink).to_csv('real_pe_test_' + pink_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_test_pink).to_csv('real_yhat_test_' + pink_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_train_pink).to_csv('real_yhat_train_' + pink_name+".csv", index=True, header=True, sep=',')

print("#### pink ####")
print('Training average PE: ',np.mean(pe_train_pink),'Test average PE: ' ,np.mean(pe_test_pink))
print("Max.",np.max(result_pink['Frequency']), "Avg.",np.mean(sel_count_pink), "Redundant iter:", nrep - len(np.nonzero(sel_count_pink)[0]))

pd.DataFrame(mat_selected_pig).to_csv(pig_name+'_selected.csv', index=True, header=True, sep=',')
pd.DataFrame(pe_train_pig).to_csv('real_pe_train_' + pig_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(pe_test_pig).to_csv('real_pe_test_' + pig_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_test_pig).to_csv('real_yhat_test_' + pig_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_train_pig).to_csv('real_yhat_train_' + pig_name+".csv", index=True, header=True, sep=',')

print("#### pig ####")
print('Training average PE: ',np.mean(pe_train_pig),'Test average PE: ' ,np.mean(pe_test_pig))
print("Max.",np.max(result_pig['Frequency']), "Avg.",np.mean(sel_count_pig), "Redundant iter:", nrep - len(np.nonzero(sel_count_pig)[0]))

pd.DataFrame(pe_test_all).to_csv('real_pe_test_' + all_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_test_all).to_csv('real_yhat_test_' + all_name+".csv", index=True, header=True, sep=',')

print("#### All ####")
print('Test average PE: ' ,np.mean(pe_test_all))

pd.DataFrame(pe_test_random).to_csv('real_pe_test_' + random_name+".csv", index=True, header=True, sep=',')
pd.DataFrame(yhat_test_random).to_csv('real_yhat_test_' + random_name+".csv", index=True, header=True, sep=',')


print("#### Random ####") 
print('Test average PE: ' ,np.mean(pe_test_random))