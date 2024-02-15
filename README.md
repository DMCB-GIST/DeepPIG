# DeepPIG 
Deep neural network with PaIrwise connected layers integrated with stochastic Gates (DeepPIG) for the feature selection model.
This source code is for replication of the paper.
![alt text](https://github.com/DMCB-GIST/DeepPIG/blob/main/Fig1.png)


## Simulation study
Run DeepPIG for the simulation study via synthetic data.

**required input** x_design = {linear, logistic} , y_design = {linear, nonlinear}, amplitude
```
ex) python run_simulation.py -xd linear -y nonlinear -a 10
```

## Real data analysis
### Cancer prognosis (Long-term survivor classification)
**required input** cancer_type = {KIRC, LIHC, PAAD}
```
ex) python run_survival.py -o KIRC
```

### Human microbiome dataset
```
python run_human_microbiome.py
```
### Human single-cell dataset
```
python run_human_sc.py
```
### Murine single-cell dataset
```
python run_murine_sc.py
```

