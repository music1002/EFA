import pandas as pd
import numpy as np
import seaborn as sns

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
X_train = train.iloc[:,8:-3]
X_test=test.iloc[:8:-3]
target_tr= train.iloc[:,-1:]



x=X_train.values
x_mean=np.mean(x,axis=0)
x_norm=x-np.matrix(x_mean)
x_norm=x_norm.T #important otherwise we'll end up having a large dimension of the covariance matrix

c=np.cov(x_norm)
co=np.corrcoef(x_norm)
ax=sns.heatmap(co,cmap='Greens')

eig_val,eig_vec=np.linalg.eig(c)
eig_sort= np.sort(eig_val)[::-1]
arg_sort=np.argsort(eig_val)[::-1]
eig_vec_ls=[]
eig_val_ls=[]
i_vec=arg_sort[:5]
for i in i_vec:
    eig_vec_ls.append(eig_vec[:,i])
    eig_val_ls.append(eig_val[i])
    
#estimation of parameter V
eig_val_arr= np.array(eig_val_ls)
lm=np.diag(eig_val_arr)
eig_vec_mat= np.matrix(eig_vec_ls).T
V=eig_vec_mat@np.sqrt(lm)
print("V:",V)
print("*"*40)
#Estimation of S
var_ls=[]
var_x=np.var(x_norm,axis=1)
var_x=np.ravel(var_x)
for i in range(V.shape[0]):
    s=np.sum(np.square(np.ravel(V[i,:])))
    sig_sq=var_x[i]-s
    var_ls.append(sig_sq)
var_ls=np.array(var_ls)
s=np.diag(var_ls)
c_inv=np.linalg.inv(c)
w=V.T@c_inv
z=w@x_norm
z_=z.T
print(z_.shape)