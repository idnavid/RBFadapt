import numpy as np
import load_data
import sys
import pylab 
sys.path.append("../")
import gen_data
import train_elm as elm
import train_ols as ols 

X_train,y_train = load_data.ailerons('train')
X_test, y_test  = load_data.ailerons('test')

# Add noise
np.random.seed(0)
N_train = y_train.shape[0]
pi1 = 0.85
mu1 = 0.;
mu2 = 1.; 
sigma1 = 1.
sigma2 = 10.
asym = True
noise = gen_data.bi_noise(N_train,pi1,sigma1,sigma2,mu1,mu2,asym)
y_train_noisy = y_train + noise

m = 10
gw = 10.0
r,center_idx = elm.train_elm(X_train, y_train_noisy,m,gw)   
alpha = 0.1
r.iterative_rbf(X_train,y_train_noisy,center_idx,alpha)
y_hat = r.sim(X_test)
print(ols.compute_mse(y_hat,y_test))
pylab.plot(y_hat)
pylab.plot(y_test)
pylab.show()