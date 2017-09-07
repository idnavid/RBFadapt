import numpy as np
import load_data
import sys
import pylab 
sys.path.append("../")
import gen_data
import train_elm as elm
import train_ols as ols 

X_train,y_train = load_data.auto_mpg('train')
X_test, y_test  = load_data.auto_mpg('test')

# Add noise
np.random.seed(0)
N_train = y_train.shape[0]
pi1 = 0.85
mu1 = 0.;
mu2 = 10.; 
sigma1 = 0.5
sigma2 = 0.25
asym = False
noise = gen_data.bi_noise(N_train,pi1,sigma1,sigma2,mu1,mu2,asym)
y_train_noisy = y_train + noise

m = 15
gw = 1.
r,center_idx = elm.train_elm(X_train, y_train_noisy,m,gw)   
alpha = .1
# r.iterative_rbf(X_train,y_train_noisy,center_idx,alpha)
y_hat = r.sim_elm(X_test)
print(ols.compute_mse(y_hat,y_test))
pylab.plot(y_hat,label='estimate')
pylab.plot(y_test,label='target')
pylab.legend()
pylab.show()