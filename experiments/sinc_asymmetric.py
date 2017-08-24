# Asymmetric noise contamination

import numpy as np
from matplotlib import pylab

import sys
sys.path.append("../")
import gen_data
import train_elm

N_max = 1010
N_range = np.arange(100,N_max,100)
m = 20
gw = 1.0
mu1 = 0.;
mu2 = 1.; 
sigma1 = 2.;
sigma2 = 5.; 
mse_ml = np.zeros(N_range.shape)
mse_wml = np.zeros(N_range.shape)
mse = np.zeros(N_range.shape)
ITER = 10
for iter in range(ITER):
############################################################
	pi1 = 1.;
	noise = gen_data.bi_noise(N_max,pi1,sigma1,sigma2,mu1,mu2)
	for i in range(len(N_range)):
		n = N_range[i]
		x,y = gen_data.sinc(n)
		y_noisy = y + noise[:n]
		r,center_idx = train_elm.train_elm(x, y_noisy,m,gw)   
		V = r.sim_elm(x)
		err = abs(V - y)
		mse[i] += np.sqrt((err**2.).sum(0))/(n)
	   	

###########################################################
	pi1 = 0.85;
	noise = gen_data.bi_noise(N_max,pi1,sigma1,sigma2,mu1,mu2)
	for i in range(len(N_range)):
		n = N_range[i]
		x,y = gen_data.sinc(n)
		y_noisy = y + noise[:n]
		r,center_idx = train_elm.train_elm(x, y_noisy,m,gw)   
		V = r.sim_elm(x)
		err = abs(V - y)
		mse_ml[i] += np.sqrt((err**2.).sum(0))/(n)
		alpha = 0.01
		r.iterative_rbf(x,y_noisy,center_idx,alpha)
		V = r.sim_elm(x)
		err = abs(V - y)
		mse_wml[i] += np.sqrt((err**2.).sum(0))/(n)

pylab.plot(N_range,mse/ITER,label='ML single Gaussian')
pylab.plot(N_range,mse_ml/ITER,label='ML point-mass')
pylab.plot(N_range,mse_wml/ITER,'r-.',label='Proposed point-mass')
pylab.legend()
pylab.grid()
pylab.show()