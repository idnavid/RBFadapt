# Point-mass noise contamination
import sys
import numpy as np
from matplotlib import pylab
pylab.rcParams['font.size'] = 18

import sys
sys.path.append("../")
import gen_data
import train_kmeans

N_max = 3010
N_range = np.arange(100,N_max,100)
m = 15
gw = 1.0
mu1 = 0.;
mu2 = 3.; 
sigma1 = 1.;
sigma2 = 10.; 
mse_ml = np.zeros(N_range.shape)
mse_wml = np.zeros(N_range.shape)
mse = np.zeros(N_range.shape)
ITER = 1
try:
	alpha = float(sys.argv[1])
except:
	alpha = 0.

for iter in range(ITER):
############################################################
	pi1 = 1.;
	noise = gen_data.bi_noise(N_max,pi1,sigma1,sigma2,mu1,mu2)
	for i in range(len(N_range)):
		n = N_range[i]
		x,y = gen_data.sinc(n)
		y_noisy = y + noise[:n]
		r,center_idx = train_kmeans.train_kmeans(x, y_noisy,m,gw)   
		V = r.sim_elm(x)
		err = abs(V - y)
		mse[i] += np.sqrt((err**2.).sum(0))/(n)
	pylab.plot(x,y_noisy,'k*',label='Noisy data')	
	pylab.plot(x,V,'g*',label='ML estimation',linewidth=2)		   	
	###########################################################
	pi1 = 0.8;
	noise = gen_data.bi_noise(N_max,pi1,sigma1,sigma2,mu1,mu2,asym=True)
	for i in range(len(N_range)):
		n = N_range[i]
		x,y = gen_data.sinc(n)
		y_noisy = y + noise[:n]
		r,center_idx = train_kmeans.train_kmeans(x, y_noisy,m,gw)   
		V = r.sim_elm(x)
		err = abs(V - y)
		mse_ml[i] += np.sqrt((err**2.).sum(0))/(n)
		
		r.iterative_rbf(x,y_noisy,center_idx,alpha)
		V = r.sim_elm(x)
		err = abs(V - y)
		mse_wml[i] += np.sqrt((err**2.).sum(0))/(n)
pylab.plot(x,V,'b.',label='Proposed estimation',linewidth=2)
pylab.plot(x,y,'r.',label='Sinc',linewidth=0.5)	
pylab.title('Sinc in assymetric noise')
pylab.legend()
pylab.show()
pylab.plot(N_range,(mse/ITER),'k',label='ML single Gaussian')
pylab.plot(N_range,(mse_ml/ITER),'g',label='ML asymmetric')
pylab.plot(N_range,(mse_wml/ITER),'r-.',label='Proposed asymmetric')
pylab.xlabel('N')
pylab.ylabel('MSE')
pylab.title('Asymmetric')
pylab.legend()
pylab.grid()
pylab.show()