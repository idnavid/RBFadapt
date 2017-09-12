import numpy as np
import load_data
import sys
import pylab 
sys.path.append("../")
import gen_data
import train_elm as elm
import train_ols as ols 

def print_results(mse):
	"""
	mse is an IxM matrix of mean squared errors. 
	I: number of iterations
	m: number of network architectures
	"""
	avg_mse = np.mean(mse,axis=0)
	std_mse = np.std(mse,axis=0)
	for i in range(avg_mse.shape[0]):
		print(avg_mse[i],std_mse[i])




X_train,y_train = load_data.weather_ankara('train')
X_test, y_test  = load_data.weather_ankara('test')
N_train = y_train.shape[0]
N_test = y_test.shape[0]
# Set noise parameters
NOISE = 'A'
if NOISE=='H':
	# Heavy tail
	pi1 = 0.8
	mu1 = 0.;
	mu2 = 0.; 
	sigma1 = 1.
	sigma2 = 12.
	asym = False
elif NOISE=='P':
	pi1 = 0.8
	mu1 = 0.;
	mu2 = 10.; 
	sigma1 = 1.
	sigma2 = 0.25
	asym = False
elif NOISE=='A':
	pi1 = 0.8
	mu1 = 0.;
	mu2 = 1.; 
	sigma1 = 1.
	sigma2 = 10.
	asym = True
elif NOISE=='G':
	# Single Gaussian noise
	pi1 = 1.0
	mu1 = 0.;
	mu2 = 0.; 
	sigma1 = 1.
	sigma2 = 0.
	asym = False
elif NOISE=='0':
	# clean
	pi1 = 1.0
	mu1 = 0.;
	mu2 = 0.; 
	sigma1 = .01
	sigma2 = 0.
	asym = False

m_range = [10,15,20]
ITER = 20
mse_wml = np.zeros((ITER,len(m_range)))
mse_ml = np.zeros((ITER,len(m_range)))
for i_iter in range(ITER):
	for i_m in range(len(m_range)):
		noise = gen_data.bi_noise(N_train,pi1,sigma1,sigma2,mu1,mu2,asym)
		y_train_noisy = y_train + noise
		m = m_range[i_m]
		gw = 1.0
		r,center_idx = elm.train_elm(X_train,y_train_noisy,m,gw)
		y_hat = r.sim(X_test)
		mse_ml[i_iter,i_m] = ols.compute_mse(y_hat,y_test)
		alpha = 0.1
		r.iterative_rbf(X_train,y_train_noisy,center_idx,alpha)
		y_hat = r.sim(X_test)
		mse_wml[i_iter,i_m] = ols.compute_mse(y_hat,y_test)

print('ML: ')
print_results(mse_ml)
print('WML: ')
print_results(mse_wml)
