import numpy as np
import pylab
import sys
sys.path.append("../")
import gen_data
import train_elm as elm 
import train_ols as ols
import train_kmeans as kmeans 

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
	pylab.plot(avg_mse)


k = 1
N_train = 500
N_test = 500

ITER = 100
p_range = [0.,0.1,0.2,0.3,0.4,0.5]
mse_wml = np.zeros((ITER,len(p_range)))
mse_ml = np.zeros((ITER,len(p_range)))
for i_iter in range(ITER):
	for i_p in range(len(p_range)):
		p_outliers = p_range[i_p]
		x,y = gen_data.mackeyglass(k*N_train+N_test)
		x_train = x[:k*N_train]
		np.random.shuffle(x_train)
		x_train = x_train[:N_train]
		y_train = y[x_train][:,:,0]
		x_test = x[k*N_train:] - k*N_train
		y_test = y[x_test][:,:,0]



		
		y_train_noisy = gen_data.add_outliers(y_train,p_outliers,2.)

		m = 30
		gw = 1.
		r,center_idx = kmeans.train_kmeans(x_train, y_train_noisy,m,gw)   
		y_hat = r.sim_elm(x_test)
		# pylab.plot(y_hat,label='ML')
		mse_ml[i_iter,i_p] = ols.compute_mse(y_hat,y_test)
		alpha = .99
		r.iterative_rbf(x_train,y_train_noisy,center_idx,alpha)
		y_hat = r.sim_elm(x_test)
		# pylab.plot(y_hat,label='WML')
		# pylab.plot(y_test,label='orig')
		# pylab.show()
		mse_wml[i_iter,i_p] = ols.compute_mse(y_hat,y_test)

print('ML')
print_results(mse_ml)
print('WML')
print_results(mse_wml)
pylab.show()

