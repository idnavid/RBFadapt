import numpy as np
import matplotlib.pyplot as plt
import collections
# Collection of methods used to generate synthetic 
# data and prepare real data for experiments. 

def bi_noise(N,p1=1.,sigma1=1.0,sigma2=1.0,mu1=0.0,mu2=0.0,asym=False):
	# generate bimodal noise
	p2 = 1. - p1
	n1 = mu1 + sigma1*np.random.randn(int(N*p1),1)
	n2 = mu2 + sigma2*np.random.randn(int(N*p2),1)
	n = np.zeros((N,1))
	if asym:
		n2 = np.abs(n2)

	temp = np.append(n1,n2)
	n[:temp.shape[0],0] = temp 
	np.random.shuffle(n)
	return n

def add_outliers(x,p_outliers=0.0,sigma=0.0):
	"""
	Add outliers to a portion of the data (p_outliers) with random noise. 

	This definition of contamination comes from:
	Yang etal 2013, "A novel self-constructing Radial Basis Function 
	Neural-Fuzzy System".
	Later, Vuković and Miljković claim to use this in  
	"Robust sequential learning of feedforward neural networks in the presence of 
	heavy-tailed noise. " 
	but it looks like Vukovic and co. replace the signal with noise (instead of adding it).
	I intend to replicate their results, but I'll stick to my interpretatoin of
	Yang's simulation, which is to add outliers to a portion of the data.  
	p_outliers: a value between 0 and 1.
	x: input signal
	"""
	x_clean = x
	N = len(x)
	N_noise = int(N*p_outliers)
	# noise = 2*sigma*(np.random.rand(N_noise,1)-0.5)
	noise = sigma*np.random.randn(N_noise,1)
	# print(noise.shape)
	idx_noise = np.array(range(N))
	np.random.shuffle(idx_noise)
	idx_noise = idx_noise[:N_noise]
	x[idx_noise] = noise + x[idx_noise]
	x = x + bi_noise(N,p1=1-p_outliers,sigma1=0.0,sigma2=0.0)
	return x



def sinc(N):
	d = 1
	I = (np.random.uniform(size=(N,d), low=-10., high=10.))
	O = np.zeros(I.shape)
	non_zero = np.where(I!=0)
	O[non_zero] = np.sin(I[non_zero])/I[non_zero]
	is_zero = np.where(I==0)
	O[is_zero] = 1
	return I,O


def mackeyglass(N_train,tau=17, seed=None):
	"""
	Found this on:
		https://github.com/mila-udem/summerschool2015/blob/master/rnn_tutorial/synthetic.py
	mackeyglass(N_train=1000, tau=17, seed = None, N_test = 1) -> input
	Generate the Mackey Glass time-series. Parameters are:
	    - N: length of the time-series in timesteps. Default is 1000.
	    - tau: delay of the MG - system. Commonly used values are tau=17 (mild 
	      chaos) and tau=30 (moderate chaos). Default is 17.
	    - seed: to seed the random generator, can be used to generate the same
	      timeseries at each invocation.
	"""
	delta_t = 1
	history_len = tau * delta_t 
	# Initial conditions for the history of the system
	timeseries = 1.2

	if seed is not None:
	    np.random.seed(seed)

	samples = []

	history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
	                            (np.random.rand(history_len) - 0.5))
	# Preallocate the array for the time-series
	inp = np.zeros((N_train,1))
	for timestep in range(N_train):
	    for _ in range(delta_t):
	        xtau = history.popleft()
	        history.append(timeseries)
	        timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
	                     0.1 * history[-1]) / delta_t
	    inp[timestep] = timeseries

	# Squash timeseries through tanh
	#inp = np.tanh(inp - 1)
	samples.append(inp)
	I = np.array(range(N_train))
	I = I.reshape((len(I),1))
	return I,samples[0]

if __name__=='__main__':
	x_train,y_train = mackeyglass(500)
	x_test,y_test = mackeyglass(500)
	x,y = sinc(500)
	# print(x.shape,y.shape)
	# print(x_train.shape,y_train.shape)
	plt.plot(x_train,y_train)
	y_train_noisy = insert_outliers(y_train,p_outliers=0.1,sigma=0.3)
	plt.plot(x_train,y_train_noisy,'r.')
	plt.show()
	