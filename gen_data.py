import numpy as np
import matplotlib.pyplot as plt
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

def sinc(N):
	d = 1
	I = (np.random.uniform(size=(N,d), low=-10., high=10.))
	O = np.zeros(I.shape)
	non_zero = np.where(I!=0)
	O[non_zero] = np.sin(I[non_zero])/I[non_zero]
	is_zero = np.where(I==0)
	O[is_zero] = 1
	return I,O
