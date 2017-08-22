import numpy as np
import matplotlib.pyplot as plt
# Collection of methods used to generate synthetic 
# data and prepare real data for experiments. 

def bi_noise(N,p1=1.,sigma1=1.0,sigma2=1.0,mu1=0.0,mu2=0.0):
	# generate bimodal noise
	p2 = 1. - p1
	n1 = mu1 + sigma1*np.random.randn(int(N*p1),1)
	n2 = mu2 + sigma2*np.random.randn(int(N*p2),1)
	n = np.zeros((N,1))
	temp = np.append(n1,n2)
	n[:temp.shape[0],0] = temp 
	np.random.shuffle(n)
	return n
