import paths
import numpy as np
import numpy.linalg as la 
from matplotlib import pylab

import gen_data

def estimate_noise(G,O):
    err = O - np.dot(G,leastsquares(G,O))
    return np.mean(np.diag(np.dot(err,err.T)))

def iterative_ls(A,b,x0,alpha = 0.0):
    """
    This estimator provides a weighted estimation 
    of the solution to Ax=b, that is not Least Squares. 
    To find this solution, we need:
    1. an initial estimate of x (i.e., x0)
    2. an iterative update of x.  

    The estimator has an intrinsic parameter, alpha, that 
    differentiates it from least-squares.
        if alpha = 0. ==> x = x_ls
        alpha>0. ==> x is different from ls
    """
    print(alpha)
    err = (np.dot(A,x0) - b)[:,0]
    W = np.exp(-alpha * (err**2))
    W = W/np.sum(W)
    W = np.sqrt(np.diag(W))
    for i in range(5):
        x = la.lstsq(np.dot(W,A),np.dot(W,b))[0]
    return x

def leastsquares(A,b):
    Apinv = np.dot(la.pinv(np.dot(A.T,A)),A.T)
    W = np.dot(Apinv,b)
    return W

def stein_estimator(G,O,I,k):
    """
    This estimator was based on an idea that 
    tried to reduce the error between the RBF output 
    and the true (hidden) function (f(x,y)). 
    w = argmin{|| f_w(x,y) - f(x)||^2}
    Normally, what people do is that they use
    w = argmin{|| f_w(x,y) - y||^2}
    
    Unfortunately, for the idea to work,
    I needed an estimate of d(x)/d(noise). ==> which is hard to get.  
    """
    teta1 = leastsquares(G,O)
    tau = np.gradient(I,axis=0)
    temp = np.diag(np.dot(tau,I.T))
    teta2 = np.dot(G.T,temp)
    teta2 = teta2.reshape((teta2.shape[0],1))
    sigma_2 = estimate_noise(G,O)
    W = teta1 - k*teta2
    return W


if __name__=='__main__':
    np.random.seed(0)
    m = 20
    N = 1000
    X = 0  + 10.*np.random.randn(N,1)
    C = 0 + 10.*np.random.randn(m,1)
    k = 10.
    H = ((X[np.newaxis,:,:] - C[:, np.newaxis, :])**2.).sum(-1)
    H = np.exp(-( H*(k**2) ))
    H = H.T
    theta = 0.1*np.random.randn(m,1)
    
    # Generate noise
    p1 = 1.0
    mu1 = 0. 
    mu2 = 0.
    sigma1 = .1
    sigma2 = 1
    noise = gen_data.bi_noise(N,p1,sigma1,sigma2,mu1,mu2)

    #
    y = np.dot(H,theta) + noise
    theta_ls = la.lstsq(H,y)[0]
    alpha = 0.0
    theta_i = iterative_ls(H,y,theta_ls,alpha)
    err1 = theta_ls - theta
    err2 = theta_i - theta
    print('LS : ',la.norm(err1))
    print('WLS: ',la.norm(err2))

    pylab.hist(noise,100)
    pylab.show()
    pylab.plot(theta_ls,label='LS')
    pylab.plot(theta_i,label='WLS')
    pylab.plot(theta,label='True')
    pylab.legend()
    pylab.show()


