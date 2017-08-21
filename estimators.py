import paths
import numpy as np
import numpy.linalg as la 
from algopy import CGraph, Function, UTPM, dot, qr, eigh, inv, solve

def estimate_noise(G,O):
    err = O - np.dot(G,leastsquares(G,O))
    return np.mean(np.diag(np.dot(err,err.T)))

def leastsquares(A,b):
    Apinv = dot(la.pinv(dot(A.T,A)),A.T)
    W = dot(Apinv,b)
    return W/la.norm(W)

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
    c = k#(2*(k**2)*sigma_2)
    W = teta1 - c*teta2
    return W


if __name__=='__main__':
    np.random.seed(0)
    m = 20
    N = 1000
    X = np.random.randn(N,1)
    C = np.random.randn(m,1)
    k = 1.
    H = ((X[np.newaxis,:,:] - C[:, np.newaxis, :])**2.).sum(-1)
    H = np.exp(-( H*(k**2) ))
    H = H.T
    theta = 0.1*np.random.randn(m,1)
    noise = 0.01*np.random.randn(N,1)
    y = np.dot(H,theta) + noise
    theta_ls = la.lstsq(H,y)[0]
    theta_ls = theta_ls/la.norm(theta_ls)
    theta_hat = leastsquares(H,y)
    from matplotlib import pylab 
    pylab.plot(theta_ls,label='LS')
    pylab.plot(theta_hat,label='Pseudo Inv')
    pylab.plot(theta,label='True')
    pylab.legend()
    pylab.show()

