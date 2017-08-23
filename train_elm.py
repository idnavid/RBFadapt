import rbfn
from rbfn import Rbfn
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import estimators as est
import gen_data


def train_elm(I, O,m=500,gw=1.):
    """
    ELM: Extreme Learning Macines
    ELM uses random centers/variances for each node, but
    the weights are estimated using LS. 
    From a well-cited paper:
    Extreme learning machine: Theory and applications
    """
    m = min(O.shape[0],m)
    # Pick n random centers
    center_idx = np.arange(O.shape[0])
    np.random.shuffle(center_idx)
    center_idx = center_idx[:m]
    
    # generate n random sigma 
    k = np.sqrt(-np.log(0.5))/gw
    sigma = k*np.random.uniform(low=1e-7,high = 1, size=m)
    sigma = sigma.reshape((m,1))
    sigma = np.diag(sigma[:,0])
    # sigma = np.eye(n)*k

    # pick centers
    temp = ((I[np.newaxis,:,:] - I[:, np.newaxis, :])**2.).sum(-1)
    temp = temp[:,center_idx]
    P = np.exp(-( np.dot(temp,sigma**2.0)))
    G = np.array(P)
    W = la.lstsq(G,O)[0]
    net = Rbfn(centers=I[center_idx,:], ibias=sigma, linw=W, obias=0)
    return net,center_idx

if __name__ == "__main__":
    np.random.seed(0)
    N_train = 500
    N_test = 5*N_train
    X_train,y_train = gen_data.sinc(N_train)
    X_test,y_test = gen_data.sinc(N_test)

    # Add noise
    pi1 = 0.85
    mu1 = 0.
    mu2 = 0.
    sigma1 = .5
    sigma2 = 10.
    noise_train = gen_data.bi_noise(N_train,pi1,sigma1,sigma2,mu1,mu2)
    noise_test = gen_data.bi_noise(N_test,pi1,sigma1,sigma2,mu1,mu2)
    m = 10
    gw = 1.0
    r,center_idx = train_elm(X_train, y_train + noise_train,m,gw)   
    V = r.sim_elm(X_train)
    err = abs(V - y_train)
    print("Train MSE-0: ", np.sqrt((err**2.).sum(0))/(N_test))
    V_test = r.sim_elm(X_test)
    err = abs(V_test - y_test)
    print("Test MSE   : ", np.sqrt((err**2.).sum(0))/(N_test))
    alpha = .511
    r.iterative_rbf(X_train,y_train+noise_train,center_idx,alpha)
    V = r.sim_elm(X_train)
    err = abs(V - y_train)
    print("Train MSE-1: ", np.sqrt((err**2.).sum(0))/(N_test))
    V_test1 = r.sim_elm(X_test)
    err = abs(V_test1 - y_test)
    print("Test MSE-1 : ", np.sqrt((err**2.).sum(0))/(N_test))
    
    plt.plot(X_test,y_test+noise_test,'.')
    plt.plot(X_test,V_test,'y.')
    plt.plot(X_test,V_test1,'g.')
    plt.plot(X_test,y_test,'r.')
    plt.show()
