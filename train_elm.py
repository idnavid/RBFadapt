import rbfn
from rbfn import Rbfn
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import estimators as est
import gen_data
import train_ols as ols

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
    x_train,y_train = gen_data.sinc(N_train)
    x_test,y_test = gen_data.sinc(N_test)

    # Add noise
    import sys
    pi1 = float(sys.argv[1])
    NOISE=(sys.argv[2])
    if NOISE=='H':# H for HEAVYTAIL
        mu1 = 0.
        mu2 = 0.
        sigma1 = .5
        sigma2 = 10.
        asym=False
    elif NOISE=='P':# P for POINTMASS
        mu1 = 0.
        mu2 = 10.
        sigma1 = .5
        sigma2 = .2
        asym=False
    elif NOISE=='A':# A for ASYMMETRIC
        mu1 = 0.
        mu2 = 1.
        sigma1 = .5
        sigma2 = 10.
        asym=True

    noise_train = gen_data.bi_noise(N_train,pi1,sigma1,sigma2,mu1,mu2,asym)
    noise_test = gen_data.bi_noise(N_test,pi1,sigma1,sigma2,mu1,mu2,asym)
    y_train_noisy = y_train + noise_train
    y_test_noisy = y_test + noise_test
    
    m = 12
    gw = 10.
    
    r,center_idx = train_elm(x_train, y_train_noisy,m,gw)   
    v_train_ml = r.sim_elm(x_train)
    v_test_ml = r.sim_elm(x_test)
    
    alpha = 0.1
    r.iterative_rbf(x_train,y_train_noisy,center_idx,alpha)
    v_train_wml = r.sim_elm(x_train)
    v_test_wml = r.sim_elm(x_test)
    
    
    # print("train ML: ",  compute_mse(v_train_ml ,y_train),end="\t")
    # print("train IT: ",  compute_mse(v_train_wml,y_train))
    # print("test ML : " ,  compute_mse(v_test_ml  , y_test),end="\t")
    # print("test IT : " ,  compute_mse(v_test_wml ,y_test))
    print("Train: ",ols.compute_mse(v_train_ml ,y_train),end=" ")
    print(" ",ols.compute_mse(v_train_wml,y_train))
    print("Test ",ols.compute_mse(v_test_ml  , y_test),end=" ")
    print(" ",ols.compute_mse(v_test_wml ,y_test))
    plt.plot(x_test,y_test_noisy,'k.')
    plt.plot(x_test,v_test_ml,   'y.')
    plt.plot(x_test,v_test_wml,  'g.')
    plt.plot(x_test,y_test,      'r.')
    plt.show()