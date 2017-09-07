import rbfn
from rbfn import Rbfn
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import estimators as est
import gen_data


def train_ols(I, O, mse, gw=1.0, verbose=False):
    """
    Build a rbfn
    I (N by M) N vector of M size
    O (N by T) N vector of T size
    """
    k = np.sqrt(-np.log(0.5))/gw
    m, d = O.shape
    d *= m
    idx = np.arange(m)
    # P is a matrix of size NxN (e.g., 1000x1000), whose ij element is:
    # sum((Ii - Ij)^2). Ik is the k-th input sample. Sum adds values across dimensions. 
    # Therefore, the ij-th element of P contains the exp of the -1*Euclidean distance between
    # the i-th and j-th samples. 
    P = np.exp(-( np.sqrt(((I[np.newaxis,:] - I[:, np.newaxis])**2.0).sum(-1)) * k)**2.0)
    G = np.array(P)
    D = (O*O).sum(0)
    # The following line implements Eq.(24) from (Chen 1991)
    e = ( ((np.dot(P.T, O)**2.) / ((P*P).sum(0)[:,np.newaxis]*D) )**2.).sum(1)
    next = e.argmax()
    center_idx = np.array([next])
    idx = np.delete(idx, next)
    W = P[:,next, np.newaxis]
    P = np.delete(P, next, 1)
    G1 = G[:, center_idx]
    t, r, _, _ = la.lstsq(G1, O)
    err = r.sum()/d
    while err > mse and P.shape[1] > 0:
        if verbose:
            print(err, m-P.shape[1])
        # wj are the orthonormal bases
        wj = W[:, -1:]
        # aik are the elements of the upper-tri matrix
        a = np.dot(wj.T, P)/np.dot(wj.T, wj)
        P = P-wj*a
        not_zero = np.ones((P.shape[1]))*np.finfo(np.float64).eps
        e = (((np.dot(P.T, O)**2.) / ((P*P).sum(0)[:,np.newaxis]*D+not_zero) )**2.).sum(1)
        next = e.argmax()
        W = np.append(W, P[:,next, np.newaxis], axis=1)
        center_idx = np.append(center_idx, idx[next])
        P = np.delete(P, next, 1)
        idx = np.delete(idx,next)
        t, r, _, _ = la.lstsq(G[:, center_idx], O)
        err = r.sum()/d
    if verbose:
        print(err, m-P.shape[1])
    net = rbfn.Rbfn(centers=I[center_idx], linw=t, ibias=k, obias=0.)
    return net,center_idx

def compute_mse(y_est,y_true):
    N = y_est.shape[0]
    err = abs(y_est - y_true)
    mse = np.sqrt((err**2.).sum(0))/(N)
    return mse[0] 

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
    
    train_mse = .1
    gw = 10.
    
    r,center_idx = train_ols(x_train, y_train_noisy,train_mse,gw,True)   
    v_train_ml = r.sim(x_train)
    v_test_ml = r.sim(x_test)
    
    alpha = .1
    r.iterative_rbf(x_train,y_train_noisy,center_idx,alpha)
    v_train_wml = r.sim(x_train)
    v_test_wml = r.sim(x_test)
    
    
    # print("train ML: ",  compute_mse(v_train_ml ,y_train),end="\t")
    # print("train IT: ",  compute_mse(v_train_wml,y_train))
    # print("test ML : " ,  compute_mse(v_test_ml  , y_test),end="\t")
    # print("test IT : " ,  compute_mse(v_test_wml ,y_test))
    print("Train: ",compute_mse(v_train_ml ,y_train),end=" ")
    print(" ",      compute_mse(v_train_wml,y_train))
    print("Test ",  compute_mse(v_test_ml  , y_test),end=" ")
    print(" ",      compute_mse(v_test_wml ,y_test))
    plt.plot(x_test,y_test_noisy,'k.')
    plt.plot(x_test,v_test_ml,   'y.')
    plt.plot(x_test,v_test_wml,  'g.')
    plt.plot(x_test,y_test,      'r.')
    plt.show()