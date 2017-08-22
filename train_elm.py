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
    return
    
