# Adopted from Stefani Brilli's implementation. 
# 
# Navid Shokouhi
# email: navid.shokouhi@unimelb.edu.au
# August 2017
import numpy as np
import estimators as est

class Rbfn(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def sim_elm(self, x):
        """
        Run the network over a single input and return the output value
        For when ibias is a diagonal matrix
        """
        v = np.atleast_2d(x)[:, np.newaxis]-self.centers[np.newaxis, :]
        v = np.sqrt( (v**2.).sum(-1) )
        v = np.dot(v,self.ibias)
        v = np.exp(-v**2.)
        v = np.dot(v, self.linw) + self.obias
        return v

    def sim(self, x):
        """
        Run the network over a single input and return the output value
        """
        if not(np.isscalar(self.ibias)):
            return self.sim_elm(x)
        v = np.atleast_2d(x)[:, np.newaxis]-self.centers[np.newaxis, :]
        v = np.sqrt( (v**2.).sum(-1) ) * self.ibias
        v = np.exp( -v**2. )
        v = np.dot(v, self.linw) + self.obias
        return v
    

    def _input_size(self):
        try:
            return self.centers.shape[1]
        except AttributeError:
            return -1
    input_size = property(_input_size)

    def _output_size(self):
        try:
            return self.linw.shape[1]
        except AttributeError:
            return -1
    output_size = property(_output_size)

    def get_G(self,I,used):
        """
        Returns estimated RBF parameters
        """
        if not(np.isscalar(self.ibias)):
            # Sometimes, as in the case of ELM, 
            # we'd like to use different sigma 
            # for the different centers. 
            temp = ((I[np.newaxis,:,:] - I[:, np.newaxis, :])**2.).sum(-1)
            temp = temp[:,used]
            P = np.exp(-( np.dot(temp,self.ibias**2.0)))
            return np.array(P)

        P = np.exp(-( np.sqrt(((I[np.newaxis,:] - I[:, np.newaxis])**2.0).sum(-1)) * self.ibias)**2.0)
        return np.array(P)[:,used]

    def get_theta(self):
        return self.linw

    def set_theta(self,t):
        self.linw = t

    def get_centers(self):
        return self.centers

    def iterative_rbf(self,I,O,center_idx,alpha=0.):
        # index of centers in input vectors I
        G = self.get_G(I,center_idx)
        theta = self.get_theta()
        self.set_theta(est.iterative_ls(G,O,theta,alpha))