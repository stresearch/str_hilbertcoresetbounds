"""
COpyright Systems & Technology Research  2020-2021

This module does the finite dimensional projection of the posterior distribution
"""
import numpy as np

class BayesianTangentSpaceFactory(object):
  def __init__(self, loglike, sampler, proj_dim):
    self.proj_dim = proj_dim
    self.loglike = loglike
    self.sampler = sampler

  def __call__(self, w = None, ids = None):
    prms = self.sampler(self.proj_dim, w, ids)
    temp = self.loglike(prms)
    D = temp.shape[2]
    dim = np.random.randint(0,D,temp.shape[1])
    vecs = np.zeros((temp.shape[0],temp.shape[1]))
    for i in range(temp.shape[1]):
        vecs[:,i] = D * temp[:,i,dim[i]]
    return vecs


