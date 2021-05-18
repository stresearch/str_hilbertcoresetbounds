import numpy as np
from ..util.errors import NumericalPrecisionError
from .snnls import SparseNNLS

class FrankWolfe(SparseNNLS):

  def __init__(self, A, b, checkpoint, checkpoint_loc):
    super().__init__(A, b, checkpoint, checkpoint_loc)

    self.Anorms = A.get_norm(np.ones(A.num_cols))
    idc = np.argwhere(self.Anorms<1E-6)
    self.Anorms[idc] = 1E-6
    if np.any( self.Anorms == 0):
      raise ValueError(self.alg_name+'.__init__(): A must not have any 0 columns')

  def _select(self):
    residual = self.b - self.A.dot_product(self.w)
    return (self.A.dot_product_normalized_trans(residual, self.Anorms)).argmax()

  def _reweight(self, f):
    if self.size() == 0:
      #special case if this is the first point to add (places iterate on constraint polytope)
      alpha = 0.
      beta = self.Anorms.sum() / self.Anorms[f]
    else:
      nsum = self.Anorms.sum()
      nf = self.Anorms[f]
      xw = self.A.dot_product(self.w)
      xf = self.A.get_col(f)

      gammanum = (nsum/nf*xf - xw).dot(self.b-xw)
      gammadenom = ((nsum/nf*xf-xw)**2).sum()

      if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:
        raise NumericalPrecisionError('precision loss in gammanum/gammadenom: num = ' + str(gammanum) + ' denom = ' + str(gammadenom))

      alpha = 1. - gammanum/gammadenom
      beta = nsum/nf*gammanum/gammadenom

    self.w = alpha*self.w
    self.w[f] = max(0., self.w[f]+beta)

  

