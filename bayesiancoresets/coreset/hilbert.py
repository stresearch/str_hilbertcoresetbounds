import numpy as np
from ..util.errors import NumericalPrecisionError
from .coreset import Coreset
from bayesiancoresets.snnls.frankwolfe import FrankWolfe
from bayesiancoresets.util.norm_calc import NormCalc

class HilbertCoreset(Coreset):
  def __init__(self, vecs, prob, new_files, checkpoint, checkpoint_loc, gpu_list, snnls = FrankWolfe, **kw):
    norm = vecs.multiply(prob, new_files)
    self.snnls = snnls(NormCalc(new_files, vecs.num_rows, vecs.num_cols, gpu_list), norm, checkpoint, checkpoint_loc)
    super().__init__(**kw)

  def reset(self):
    self.snnls.reset()
    super().reset()

  def _build(self, itrs, sz):
    if self.snnls.size()+itrs > sz:
      raise ValueError(self.alg_name + '._build(): # itrs + current size cannot exceed total desired size sz. # itr = ' + str(itrs) + ' cur sz: ' + str(self.snnls.size()) + ' desired sz: ' + str(sz))
    self.snnls.build(itrs)
    w = self.snnls.weights()
    self._overwrite(w[w>0], np.where(w>0)[0])

  def _optimize(self):
    self.snnls.optimize()
    w = self.snnls.weights()
    self._overwrite(w[w>0], np.where(w>0)[0])

  def error(self):
    return self.snnls.error()

  def start_from_checkpoint(self, weights, its):
    self.snnls.start_from_checkpoint(weights, its)
