from numpy import save as np_save, load as np_load
from scipy.io import savemat, loadmat

class NPY():
    def __init__(self, allow_pickle=False):
        self.allow_pickle = allow_pickle

    def save(self, arr, pth):
        with open(pth, 'wb+') as fh:
            np_save(fh, arr, allow_pickle=self.allow_pickle)
    
    def load(self, pth):
        return np_load(pth, allow_pickle=self.allow_pickle)

class MatFile():
	def save(self, arr, pth):
		with open(pth, 'w+') as fh:
			savemat(fh, dict(data=arr))

	def load(self, pth):
		with open(pth, 'r') as fh:
			return loadmat(fh)['data']