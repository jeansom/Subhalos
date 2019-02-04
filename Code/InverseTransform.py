import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

class InverseTransform():
    def __init__(self, pdf, vals, nsamples=100, args=(), cdfinv=None):
        self.range = vals
        self.pdf = lambda x: pdf( x, *args )
        self.pdf_arr = self.pdf(self.range)
        self.nsamples = nsamples
        if cdfinv == None: self.cdfinv = self.CDFInv()
        else: self.cdfinv = cdfinv
            
    def CDF(self, x, cdf=None):
        if cdf != None: 
            return cdf(x)
        else:
            return quad(self.pdf, self.range[0], x)[0] 
    
    def CDFInv(self, cdf=None):
        cum = []
        for i in (self.range):
            cum.append(self.CDF(i, cdf=cdf))
        return np.array(cum)/max(cum)
        
    def sample(self):
        rands = np.random.rand(self.nsamples)
        if callable(self.cdfinv): return self.cdfinv(rands)
        else: return np.interp(rands, self.cdfinv, self.range)