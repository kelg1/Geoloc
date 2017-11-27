import numpy as np 

class SIN:
    def __init__(self, k, ak):
        self.k = k 
        self.ak = ak
        
    def __str__(self):
        return '{ak} . sin({k} x)'.format(ak = self.ak, k = self.k)
    
    def value(self, theta):
        return self.ak * np.sin(self.k * theta)
    
class COS:
    def __init__(self, k, ak):
        self.k = k 
        self.ak = ak
        
    def __str__(self):
        return '{ak} . cos({k} x)'.format(ak = self.ak, k = self.k)
    
    def value(self, theta):
        return self.ak * np.cos(self.k * theta )
    
def ind(r, a, b, boolean=False):
    a_ = min(a, b)
    b_ = max(a, b)
    
    if boolean:
        f = np.logical_and(r<b_, r >= a_)
    else: 
        f = 1*np.logical_and(r<b_, r >= a_)
    return f

class IND:
    def __init__(self, a, b, beta):
        if a >= b:
            raise('a < b')
        else:
            self.a = a 
            self.b = b
            self.beta = beta
        
    def __str__(self):
        return '{beta} 1_[{a}, {b}[ (x)'.format(a = self.a, b = self.b, beta=self.beta)
    
    def value(self, rho):
        return self.beta * ind(rho, self.a, self.b)