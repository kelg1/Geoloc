import numpy as np 
import scipy
from scipy import stats
from collections import defaultdict
import pandas as pd 


class KernelRegression:
    """
    Nadaraya-Watson kernel regression
    
    See also
    --------
    
    """

    def __init__(self, kernel=None, h_=None, ordre=0, kind='tophat'):
        self.kernel = kernel
        self.h_ = h_
        self.ordre = ordre
        self.kind = kind
        self.lru_cache = defaultdict(list)
    def lru_cache_(self, d, t):
        raise NotImplentedError
        
        
    def U(self, rhop, thetap):
        test = defaultdict(dict)
        test[0]
        test.get(0).update({(0,0): 1})
        for order in range(1, self.ordre+1):
            for k in range(order+1):
                l = order - k
                #print(order, k, l)
                val = np.math.factorial(k)*np.math.factorial(l)
                test[order]
                test.get(order).update({(k,l): rhop**k*thetap**l/(1.*val)})
                test.get(order).update({(l,k): rhop**l*thetap**k/(1.*val)}) 
        self.memo_U = test
        res = np.array([j for j in np.concatenate([p.values() for
                                            p in test.values()])])[:,np.newaxis]
        return res
    
    def K(self, x, y):
        # Kernel séparable en theta et rho #
        if self.kind == 'tophat':
            return .5 * (np.abs(x) <= 1) * .5 * (np.abs(y) <= 1)
        if self.kind == 'gaussian':
            return scipy.stats.multivariate_normal.pdf([x, y], mean=[0, 0])
        else:
            raise NotImplementedError
    
    @property
    def h(self):
        #assert isinstance(self.h_, dict), 'h must be a dict'
        return self.h_
    
    @h.setter
    def h(self, value):
        if not isinstance(value, dict): 
            raise TypeError("""KernelRegression.h must be 
            a dict {h_rho: val1, h_theta: val2}""")
        else:
            self.h_ = value
        
    def optimized_h(self, X, y, force=False):
        if (not hasattr(self.h_, '__iter__')) or (force==True):
            n=len(X)
            sig_R_wrt_theta = np.nanmax([y.iloc[indices].std() 
                     for indices in X.groupby('angle').indices.values()])
            sig_R_wrt_rho = np.nanmax([y.iloc[indices].std() 
                     for indices in X.groupby('distance').indices.values()])
            sigs = {'s_rho': sig_R_wrt_rho ,'s_theta': sig_R_wrt_theta }
            self.h_ = {'h_rho': sigs['s_rho']**(2./5)*n**(-1./5), 
                      'h_theta': sigs['s_theta']**(2./5)*n**(-1./5)}
    
    def B_nx(self, X, y=None):
        n = len(X)
        def B_nx_wrapper(d, alpha):
            rho_centred = (X['distance'] - d)/self.h.get('h_rho')
            theta_centred = (X['angle'] -  alpha)/self.h.get('h_theta')
            B=0
            for r, t in zip(rho_centred.values, theta_centred.values):
                u = self.U(r,t)
                B += (u.T * u) * self.K(r, t)
            return B
        self.B = B_nx_wrapper
        
    
    def a_nx(self, X, y):
        n = len(X)
        #self.optimized_h(X, y)

        def a_wrapper(d, alpha):
            rho_centred = (X['distance'] - d)/(1.*self.h.get('h_rho'))
            theta_centred = (X['angle'] -  alpha)/(1.*self.h.get('h_theta'))
            a=0
            for loc, v in pd.concat((rho_centred,
                                     theta_centred), 1).iterrows():
                r, t = v.distance, v.angle
                u = self.U(r, t)
                a += u.T * self.K(r, t) * y.loc[loc] 
            return a
        self.a = a_wrapper
    
    def fit(self, X, y):
        self.B_nx(X, y)
        self.a_nx(X, y)
    
    def predict(self, d, t):
        if self.lru_cache.get((d, t)) is None:
            F = np.linalg.lstsq(a=self.B(d, t),
                    b=self.a(d, t).ravel())
            res = F[0][0]
            self.lru_cache[(d, t)] = res
        else:
            res = self.lru_cache.get((d, t))
        return res 
        
    @deprecated
    def Wni(self, X, y):
        def Wni_wrapper(d, alpha):
            n = len(X)
            rho_centred = (X['distance'] - d)/self.h.get('h_rho')
            theta_centred = (X['angle'] -  alpha)/self.h.get('h_theta')
            B_nx_ = self.B_nx(X)(d, alpha) 
            #try:
            #    self.HPL(X, d, alpha)
                 
            #except AssertionError:
            #    B_nx_ += 2*np.abs(np.linalg.eigvals(B_nx_).min())*np.eye(len(B_nx_))
            
            inv_B = np.linalg.pinv(B_nx_)
            UT0 = self.U(0, 0).T
            UTBi = np.dot(UT0, inv_B)
            w = 0
            for loc, v in X.iterrows(): 
                r, t = v.distance, v.angle
                u = self.U(r, t)
                wni = np.dot(UTBi, u)
                wni *= self.K(r, t)
                w += wni*y.loc[loc]
            return w
        return Wni_wrapper
        
            
    def HPL(self, d, alpha):
        emin = np.min(np.linalg.eigvals(self.B(d, alpha))) 
        assert emin > 0, '(alpha={a}, distance={d}) B_nx must be > 0 (but = {e})'.format(a=alpha, d=d, e=emin)
        
        