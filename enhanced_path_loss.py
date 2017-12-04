import numpy as np 
import sys
import pandas as pd
import warnings
import functools
import sklearn
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from collections import defaultdict
from functools import partial
from base_dico import SIN as SIN
from base_dico import COS as COS
from base_dico import IND as IND
from base_dico import ind as ind
import prox_tv as ptv
import time
import scipy
import scipy.stats






###########
## Utile ##
###########

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning) #turn off filter 
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning) #reset filter
        return func(*args, **kwargs)

    return new_func 


def matrix_fill_cos(val, i, j):
    return np.cos(np.around(val.angle.iloc[i], 5)*j)

def matrix_fill_sin(val, i, j):
    return np.sin(np.around(val.angle.iloc[i], 5)*j)


#################################
#######------------------########
#################################


class Model:
    """ 
    Generic class for models. 
    q(theta_k(z), "r_k") = P_\theta(Rk \in dr | Z = z)  = N(f(dist(z, zk))g(arg(z, zk)), sigma2)
    The purpose of the class is to enable us to infer the parameter
    \theta_k in a more friendly manner.
    As refered in the RSSI based geolocation, the suboptimal 
    estimator of theta is:
    
    theta_hat = arg min - sum ln(q(theta(z_i), r_i)) + lambda * regularization(theta)
    
    this is perfored by "self.infer_model" method.

    """
    
    n_iters=50
    
    def __init__(self):
        pass
        #self.params = params
        #self.success = False
        
    
    #def _get_params(self):
    #    return self.__dict__.get('params')
    
    @property
    def regularization(self):
        return self._regularization
    
    @regularization.setter
    def regularization(self, pen):
        self._regularization = pen
        
    def _objective_function(self, X_test, y_test=None):
        raise NotImplementedError
        
        
    def fit(self, X, y=None, niters=None, cond_alpha=1e-2, cond_beta=1e-2):
        
        y0 = y.copy()
        y_ = y0.copy()
        self.error_through_iterations_ = []
        sys.stdout.write('\r mu_alpha = {mu_alpha}, mu_l1 = {mu_l1}, mu_l2= {mu_l2} \n'.format(mu_alpha=self.mu_alpha,
                                                                               mu_l1=self.mu_l1, mu_l2=self.mu_l2))
        N_ITER = niters or self.n_iters
        for iter_ in range(N_ITER):
            sys.stdout.write('\r iteration: {iter_} '.format(iter_=iter_))
            
            #Psi step:
            #---------
            beta_iter = self.beta_.copy()
            self._infer_Psi_PTV(X, y_)
            cond_beta = np.linalg.norm(beta_iter - self.beta_)/np.linalg.norm(self.beta_) <= cond_beta
            #Psi step:
            #---------
            y_ = np.divide(y, self.Psi(X.distance.values))
            #---------
            alpha_iter = self.alpha_.copy()
            self._infer_Phi(X, y_)
            cond_alpha = np.linalg.norm(alpha_iter - self.alpha_)/np.linalg.norm(self.alpha_) <= cond_alpha

            y_ = np.divide(y, self.Phi(X.angle))
            
            self.error_through_iterations_.append(np.around(self.score(X, y), 6))
            
            if cond_alpha & cond_beta:
                break
            #sys.stdout.write("\r \t Error 'l2' at iteration {i}: {e} \n".format(i=iter_,
            #                                                                        e=np.around(self.score(X, y), 4),
            #                                                                       )
            #                    )
            
        self.error_through_iterations = self.error_through_iterations_
        success=True 
        self.alpha = self.alpha_
        self.beta = self.beta_
        self.scale = np.sqrt(np.mean((y - self.predict(X))**2))
        #self.success = success
        return self

    
    def predict(self, X):
        return self.Psi(X.distance.values) * self.Phi(X.angle)
        
    
    def score(self, X, y):
        #scorer = metrics.make_scorer(np.linalg.norm, greater_is_better=False)
        y_pred = self.predict(X)
        y_true = y.copy()
        error_on_data = np.sqrt(((y_true - y_pred) ** 2).mean())
        #regularization = self.mu_l1 * np.abs(self.beta_[1:-1] - self.beta_[2:]).mean()
        return - error_on_data 

    def max_error(self, X, y):
        #scorer = metrics.make_scorer(np.linalg.norm, greater_is_better=False)

        y_pred = self.predict(X)
        y_true = y.copy()
        error_on_data = np.sqrt(((y_true - y_pred) ** 2).mean())
        #regularization = self.mu_l1 * np.abs(self.beta_[1:-1] - self.beta_[2:]).mean()
        return - error_on_data 

    def _loglikelihood_wrapper(self, X, y):
        """
        Parameters:
        -----------
        X : [RSSI, lat, long] DataFrame
        observation (z_k, r_k) of (Z, R) 
        
        Returns:
        --------
        likelihood : scalar
        log q(theta(z_k), r_k)
        
        """
        
        loglikelihood = scipy.stats.norm.logpdf(y, loc=self.predict(X), scale=self.scale)
        return loglikelihood
    

class EnhancedPathLoss(Model):
    """
    Enhanced Path Loss model as refered
    in RSSI based geolocation doc
    """

    DEFAULT_Kr = 50
    DEFAULT_Kt = 50
    
    DEFAULT_alpha = np.zeros((2*DEFAULT_Kt+1))
    DEFAULT_alpha[0] = 1
    
    DEFAULT_beta = np.zeros((DEFAULT_Kr+1))
    DEFAULT_beta[0] = -80
    
    DEFAULT_grid_params = {'mu_alpha': np.arange(0, 1, .01),
                          'mu_l2': np.linspace(1e-3, 1e-1, 5)}
    DEFAULT_h_rho = np.linspace(0, 30, DEFAULT_Kr+1)

    
    def __init__(self, params=defaultdict(list), mu_alpha=1e-3, 
                 Kr=50, Kt=25, alpha_=None, beta_=None,
                 mu_l2=0.003, 
                 mu_l1=3):
        #self.type = 'Enhanced Pathloss model'
        self.params = params
        self.mu_alpha = mu_alpha
        self.Kr = Kr
        self.Kt = Kt
        self.alpha_ = np.ones((2*self.Kt+1))
        self.beta_ = -200*np.ones(self.Kr)
        self.h_rho = np.linspace(0, 30, self.Kr+1)
        self.mu_l2 = mu_l2
        self.mu_l1 = mu_l1
        
    def __str__(self):
        return """
        Enhanced Path Loss model. 
        q(theta_k(z), "r_k") = P_\theta(Rk \in dr | Z = z)  = N(f(dist(z, zk))g(arg(z, zk)), sigma2)
    
        As refered in the RSSI based geolocation, the suboptimal 
        estimator of theta is:
    
        theta_hat = arg min - sum ln(q(theta(z_i), r_i)) + ({mu_alpha} || phi ||_2  + {mu_l1}|| d Psi ||_1)
    
        """.format(mu_alpha=str(self.mu_alpha), mu_l1=str(self.mu_l1))
    
    def get_params(self, deep=True):
        return {"params": self.params,
                "mu_alpha": self.mu_alpha,
                "Kr": self.Kr,
                "Kt": self.Kt,
                "alpha_": self.alpha_,
                "beta_": self.beta_,
                "mu_l2": self.mu_l2,
                "mu_l1": self.mu_l1}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    @property
    def psi(self):
        return self._psi
   
    @psi.setter
    def psi(self, psi):
        self._psi=psi
        self.params['psi'] = psi.__code__.co_varnames[1:]
        raise Warning('This functionality is not implemented yet')
        
    @property
    def phi(self):
        return self._phi
    
    @phi.setter
    def phi(self, phi):
        self._phi=phi
        self.params['phi'] = phi.__code__.co_varnames[1:]
        raise Warning('This functionality is not implemented yet')
        
    
    def functions_dico_t(self):
        try :
            functions_a0 = np.array([COS(0, self.alpha_[0])])
        except IndexError:
            functions_a0 = np.array([COS(0, self.alpha_)])
        functions_dico_cos = np.array([COS(k, self.alpha_[k]) for k in np.arange(1, self.Kt)])
        functions_dico_sin = np.array([SIN(k, self.alpha_[k+self.Kt]) for k in np.arange(1, self.Kt)])
        functions_dico_t = np.hstack((functions_a0, functions_dico_cos, functions_dico_sin))
        return functions_dico_t
    
        
    def _infer_Phi(self, X, y=None):
        if 'phi' in self.params:
            raise NotImplementedError
        else:
            t0 = time.clock()
            reg = linear_model.ElasticNet(alpha=self.mu_alpha, l1_ratio=1., fit_intercept=False)
            #sys.stdout.write('\r'+ 'Function Phi is fitted with: ' +reg.__str__()+ '\n')
            self.build_matrix_Xt(X)
            reg.fit(self.matrix_Xt, y)
            self.alpha_ = np.copy(reg.coef_)
        #sys.stdout.write('\r \t'+'Function Phi is fitted ({t} s)\n'.format(t=np.round(time.clock()-t0, 2)))

    def F(self, y, beta):
        return .5 * np.linalg.norm(y - np.dot(self.matrix_Xr, beta))**2 + self.mu_l2 * np.linalg.norm(beta)**2

    def grad_F(self, y, beta):
        return  - np.dot(self.matrix_Xr.T, (y - np.dot(self.matrix_Xr, beta))) + 2*self.mu_l2*beta
        
    def _infer_Psi_PTV(self, X, y=None):
        self.build_matrix_Xr(X)
        max_ev = np.max(np.linalg.eigvals(np.dot(self.matrix_Xr.T, self.matrix_Xr)))
        proxtv = ADMM(partial(self.F, y), partial(self.grad_F, y), weights=float(self.mu_l1), gamma=float(.9/(self.mu_l2 + max_ev)))
    
        beta0 = -200*np.ones((self.Kr))
        beta_PTV = proxtv.run_algo(x0=beta0, epsi=1e-1)
        self.beta_ = beta_PTV
    
    def Phi(self, theta):
        return np.sum([f.value(theta) for f in self.functions_dico_t()], 0)
    
    @deprecated
    def _infer_Psi(self, X, y=None):
        if 'psi' in self.params:
            raise NotImplementedError
        else:
            t0 = time.clock()
            bds = tuple([(-1. * self.mu_beta, self.mu_beta) for i in range(self.Kr+1)])
            reg_ = linear_model.LinearRegression(fit_intercept=False)
            #sys.stdout.write('\r'+ 'Function Psi is fitted with: ' +reg_.__str__()+ '\n')
            self.build_matrix_Xr(X)
            D = - np.diag(np.ones(self.Kr), 1) + np.eye(self.Kr+1)
            D[0,0], D[0,1], D[-1, -1] = 0, 0, 0
            DtiXt = (np.dot(D, np.linalg.pinv(self.matrix_Xr))).T
            XtX = np.dot(self.matrix_Xr.T, self.matrix_Xr)
            iXtX = np.linalg.pinv(XtX)
            _u = reg_.fit(DtiXt, y)
            u_reg = np.copy(_u.coef_)
            u = self.proj(u_reg, - self.mu_beta, self.mu_beta)
            beta_ = np.dot(self.matrix_Xr.T, y) - np.dot(D.T, u)
            self.beta_ = np.dot(iXtX, beta_)
            
    def objective_function(self, X, y):
        return .5*((y - self.predict(X))**2).sum()
    
    def sqrt_error(self, X, y):
        return (y - self.predict(X))
    
    def grad_J_alpha_(self, X, y):
        self.build_matrix_Xt(X)
        g = -( np.dot(self.alpha_, self.matrix_Xt.T).shape * self.sqrt_error(X, y) )[:,np.newaxis] \
            * self.matrix_Xr
        return g.sum(0) 
        
    def grad_J_beta_(self, X, y):
        self.build_matrix_Xr(X)
        g = -( np.dot(self.beta_, self.matrix_Xr.T).shape * self.sqrt_error(X, y) )[:,np.newaxis] \
            * self.matrix_Xt
        return g.sum(0)
    
    
    def grad_J_theta(self, X, y):
        return np.concatenate((self.grad_J_alpha_(X, y), self.grad_J_beta_(X, y)))
    
    def join_fit(self, X, y):
        #(y - self.predict(X))
        self.build_matrix_Xr(X)
        self.build_matrix_Xt(X)
        def J(theta):
            alpha = theta[:2*self.Kt+1]
            beta = theta[2*self.Kt+1:]
            model_ = copy.copy(self)
            model_.alpha_ = alpha
            model_.beta_ = beta
            return model_.objective_function(X, y)

        def gradJ(theta):
            alpha = theta[:2*self.Kt+1]
            beta = theta[2*self.Kt+1:]
            model_ = copy.copy(self)
            model_.alpha_ = alpha
            model_.beta_ = beta
            return model_.grad_J_theta(X, y)
        
        res = scipy.optimize.minimize(fun=J, x0=np.concatenate((self.alpha_, 
                                                  self.beta_ )),
                        jac=gradJ, method='Nelder-Mead', callback=J, options={'fatol': 10})
        self.alpha_, self.beta_ = res.x[:2*self.Kt+1], res.x[2*self.Kt+1:]
        return self
        
    def Psi(self, rho):
        def wrapper(r):
            if self.h_rho[-1] > r:
                j = np.min(np.where(r < self.h_rho)[0])
                return self.beta_[j - 1]
            else:
                return self.beta_[-1]
        return np.array([wrapper(r) for r in rho])

    
    def build_matrix_Xt(self, X):
        if not hasattr(self, 'matrix_Xt'):
            matrix_X_cos = np.zeros((X.shape[0], self.Kt+1))
            matrix_X_sin = np.zeros((X.shape[0], self.Kt))
            matrix_X_cos[:,0] = 1

            ### FILL matrix X ### 
            if self.Kt > 0:
                matrix_X_cos[:,1:] = np.cos(np.column_stack([X.angle*j 
                    for j in np.arange(1, self.Kt+1)]))

                matrix_X_sin = np.sin(np.column_stack([X.angle*j
                    for j in np.arange(1, self.Kt+1)]))

            matrix_Xt = np.column_stack((matrix_X_cos, matrix_X_sin))
                #sys.stdout.write('\r \t'+'Matrix Xtheta is built. \n')
            self.matrix_Xt = matrix_Xt
        
    
    def functions_dico_r(self):
        return np.array([IND(self.h_rho[k-1],self.h_rho[k], self.beta_[k]) for k in np.arange(1, self.Kr)])
    
    def build_matrix_Xr(self, X):
        #print(not hasattr(self, 'matrix_Xr'))
        if not hasattr(self, 'matrix_Xr'):
            matrix_Xr = np.zeros((X.shape[0], self.Kr))
            

            ### FILL matrix Xr ### 

            for i in np.arange(0, matrix_Xr.shape[0]):
                j = np.min(np.where(X.distance.iloc[i] < self.h_rho)[0])
                matrix_Xr[i,j-1] = 1
            #print(matrix_Xr.sum(0))
            self.matrix_Xr = matrix_Xr
            #sys.stdout.write('\r \t'+'Matrix Xrho is built. \n')    
    
    def hyperparameters_tuner(self, X, y, params={'mu_alpha', 'mu_l2', 'mu_l1'}):
        param_grid = {p: self.DEFAULT_grid_params.get(p) for p \
                     in params}
        
        gscv = GridSearchCV(self, param_grid)
        gscv.fit(X, y)
        return gscv.best_estimator_
    
    @staticmethod   
    def SplitbyDid(X, y, n_splits=5): 
        """
        Parameters:
        -----------
        
        X: DataFrame
        
        y: Serie 
        
        Return:
        -------
        generator
        """
        group_kfold = GroupKFold(n_splits=n_splits)
        groups = X.did.values
        return group_kfold.split(X, y, X.did)

    @deprecated
    def proj(vector, lb, ub):
        return np.minimum(ub, np.maximum(vector, lb))

        
        
            
class ADMM:
    def __init__(self, F, grad_F, gamma=None, weights=None):
        self.F = F
        #self.R = R
        self.grad_F = grad_F
        self.gamma = gamma 
        self.weights = weights
        
    def gamma_(self, F):
        raise NotImplementedError
        
    def forward_step(self, current_state):
        y = current_state - self.gamma * self.grad_F(current_state)
        return y
        
    def backward_step(self, current_state):
        #x = ptv.tv1_1d(current_state, self.gamma, method='hybridtautstring')
        #w = np.concatenate(([0], [self.gamma]*(len(current_state)-2)))
        x = ptv.tv1_1d(current_state, w=float(self.weights*self.gamma), method='hybridtautstring',)
        return x
    
    def run_algo(self, x0, epsi=1e-2, n_iterations=100):
        x_ = x0.copy()
        for iteration in range(n_iterations):
            y = self.forward_step(x_)
            x__ = self.backward_step(y)
            #print(self.F(x__))
            if np.linalg.norm(x__ - x_)/np.linalg.norm(x_) <= epsi:
                return x__
            else:
                x_ = x__.copy()
            #print(x__)
        #did not converge
        return x__


class util:
        """
    In order to convert DataFrame to an other in which each row (msg)
    give the rssi value to each bs

    """

    def __init__(self):
        pass

    def describe(self, df):
        st = """Number of BS:\t {nbs}\t \n
========================== \n
Number of msg:\t {nmsg}\t \n 
========================== \n""".format(nbs=df.bsid.nunique(),
                                       nmsg=df.messageid.nunique())
        return st

    def groupbymsg(self, df):
        self.dd_tmp = pd.DataFrame()
        df_msg = df.groupby('messageid')
        for g, v in df_msg:
            tmp = pd.DataFrame(v.groupby('bsid').rssi.max()[:,np.newaxis].T, index=[g], columns=v.bsid.unique())
            tmp['latitude'] = v.latitude.unique()
            tmp['longitude'] = v.longitude.unique()
            self.dd_tmp = self.dd_tmp.append(tmp)
            sys.stdout.write('\r'+str(np.round(100*len(self.dd_tmp)/len(df_msg), 0)) + ' % ' + '(' + \
                            str(len(self.dd_tmp)) +'/'+ str(len(df_msg)) + ')')
        return self.dd_tmp
