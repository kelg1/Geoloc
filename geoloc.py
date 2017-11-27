import pandas as pd 
import numpy as np 
import scipy
import scipy.optimize
import utm
import geopy
from geopy import distance


class GeoLoc:
    def __init__(self, dict_models, df_latlong=None):
        self.dict_models = dict_models 
        self.bs_latlong = bs_latlong
        
    def getGeoLocWrapper(self, serie):
        J = defaultdict(list)
        for bs, r in serie.items():
            if bs in set(basestations.objid):
                if ~np.isnan(r):
                    #print(bs, r)
                    model_bs = self.dict_models.get(bs)
                    wrap = partial(self.wrapper, bs)
                    #print(wrap([2.2, 48.9]))
                    #print(model_bs.predict(wrap([2.2, 48.9])))
                    #print(scipy.stats.norm.logpdf(r, loc=model_bs.predict(wrap([2.2, 48.9])),
                    #                        scale=model_bs.scale))
                    J.update({bs: [model_bs, wrap, r]})
        def f_to_minimize(z):
            return -np.sum([(model._loglikelihood_wrapper(part(z),
                                                  rssi)) for bsid,
                    (model, part, rssi) in J.items()])
        res = scipy.optimize.minimize(f_to_minimize, [2, 48], tol=1e-8,
                                     method='Nelder-Mead').x
        return res


    
    def wrapper(self, bs, z):
        res = {'angle': angle(self.bs_latlong[bs]['long'],
                             self.bs_latlong[bs]['lat'],
                              *z),
              'distance': vincenty_(self.bs_latlong[bs]['long'],
                                   self.bs_latlong[bs]['lat'], 
                                   *z)
              }
        return pd.DataFrame(res, index=[0])
    
    
    
    def _plus_(f,g):
    
        def h(x):
            return f(x)+g(x)
        return h

## Compute angle between to lat long points ##

def vincenty_(lon1, lat1, lon2, lat2):
    return distance.vincenty((lon1, lat1), (lon2, lat2)).km


def angle(lon1, lat1, lon2, lat2):
    E1, N1, _, _ = utm.from_latlon(lat1, lon1)
    E2, N2, _, _ = utm.from_latlon(lat2, lon2)
    dE, dN = E2 - E1, N2 - N1
    return np.arctan2(dE, dN) % (2*np.pi)

    


def compute_angle(df, bs_latlong):
    return df[['bsid', 'latitude', 'longitude']].apply(lambda row: angle(bs_latlong[row['bsid']]['lat'],
                                                              bs_latlong[row['bsid']]['long'],
                                                            row['latitude'], row['longitude'],
                                                              ), 1)



def compute_distance(df, bs_latlong):
    return df[['bsid', 'latitude', 'longitude']].apply(lambda row: 
                                                             vincenty_(row['longitude'],
                                                                       row['latitude'],
                                                                       bs_latlong[row['bsid']]['long'],
                                                                       bs_latlong[row['bsid']]['lat']), 1)
