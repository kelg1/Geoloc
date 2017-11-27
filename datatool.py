import pandas as pd
import numpy as np 
import sys



class DataTool:
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

