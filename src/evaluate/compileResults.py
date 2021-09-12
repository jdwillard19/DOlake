import numpy as np
import pandas as pd
import pdb
site_ids = np.load("../../metadata/201site_ids.npy",allow_pickle=True)
raw_data_dir = '../../data/raw/DOzip/'
process_rmses = []

def rmse(pred,targ):
	return np.sqrt(((pred - targ)**2).mean())
for site_id in site_ids:
	site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")

	#filter to test obs
	site_df = site_df[site_df['splitsample']==1]
	pdb.set_trace()
