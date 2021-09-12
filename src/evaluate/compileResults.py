import numpy as np
import pandas as pd
import pdb
site_ids = np.load("../../metadata/201site_ids.npy",allow_pickle=True)
raw_data_dir = '../../data/raw/DOzip/'
process_rmses = []

def rmse(pred,targ):
	return np.sqrt(((pred - targ)**2).mean())
for ct, site_id in enumerate(site_ids):
	print(ct,"/",len(site_ids), " sites: ",site_id)
	site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")

	#filter to test obs
	site_df_test = site_df[(site_df['splitsample']==1) & (pd.notnull(site_df['obs_hyp']))]

	site_rmse = rmse(site_df_test['obs_hyp'], site_df_test['o2_hyp'])

	if np.isnan(site_rmse):
		print("nan?")
		pdb.set_trace()
	process_rmses.append(site_rmse)

pdb.set_trace()