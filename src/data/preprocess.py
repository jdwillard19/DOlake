import numpy as np
import pandas as pd
import pdb
import os


#get list of lakes
raw_data_dir = '../../data/raw/DOzip/'
site_ids = np.load("../../metadata/201site_ids.npy",allow_pickle=True)
dyn_feats = ["thermocline_depth","temperature_epi","temperature_hypo",\
			  "volume_epi","volume_hypo","td_area","wind","airtemp",\
			  "fnep","fmineral","fsed","fatm","fentr_epi","fentr_hyp",\
			  "o2_epi"]
stat_feats = ["area_surface","max.d"]
all_feats = dyn_feats + stat_feats

 
#get features and calc stats\
total_df = pd.DataFrame(columns=all_feats)

for i,site_id in enumerate(site_ids):
	print("site ",i,"/",len(site_ids))
	if os.path.exists(raw_data_dir+site_id+"/"+site_id+".feather"):
		site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")
		site_df['site_id'] = site_id
		total_df = pd.concat([total_df,site_df])
	else:
		print("no file?")
		continue

pdb.set_trace()