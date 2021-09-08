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

if not os.path.exists("./temp/all_site_feats.feather"):
	for i,site_id in enumerate(site_ids):
		print("site ",i,"/",len(site_ids))
		if os.path.exists(raw_data_dir+site_id+"/"+site_id+".feather"):
			site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")
			site_df['site_id'] = site_id
			total_df = pd.concat([total_df,site_df])
		else:
			print("no file?")
			continue
else:
	total_df = pd.read_feather("./temp/all_site_feats.feather")

total_df = total_df.drop(['date','datetime'],axis=1)
mean_feats = []
std_feats = []
for i in range(total_df.shape[1]):
	print("feat ",i)
	mean_feats.append(np.nanmean(total_df.iloc[:,i],axis=0))
	std_feats.append(np.nanstd(total_df.iloc[:,i],axis=0))

pdb.set_trace()