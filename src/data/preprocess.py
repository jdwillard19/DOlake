import numpy as np
import pandas as pd
import pdb
import os


#get list of lakes
raw_data_dir = '../../data/raw/DOzip/'
site_ids = np.load("../../metadata/201site_ids.npy",allow_pickle=True)
dyn_feats = ["thermocline_depth","temperature_epi","temperature_hypo",\
			  "volume_epi","volume_hypo","td_area","wind","airtemp",\
			  "fnep","fmineral","fsed","fatm","fentr_epi",\
			  "o2_epi"]
			  # fentr_hyp all none?
stat_feats = ["area_surface","max.d"]
all_feats = dyn_feats + stat_feats

 
#get features and calc stats\
total_df = pd.DataFrame(columns=all_feats)
hardcode = False
if not hardcode:
	if not os.path.exists("./temp/all_site_feats.feather"):
		for i,site_id in enumerate(site_ids):
			print("pre: site ",i,"/",len(site_ids))
			if os.path.exists(raw_data_dir+site_id+"/"+site_id+".feather"):
				site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")
				site_df['site_id'] = site_id
				total_df = pd.concat([total_df,site_df])
			else:
				print("no file?")
				continue
			total_df.to_feather("./temp/all_site_feats.feather")
	else:
		total_df = pd.read_feather("./temp/all_site_feats.feather")

	total_feat_df = total_df.drop(['date','datetime','site_id','fentr_hyp','index'],axis=1)
	total_feat_df = total_feat_df.fillna(value=np.nan)
	mean_feats = []
	std_feats = []
	for i in range(total_feat_df.shape[1]):
		print("feat ",i)
		mean_feats.append(np.nanmean(total_feat_df.iloc[:,i],axis=0))
		std_feats.append(np.nanstd(total_feat_df.iloc[:,i],axis=0))
	np.save("temp/mean_feats",np.array(mean_feats))
	np.save("temp/std_feats",np.array(mean_feats))
else:
	total_df = pd.read_feather("./temp/all_site_feats.feather")
	total_df = total_df.fillna(value=np.nan)
	total_feat_df = total_df.drop(['date','datetime','site_id'],axis=1)


mean_feats = np.array(mean_feats)
std_feats = np.array(std_feats)



pdb.set_trace()
for i,site_id in enumerate(site_ids):
	site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")
	feats = site_df[dyn_feats]
	dates = site_df['datetime']
	site_id = site_df['site_id']
	pdb.set_trace()
	# total_df.to_feather("./temp/all_site_feats.feather")
