import numpy as np
import pandas as pd
import pdb
import os


#get list of lakes
raw_data_dir = '../../data/raw/DOzip/'
site_ids = [x[1] for x in os.walk(raw_data_dir)][0]
dull_lakes = ['nhdhr_120018013','nhdhr_121622916','nhdhr_120020396','nhdhr_121622760']
dyn_feats = ["thermocline_depth","temperature_epi","temperature_hypo",\
			  "volume_epi","volume_hypo","td_area","wind","airtemp",\
			  "fnep","fmineral","fsed","fatm","fentr_epi","fentr_hyp",\
			  "o2_epi"]
# stat_feats = ["area_surface","max.d"]

 
#get features and calc stats\
total_df = pd.DataFrame(columns=dyn_feats)

for i,site_id in enumerate(site_ids):
	print("site ",i,"/",len(site_ids))
	if os.path.exists(raw_data_dir+site_id+"/"+site_id+".feather"):
		site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")
		total_df = pd.concat([total_df,site_df])
	else:
		print("no file?")
		continue

pdb.set_trace()