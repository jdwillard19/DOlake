import numpy as np
import pandas as pd
import pdb
import os


#get list of lakes
raw_data_dir = '../../data/raw/DOzip/'
pdb.set_trace()
site_ids = [x[1] for x in os.walk(raw_data_dir)][0]

dyn_feats = ["thermocline_depth","temperature_epi","temperature_hypo",\
			  "volume_epi","volume_hypo","td_area","wind","airtemp",\
			  "fnep","fmineral","fsed","fatm","fentr_epi","fentr_hyp",\
			  "o2_epi"]
stat_feats = ["area_surface","max.d"]

 
#get features and calc stats\
total_df = pd.DataFrame(columns=dyn_feats)

for site_id in site_ids:
	site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")
	pdb.set_trace()