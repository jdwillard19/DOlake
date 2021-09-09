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
obs_pt = ['o2_hypo']
obs = ['obs_hypo']
all_feats = dyn_feats + stat_feats
win_shift = 30
# all_dyn_feats_nonan_og = ['wind', 'airtemp', 'fnep', 'fmineral', 'fsed', 'fatm', 'o2_epi']
# all_dyn_feats_nonan = ['wind', 'airtemp', 'fnep', 'fmineral', 'fsed', 'fatm', 'o2_epi',\
# 					   'strat','temperature_total'
# 					   ''
#replace nan temperature_hypo with temperature_total
#replace nan volume_hype with volume_total
#replace nan thermocline depth with zero
#replace nan td_area with surface area
# all_feats_nonan = all_dyn_feats_nonan + ['area_surface','max.d']
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
	total_feat_df = total_df[all_feats]
	total_feat_df = total_feat_df.fillna(value=np.nan)
	mean_feats = []
	std_feats = []
	for i in range(total_feat_df.shape[1]):
		print("calc stats feat ",i)
		# if i==15:
		# 	pdb.set_trace()
		mean_feats.append(np.nanmean(total_feat_df.iloc[:,i],axis=0))
		std_feats.append(np.nanstd(total_feat_df.iloc[:,i],axis=0))
	np.save("temp/mean_feats",np.array(mean_feats))
	np.save("temp/std_feats",np.array(std_feats))
else:
	total_df = pd.read_feather("./temp/all_site_feats.feather")
	total_df = total_df.fillna(value=np.nan)
	total_feat_df = total_df.drop(['date','datetime','site_id'],axis=1)

#code to check number of stratification sequences
# min_seq = 999
# ct = 0
# min_ind = None
# seqs = []
# for ind,i in enumerate(total_df['strat'].values):
# 	if i == 1:
# 		ct += 1
# 	else:
# 		if ct < min_seq and ct != 0:
# 			min_seq = ct
# 			min_ind = ind
# 		elif ct != 0:
# 			seqs.append(ct)

# 		ct = 0
# print("min seq: ",min_seq)
# print("min ind: ",min_ind)

mean_feats = np.array(mean_feats)
std_feats = np.array(std_feats)


#preprocess per site

if not os.path.exists("../../data/processed"):
	os.mkdir("../../data/processed")

for i,site_id in enumerate(site_ids):
	site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")
	feats = site_df[all_feats]
	dates = site_df['datetime']
	# site_id = site_df['site_id']
	if not os.path.exists("../../data/processed/"+site_id):
		os.mkdir("../../data/processed/"+site_id)
n
	#create one df per statification period > 90 days
	current_window_length = 0
	strat_period_list = []
	temp_df = pd.DataFrame()
	pdb.set_trace()
	for j in range(site_df.shape[0]):
		if site_df.iloc[j]['strat'] == 0:
			if current_window_length > 90:
				#save window
				strat_period_list.append(temp_df.copy())
			#reset df
			del temp_df
			temp_df = pd.DataFrame()
			current_window_length = 0
		else:
			current_window_length += 1
			temp_df = pd.concat([temp_df,site_df.iloc[j,:]])



	# total_df.to_feather("./temp/all_site_feats.feather")
