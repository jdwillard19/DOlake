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
obs_pt = ['o2_hyp']
obs_name = ['obs_hyp']
pt_fields = dyn_feats + obs_pt
trn_test_fields = dyn_feats + obs_name
all_feats = dyn_feats + stat_feats
win_shift = 30
seq_len = 60


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
	print("processing site ",i,"/",len(site_ids),": ",site_id)
	site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")
	feats = site_df[all_feats]
	dates = site_df['datetime']
	# site_id = site_df['site_id']
	if not os.path.exists("../../data/processed/"+site_id):
		os.mkdir("../../data/processed/"+site_id)

	#create one df per statification period > 90 days
	current_window_length = 0
	strat_period_list = []
	temp_df = pd.DataFrame()
	for j in range(site_df.shape[0]):
		if j % 100 == 0:
			print("day ",j,"/",site_df.shape[0])
		if site_df.iloc[j]['strat'] == 0:
			if current_window_length > 90:
				#save window
				temp_df = pd.DataFrame(site_df.iloc[j-current_window_length+1:j].values,columns = site_df.iloc[j].index)
				strat_period_list.append(temp_df)
				print("saved stratification period of length ",current_window_length)
			#reset df
			del temp_df
			temp_df = pd.DataFrame()
			current_window_length = 0
		else:
			current_window_length += 1
			
	#create sliding windows for pre-train,train, and test
	pt_data = np.empty((0,seq_len,len(pt_fields)))
	trn_data = np.empty((0,seq_len,len(trn_test_fields)))
	tst_data = np.empty((0,seq_len,len(trn_test_fields)))

	#for each strat period, append to data matrices
	for strat_period in strat_period_list:
		start_ind = 0
		end_ind = start_ind + seq_len
		while end_ind < strat_period.shape[0]:
			#append to pt data
			pdb.set_trace()

			#append to train data

			#append to tst data

			start_ind += seq_len
			end_ind += seq_len



	# total_df.to_feather("./temp/all_site_feats.feather")
