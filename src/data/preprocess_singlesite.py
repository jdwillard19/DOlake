import numpy as np
import pandas as pd
import pdb
import os
import sys

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
				pdb.set_trace()
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


site_id = sys.argv[1]
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
	# if j % 100 == 0:
	# 	print("day ",j,"/",site_df.shape[0])
	if site_df.iloc[j]['strat'] == 0:
		if current_window_length > 90:
			#save window
			temp_df = pd.DataFrame(site_df.iloc[j-current_window_length+1:j].values,columns = site_df.iloc[j].index)
			strat_period_list.append(temp_df)
			# print("saved stratification period of length ",current_window_length)
		#reset df
		del temp_df
		temp_df = pd.DataFrame()
		current_window_length = 0
	else:
		current_window_length += 1
		
#create sliding windows for pre-train,train, and test
pt_data = np.empty((0,seq_len,len(pt_fields)))
pt_dates = np.empty((0,seq_len),dtype=np.datetime64)
trn_data = np.empty((0,seq_len,len(trn_test_fields)))
trn_dates = np.empty((0,seq_len),dtype=np.datetime64)
tst_data = np.empty((0,seq_len,len(trn_test_fields)))
tst_dates = np.empty((0,seq_len),dtype=np.datetime64)

print("strat period list: ",len(strat_period_list))
pdb.set_trace()
#for each strat period, append to data matrices
for strat_period_ind, strat_period in enumerate(strat_period_list):
	# print("processing strat period ",strat_period_ind,"/",len(strat_period_list))
	start_ind = 0
	end_ind = start_ind + seq_len
	while end_ind < strat_period.shape[0]:
		to_append_pt = np.expand_dims(strat_period[start_ind:end_ind][pt_fields].values,0)
		pt_data = np.concatenate((pt_data,to_append_pt),axis=0)
		tmp_df = strat_period[start_ind:end_ind]
		#if no obs, continue
		if not pd.isnull(tmp_df['obs_hyp']).all():
			to_append_dates = np.expand_dims(tmp_df['datetime'].values,0)

			#if train data, append to train data
			if ((not pd.isnull(tmp_df['obs_hyp']).all()) & (tmp_df['splitsample']==0)).any():
				# print("time to append trn data")
				to_append_trn = np.expand_dims(tmp_df[trn_test_fields].values,0)
				#delete test data in train seq
				if np.where(tmp_df[tmp_df['splitsample']==1])[0].shape[0] != 0:
					tst_ind_to_del = np.where(tmp_df['splitsample']==1)[0]
					to_append_trn[:,tst_ind_to_del,-1] = np.nan
				assert pd.notnull(to_append_trn[:,:,:-1]).all()
				assert pd.notnull(to_append_trn[:,:,-1]).any()
				trn_data = np.concatenate((trn_data,to_append_trn),axis=0)
				trn_dates = np.concatenate((trn_dates,to_append_dates),axis=0)

			#if test data, append to tst data
			if ((not pd.isnull(tmp_df['obs_hyp']).all()) & (tmp_df['splitsample']==1)).any():
				# print("time to append tst data")
				to_append_tst = np.expand_dims(tmp_df[trn_test_fields].values,0)
				if np.where(tmp_df[tmp_df['splitsample']==0])[0].shape[0] != 0:
					# print("time to delete train obs in test seq")
					trn_ind_to_del = np.where(tmp_df['splitsample']==0)[0]
					to_append_tst[:,trn_ind_to_del,-1] = np.nan
				assert pd.notnull(to_append_tst[:,:,:-1]).all()
				assert pd.notnull(to_append_tst[:,:,-1]).any()
				tst_data = np.concatenate((tst_data,to_append_tst),axis=0)
				tst_dates = np.concatenate((tst_dates,to_append_dates),axis=0)

		start_ind += seq_len
		end_ind += seq_len
	if not strat_period.shape[0] % seq_len == 0:
		#get last sequence which starts at end_ind - seq_length
		end_ind = strat_period.shape[0] - 1
		start_ind = end_ind - seq_len
		to_append_pt = np.expand_dims(strat_period[start_ind:end_ind][pt_fields].values,0)
		pt_data = np.concatenate((pt_data,to_append_pt),axis=0)
		tmp_df = strat_period[start_ind:end_ind]
		#if no obs, continue
		if not pd.isnull(tmp_df['obs_hyp']).all():
			to_append_dates = np.expand_dims(tmp_df['datetime'].values,0)
			#if train data, append to train data
			if ((not pd.isnull(tmp_df['obs_hyp']).all()) & (tmp_df['splitsample']==0)).any():
				to_append_trn = np.expand_dims(tmp_df[trn_test_fields].values,0)
				#delete test data in train seq
				if np.where(tmp_df[tmp_df['splitsample']==1])[0].shape[0] != 0:
					tst_ind_to_del = np.where(tmp_df['splitsample']==1)[0]
					to_append_trn[:,tst_ind_to_del,-1] = np.nan
				assert pd.notnull(to_append_trn[:,:,:-1]).all()
				assert pd.notnull(to_append_trn[:,:,-1]).any()
				trn_data = np.concatenate((trn_data,to_append_trn),axis=0)
				trn_dates = np.concatenate((trn_dates,to_append_dates),axis=0)

			#if test data, append to tst data
			if ((not pd.isnull(tmp_df['obs_hyp']).all()) & (tmp_df['splitsample']==1)).any():
				to_append_tst = np.expand_dims(tmp_df[trn_test_fields].values,0)
				if np.where(tmp_df[tmp_df['splitsample']==0])[0].shape[0] != 0:
					# print("time to delete train obs in test seq")
					trn_ind_to_del = np.where(tmp_df['splitsample']==0)[0]
					to_append_tst[:,trn_ind_to_del,-1] = np.nan
				assert pd.notnull(to_append_tst[:,:,:-1]).all()
				assert pd.notnull(to_append_tst[:,:,-1]).any()
				tst_data = np.concatenate((tst_data,to_append_tst),axis=0)
				tst_dates = np.concatenate((tst_dates,to_append_dates),axis=0)
trn_data_norm = trn_data.copy()
tst_data_norm = tst_data.copy()
pt_data_norm = pt_data.copy()
trn_data_norm[:,:,:-1] = (trn_data_norm[:,:,:-1] - mean_feats[:-2]) / std_feats[:-2]
tst_data_norm[:,:,:-1] = (tst_data_norm[:,:,:-1] - mean_feats[:-2]) / std_feats[:-2]
pt_data_norm[:,:,:-1] = (pt_data_norm[:,:,:-1] - mean_feats[:-2]) / std_feats[:-2]

pt_data_path = "../../data/processed/"+site_id+"/pt"
pt_norm_data_path = "../../data/processed/"+site_id+"/pt_norm"
pt_dates_path = "../../data/processed/"+site_id+"/pt_dates"
trn_data_path = "../../data/processed/"+site_id+"/trn"
trn_norm_data_path = "../../data/processed/"+site_id+"/trn_norm"
trn_dates_path = "../../data/processed/"+site_id+"/trn_dates"
tst_data_path = "../../data/processed/"+site_id+"/tst"
tst_norm_data_path = "../../data/processed/"+site_id+"/tst_norm"
tst_dates_path = "../../data/processed/"+site_id+"/tst_dates"

np.save(pt_data_path,pt_data)
np.save(pt_norm_data_path,pt_data_norm)
np.save(pt_dates_path,pt_dates)
np.save(trn_data_path,trn_data)
np.save(trn_norm_data_path,trn_data_norm)
np.save(trn_dates_path,trn_dates)
np.save(tst_data_path,tst_data)
np.save(tst_norm_data_path,tst_data_norm)
np.save(tst_dates_path,tst_dates)



	# total_df.to_feather("./temp/all_site_feats.feather")
