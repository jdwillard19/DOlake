import numpy as np
import pandas as pd
import pdb
import os

######################################################################3
# adds static features to existing preprocessed data for 201 lakes
#########################################################################

#get list of lakes
raw_data_dir = '../../data/raw/DOzip/'

#load land use data
land_use = pd.read_csv(raw_data_dir+"../landuse.csv")
asym_bin_landuse = pd.get_dummies(land_use['LANDUSE'])
asym_bin_landuse.columns = ["is_"+x for x in asym_bin_landuse.columns]
land_use = pd.concat([land_use,asym_bin_landuse],axis=1)
land_use.drop(['LANDUSE'],axis=1,inplace=True)

site_ids = np.load("../../metadata/201site_ids.npy",allow_pickle=True)
dyn_feats = ["thermocline_depth","temperature_epi","temperature_hypo",\
			  "volume_epi","volume_hypo","td_area","wind","airtemp",\
			  "fnep","fmineral","fsed","fatm","fentr_epi",\
			  "o2_epi"]

			  # fentr_hyp all none?
stat_feat_names = ["area_surface","max.d"]
land_use_stat_feats = ["water","developed","barren","forest","shrubland",
					   "herbaceous","cultivated","wetlands","is_cultivated","is_developed",
					   "is_forest","is_herbaceous","is_water","is_wetlands"]

obs_pt = ['o2_hyp']
obs_name = ['obs_hyp']
pt_fields = dyn_feats + obs_pt
trn_test_fields = dyn_feats + obs_name
pt_fields_wStat = dyn_feats + stat_feat_names + land_use_stat_feats + obs_pt
trn_test_fields_wStat = dyn_feats + stat_feat_names + land_use_stat_feats + obs_name
all_feats = dyn_feats + stat_feat_names + land_use_stat_feats
win_shift = 30
seq_len = 60


#get features and calc stats\
total_df = pd.DataFrame(columns=all_feats)
hardcode = True
if not hardcode:
	if not os.path.exists("./temp/all_site_feats_wStat.feather"):
		for i,site_id in enumerate(site_ids):
			print("pre: site ",i,"/",len(site_ids))
			if os.path.exists(raw_data_dir+site_id+"/"+site_id+".feather"):
				site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")
				site_df['site_id'] = site_id
				site_df = pd.merge(land_use[land_use['nhdr_id']==site_id], site_df,left_on='nhdr_id',right_on='site_id')
				site_df.drop(['nhdr_id'],axis=1,inplace=True)
				total_df = pd.concat([total_df,site_df])
			else:
				print("no file?")
				continue
		total_df.reset_index(inplace=True)
		total_df.to_feather("./temp/all_site_feats_wStat.feather")
	else:
		total_df = pd.read_feather("./temp/all_site_feats_wStat.feather")

	total_feat_df = total_df.drop(['date','datetime','site_id','fentr_hyp'],axis=1)
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
	total_df = pd.read_feather("./temp/all_site_feats_wStat.feather")
	total_df = total_df.fillna(value=np.nan)
	total_feat_df = total_df.drop(['date','datetime','site_id'],axis=1)
	mean_feats = np.load("temp/mean_feats.npy")
	std_feats = np.load("temp/std_feats.npy")

#number of static features we're adding
n_static = 16
n_dyn = 14
for i,site_id in enumerate(site_ids):
	# site_id = 'nhdhr_120018027'
	print("processing site ",i,"/",len(site_ids),": ",site_id)

	dir_path = "../../data/processed/"+site_id+"/"

	#load old
	trn = np.array(np.load(dir_path+"trn.npy",allow_pickle=True),dtype=np.float32)
	trn_norm = np.array(np.load(dir_path+"trn_norm.npy",allow_pickle=True),dtype=np.float32)
	tst = np.array(np.load(dir_path+"tst.npy",allow_pickle=True),dtype=np.float32)
	tst_norm = np.array(np.load(dir_path+"tst_norm.npy",allow_pickle=True),dtype=np.float32)
	pt = np.array(np.load(dir_path+"pt.npy",allow_pickle=True),dtype=np.float32)
	pt_norm = np.array(np.load(dir_path+"pt_norm.npy",allow_pickle=True),dtype=np.float32)


	dyn_feat_inds = np.arange(0,14,dtype=np.int16)
	stat_feat_inds = np.arange(14,30,dtype=np.int16)

	stat_feats = total_df[total_df['site_id']==site_id].iloc[0][stat_feat_names+land_use_stat_feats].values.astype(float)
	stat_feats_norm = (stat_feats - mean_feats[n_dyn:])/std_feats[n_dyn:]
	#new structs
	new_trn = np.empty((trn.shape[0],trn.shape[1],trn.shape[2]+n_static),dtype=np.float32)
	new_trn[:] = np.nan
	new_trn_norm = np.empty((trn_norm.shape[0],trn_norm.shape[1],trn_norm.shape[2]+n_static),dtype=np.float32)
	new_trn_norm[:] = np.nan
	new_tst = np.empty((tst.shape[0],tst.shape[1],tst.shape[2]+n_static),dtype=np.float32)
	new_tst[:] = np.nan
	new_tst_norm = np.empty((tst_norm.shape[0],tst_norm.shape[1],tst_norm.shape[2]+n_static),dtype=np.float32)
	new_tst_norm[:] = np.nan
	new_pt = np.empty((pt.shape[0],pt.shape[1],pt.shape[2]+n_static),dtype=np.float32)
	new_pt[:] = np.nan
	new_pt_norm = np.empty((pt_norm.shape[0],pt_norm.shape[1],pt_norm.shape[2]+n_static),dtype=np.float32)
	new_pt_norm[:] = np.nan
	new_trntst = np.empty((trn.shape[0]+tst.shape[0],trn.shape[1],trn.shape[2]+n_static),dtype=np.float32)
	new_trntst[:] = np.nan
	new_trntst_norm = np.empty((trn_norm.shape[0]+tst_norm.shape[0],trn_norm.shape[1],trn_norm.shape[2]+n_static),dtype=np.float32)
	new_trntst_norm[:] = np.nan
	#add data to each
	new_trn[:,:,dyn_feat_inds] = trn[:,:,dyn_feat_inds] #dyn feats
	new_trn[:,:,stat_feat_inds] = stat_feats
	new_trn[:,:,-1] = trn[:,:,-1] #obs

	new_trn_norm[:,:,dyn_feat_inds] = trn_norm[:,:,dyn_feat_inds] #dyn feats
	new_trn_norm[:,:,stat_feat_inds] = stat_feats_norm
	new_trn_norm[:,:,-1] = trn_norm[:,:,-1] #obs


	new_tst[:,:,dyn_feat_inds] = tst[:,:,dyn_feat_inds] #dyn feats
	new_tst[:,:,stat_feat_inds] = stat_feats
	new_tst[:,:,-1] = tst[:,:,-1] #obs

	new_tst_norm[:,:,dyn_feat_inds] = tst_norm[:,:,dyn_feat_inds] #dyn feats
	new_tst_norm[:,:,stat_feat_inds] = stat_feats_norm
	new_tst_norm[:,:,-1] = tst_norm[:,:,-1] #obs

	new_pt[:,:,dyn_feat_inds] = pt[:,:,dyn_feat_inds] #dyn feats
	new_pt[:,:,stat_feat_inds] = stat_feats
	new_pt[:,:,-1] = pt[:,:,-1] #obs

	new_pt_norm[:,:,dyn_feat_inds] = pt_norm[:,:,dyn_feat_inds] #dyn feats
	new_pt_norm[:,:,stat_feat_inds] = stat_feats_norm
	new_pt_norm[:,:,-1] = pt_norm[:,:,-1] #obs

	new_trntst = np.vstack((new_trn,new_tst))
	new_trntst_norm = np.vstack((new_trn_norm,new_tst_norm))


	#save new datas
	pt_data_path = "../../data/processed/"+site_id+"/pt_wStat"
	pt_norm_data_path = "../../data/processed/"+site_id+"/pt_norm_wStat"
	trn_data_path = "../../data/processed/"+site_id+"/trn_wStat"
	trn_norm_data_path = "../../data/processed/"+site_id+"/trn_norm_wStat"
	tst_data_path = "../../data/processed/"+site_id+"/tst_wStat"
	tst_norm_data_path = "../../data/processed/"+site_id+"/tst_norm_wStat"
	trntst_data_path = "../../data/processed/"+site_id+"/trntst_wStat"
	trntst_norm_data_path = "../../data/processed/"+site_id+"/trntst_norm_wStat"
	np.save(pt_data_path,new_pt)
	np.save(pt_norm_data_path,new_pt_norm)
	np.save(trn_data_path,new_trn)
	np.save(trn_norm_data_path,new_trn_norm)
	np.save(tst_data_path,new_tst)
	np.save(tst_norm_data_path,new_tst_norm)
	np.save(trntst_data_path,new_trntst)
	np.save(trntst_norm_data_path,new_trntst_norm)
