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
hardcode = True
if not hardcode:
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
			total_df.to_feather("./temp/all_site_feats.feather")
	else:
		total_df = pd.read_feather("./temp/all_site_feats.feather")

	total_df = total_df.drop(['date','datetime','site_id'],axis=1)
	total_df = total_df.fillna(value=np.nan)
	mean_feats = []
	std_feats = []
	for i in range(total_df.shape[1]):
		print("feat ",i)
		mean_feats.append(np.nanmean(total_df.iloc[:,i],axis=0))
		std_feats.append(np.nanstd(total_df.iloc[:,i],axis=0))
else:
	total_df = pd.read_feather("./temp/all_site_feats.feather")
	total_df = total_df.fillna(value=np.nan)
	total_feat_df = total_df.drop(['date','datetime','site_id'],axis=1)
	mean_feats = [7441.5, 4.931483423710512, 19.70167169739363, 14.904954037755175, \
				  78237347.42516738, 35354870.6379829, 7791403.357730445, 4.530119869947415,\
				  6.96297079241955, 121.5697981498898, 234.48922720701856, 485.30388998631906,\
				  24.853238995672296, 8996.263618306573, nan, 11.089948649244011, 17403670.32968644, 12.156414210564925, 10.21764712759074, 133389098.04953545, 4.158370887251586, 7.063745525902752, 1999.1189196452567, 183.94215264713787, 0.44068558042894906, 1.0529250965601147, 0.7385204901156043, 0.5618458977525399, 1.0664130581215956, 0.723603183152242, 8.596011978838595, 10.4482636840888, 179.24117069212775, 2289.026613040851, 442.3825882929269, 2003.6905684912253, 101.6064096745379, 34.33592717912253, 92.60627300195189, 906595532696.0215, 9.211200117975048, 8.958354839997888, 3.2436941103763113, 183.94215264713787, 0.303338223192243]
	std_feats = [4296.640693611696, 3.9135773130155114, 5.1938443992663945, 5.41263358522884, 349295269.7526865, 182450151.17300767, 42878922.26957663, 1.6534891965038068, 12.790476005382871, 275.49552987718704, 313.97513072365814, 659.7991525525778, 575.8894069589172, 1565.4929930553565, nan, 4.875297689585714, 68337784.28012213, 10.184009422543403, 7.471911914428431, 538521186.249026, 4.379090285894764, 4.164327407477185, 11.763992886628667, 105.18270892602168, 0.49646933402874877, 0.38737433202823857, 0.32553888514525603, 0.3606137050276429, 0.38928466540873613, 0.5509708197369108, 44.741285527322866, 5.294555245934269, 313.8957571884795, 1362.7927637566015, 421.8727699861578, 778.8547443096875, 15.568745125437568, 706.4754762940715, 37.80524213145845, 4083881003742.3457, 3.0937110295878205, 2.218655202625219, 3.5991971773803284, 105.18270892602168, 0.4597000604120212]
