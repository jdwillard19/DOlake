import numpy as np
import pandas as pd
import pdb
site_ids = np.load("../../metadata/201site_ids.npy",allow_pickle=True)
raw_data_dir = '../../data/raw/DOzip/'
process_rmses = []

sites_without_data = ['nhdhr_121857622','nhdhr_120018361','nhdhr_126212479','nhdhr_120018006']
sites_without_data2 = sites_without_data + ['nhdhr_120018027', 'nhdhr_120020360', 'nhdhr_120020376',\
                                            'nhdhr_120018182', 'nhdhr_120018012', 'nhdhr_120017997',\
                                            'nhdhr_120018017', 'nhdhr_105231881', 'nhdhr_120020352',\
                                            'nhdhr_120017997']
site_ids = site_ids[~np.isin(site_ids,sites_without_data2)]
def rmse(pred,targ):
    return np.sqrt(((pred - targ)**2).mean())

all_obs_df = pd.DataFrame()
per_site_df = pd.DataFrame()
for ct, site_id in enumerate(site_ids):
    print(ct,"/",len(site_ids), " sites: ",site_id)
    site_df = pd.read_feather(raw_data_dir+site_id+"/"+site_id+".feather")

    #filter to test obs
    site_df_test = site_df[(site_df['splitsample']==1) & (pd.notnull(site_df['obs_hyp']))]

    site_rmse_process = rmse(site_df_test['obs_hyp'], site_df_test['o2_hyp'])

    if np.isnan(site_rmse_process):
        print("nan?")
        pdb.set_trace()
    process_rmses.append(site_rmse_process)

    lstm_df = pd.read_feather("../../results/singleSiteLSTM_"+site_id+".feather")
    lstm_df['date_ts'] = pd.to_datetime(lstm_df['date'], utc = True) 

    site_df_test = site_df_test.merge(lstm_df, left_on='date', right_on='date_ts')
    all_obs_df = pd.concat([site_df_test,all_obs_df],ignore_index=True)

    #rmse df
    site_rmse_lstm = rmse(site_df_test['lstm_pred'],site_df_test['obs_hyp'])
    site_rmse_df = pd.DataFrame()
    site_rmse_df['site_id'] = [site_id]
    site_rmse_df['o2_model_rmse'] = [site_rmse_process]
    site_rmse_df['lstm_model_rmse'] = [site_rmse_lstm]


pdb.set_trace()