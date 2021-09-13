import re
import pandas as pd
import pdb
import numpy as np

site_ids = np.load("../../metadata/201site_ids.npy",allow_pickle=True)
raw_data_dir = '../../data/raw/DOzip/'
process_rmses = []

sites_without_data = ['nhdhr_121857622','nhdhr_120018361','nhdhr_126212479','nhdhr_120018006',\
                      'nhdhr_120018027']
site_ids = site_ids[~np.isin(site_ids,sites_without_data)]

sbatch = ""
ct = 0
# start = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000]
# end = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,12416]

for i in site_ids:
    ct += 1
    #for each unique lake
    print(i)

    # if not os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAll_pball"): 
    header = "#!/bin/bash -l\n#SBATCH --time=00:30:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --gres=gpu:k40:2\n#SBATCH --output=DO_indLSTM_%s.out\n#SBATCH --error=DO_indLSTM_%s.err\n\n#SBATCH -p k40"%(i,i)
    script = "source /home/kumarv/willa099/takeme_DOtrain.sh\n" #cd to directory with training script
    # script2 = "python write_NLDAS_xy_pairs.py %s %s"%(l,l2)
    script2 = "python trian_ind_lake_lstm.py %s"%(i)
    # script2 = "python predict_lakes_EALSTM_COLD_DEBUG.py %s %s"%(l,l2)
    
    # script3 = "python singleModel_customSparse.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch indLSTM_test_%s.sh"%(i),sbatch])
    with open('../../hpc/indLSTM_test_{}.sh'.format(i), 'w') as output:
        output.write(all)


compile_job_path= '../../hpc/indLSTM_test_jobs.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)

print(ct, " jobs created, run this to submit: ", compile_job_path)