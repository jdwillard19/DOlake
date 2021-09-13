import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.init import xavier_normal_
from datetime import date
import pandas as pd
import pdb
import random
import math
import sys
import os
sys.path.append('../../data')
sys.path.append('../data')
sys.path.append('../../models')
sys.path.append('../models')
# from data_operations import calculatePhysicalLossDensityDepth
from pytorch_model_operations import saveModel
import pytorch_data_operations
import datetime
from torch.utils.data import DataLoader



#script start
currentDT = datetime.datetime.now()
print(str(currentDT))


####################################################3
# (Sept 2020 - Jared) source model script, takes lakename as required command line argument
###################################################33

#enable/disable cuda 
use_gpu = True 
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=10)

#cmd args
site_id = sys.argv[1]


### debug tools
debug_train = False
debug_end = False
verbose = True
pretrain = True
save = True
save_pretrain = True

#RMSE threshold for pretraining
rmse_threshold = .7 #TBD



#####################3
#params
###########################33
first_save_epoch = 0
patience = 100

n_hidden = 128 #fixed
train_epochs = 10000
pretrain_epochs = 70

unsup_loss_cutoff = 40
dc_unsup_loss_cutoff = 1e-3
dc_unsup_loss_cutoff2 = 1e-2
#ow
seq_length = 60 #how long of sequences to use in model
begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
n_features = 14  #number of physical drivers
win_shift = 30 #how much to slide the window on training set each time
save = True 


lakename = site_id
print("lake: "+lakename)
data_dir = "../../data/processed/"+lakename+"/"

#data paths
pt_data_path = "../../data/processed/"+site_id+"/pt.npy"
pt_norm_data_path = "../../data/processed/"+site_id+"/pt_norm.npy"
pt_dates_path = "../../data/processed/"+site_id+"/pt_dates.npy"
trn_data_path = "../../data/processed/"+site_id+"/trn.npy"
trn_norm_data_path = "../../data/processed/"+site_id+"/trn_norm.npy"
trn_dates_path = "../../data/processed/"+site_id+"/trn_dates.npy"
tst_data_path = "../../data/processed/"+site_id+"/tst.npy"
tst_norm_data_path = "../../data/processed/"+site_id+"/tst_norm.npy"
tst_dates_path = "../../data/processed/"+site_id+"/tst_dates.npy"

#data
pt_data_raw = torch.from_numpy(np.array(np.load(pt_data_path,allow_pickle=True),dtype=np.float32))
pt_data = torch.from_numpy(np.array(np.load(pt_norm_data_path,allow_pickle=True),dtype=np.float32))
pt_dates = np.load(pt_dates_path,allow_pickle=True)
trn_data_raw = torch.from_numpy(np.array(np.load(trn_data_path,allow_pickle=True),dtype=np.float32))
trn_data = torch.from_numpy(np.array(np.load(trn_norm_data_path,allow_pickle=True),dtype=np.float32))
trn_dates = np.load(trn_dates_path, allow_pickle=True)
tst_data_raw = torch.from_numpy(np.array(np.load(tst_data_path,allow_pickle=True),dtype=np.float32))
tst_data = torch.from_numpy(np.array(np.load(tst_norm_data_path,allow_pickle=True),dtype=np.float32))
tst_dates = np.load(tst_dates_path, allow_pickle=True)

if trn_data.shape[0] < 2:
    print("not enough for validation set")
    pdb.set_trace()

last_trn_ind = int(np.round((trn_data.shape[0]*2)/3))
val_data = trn_data[last_trn_ind:,:,:]
print("trian size",trn_data.size())
###############################
# data preprocess
##################################
#create train and test sets


####################
#model params
########################

batch_size =trn_data.size()[0] #single batch
# yhat_batch_size = n_depths*1 #for physics based loss, not used right now
grad_clip = 1.0 #how much to clip the gradient 2-norm in training
lambda1 = 0.0001 #magnitude hyperparameter of l1 loss
                                               
#Dataset classes
class OxygenTrainDataset(Dataset):
    #training dataset class, allows Dataloader to load both input/target
    def __init__(self, trn_data):
        # depth_data = depth_trn
        self.len = trn_data.shape[0]
        # assert data.shape[0] ==trn_data depth_data.shape[0]
        self.x_data = trn_data[:,:,:-1].float()
        # self.x_depth = depth_data.float()
        self.y_data = trn_data[:,:,-1].float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# class TotalModelOutputDataset(Dataset):
#     #dataset for unsupervised input(in this case all the data)
#     def __init__(self, all_data, all_phys_data, all_dates):
#         #data of all model output, and corresponding unstandardized physical quantities
#         #needed to calculate physical loss
#         self.len = all_data.shape[0]
#         self.data = all_data[:,:,:-1].float()
#         self.label = all_data[:,:,-1].float() #DO NOT USE IN MODEL, FOR DEBUGGING
#         self.phys = all_phys_data[:,:,:].float()
#         helper = np.vectorize(lambda x: date.toordinal(pd.Timestamp(x).to_pydatetime()))
#         dates = helper(all_dates)
#         self.dates = dates

#     def __getitem__(self, index):
#         return self.data[index], self.phys[index], self.dates[index], self.label[index]

#     def __len__(self):
#         return self.len

#format training data for loading
pretrain_data = OxygenTrainDataset(pt_data)

#get depth area percent data, not used
# depth_areas = torch.from_numpy(hypsography).float().flatten()

# if use_gpu:
    # depth_areas = depth_areas.cuda()

#format total y-hat data for loading
# total_data = TotalModelOutputDataset(all_data, all_phys_data, all_dates)
n_batches = math.floor(trn_data.size()[0] / batch_size)

# assert yhat_batch_size == n_depths

#batch samplers used to draw samples in dataloaders
batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)


#method to calculate l1 norm of model
def calculate_l1_loss(model):
    def l1_loss(x):
        return torch.abs(x).sum()

    to_regularize = []
    # for name, p in model.named_parameters():
    for name, p in model.named_parameters():
        if 'bias' in name:
            continue
        else:
            #take absolute value of weights and sum
            to_regularize.append(p.view(-1))
    l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    l1_loss_val = l1_loss(torch.cat(to_regularize))
    return l1_loss_val


#lstm class
class myLSTM_Net(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(myLSTM_Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size = n_features, hidden_size=hidden_size, batch_first=True) 
        self.out = nn.Linear(hidden_size, 1)
        self.hidden = self.init_hidden()
        # self.w_upper_to_lower = []
        # self.w_lower_to_upper = []           

    def init_hidden(self, batch_size=0):
        # initialize both hidden layers
        if batch_size == 0:
            batch_size = self.batch_size
        ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
                xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
        if use_gpu:
            item0 = ret[0].cuda(non_blocking=True)
            item1 = ret[1].cuda(non_blocking=True)
            ret = (item0,item1)
        return ret
    
    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.float()
        x, hidden = self.lstm(x, self.hidden)
        self.hidden = hidden
        x = self.out(x)
        return x, hidden


lstm_net = myLSTM_Net(n_features, n_hidden, batch_size)

if use_gpu:
    lstm_net = lstm_net.cuda(0)

#define training loss function and optimizer
mse_criterion = nn.MSELoss()
optimizer = optim.AdamW(lstm_net.parameters())

#paths to save
if not os.path.exists("../../models/"+lakename):
    os.mkdir("../../models/"+lakename)
save_path = "../../models/"+lakename+"/pretrain_source_model"

min_loss = 99999
min_mse_tsterr = None
ep_min_mse = -1
epoch_since_best = 0

manualSeed = [random.randint(1, 99999999) for i in range(pretrain_epochs)]

#convergence variables

min_tst_rmse = 999
min_tst_epoch = -1
#############################################################
#pre- training loop
####################################################################
if pretrain:
    for epoch in range(pretrain_epochs):
        if verbose:
            print("pretrain epoch: ", epoch)
        torch.manual_seed(manualSeed[epoch])
        if use_gpu:
            torch.cuda.manual_seed_all(manualSeed[epoch])
        running_loss = 0.0

        #reload loader for shuffle
        #batch samplers used to draw samples in dataloaders
        batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)
        trainloader = DataLoader(pretrain_data, batch_sampler=batch_sampler, pin_memory=True)


        #zero the parameter gradients
        optimizer.zero_grad()
        avg_loss = 0
        # avg_unsup_loss = 0
        # avg_dc_unsup_loss = 0

        batches_done = 0
        # for i, batches in enumerate(multi_loader):
        #     #load data
        #     inputs = None
        #     targets = None
        #     depths = None
        #     unsup_inputs = None
        #     unsup_phys_data = None
        #     unsup_depths = None
        #     unsup_dates = None
    #     unsup_labels = None
        for m, data in enumerate(trainloader, 0):
            inputs = data[0].float()
            targets = data[1].float()
            #cuda commands
            if(use_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()

            #forward  prop
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            h_state = None
            inputs = inputs.float()
            outputs, h_state = lstm_net(inputs, h_state)
            outputs = outputs.view(outputs.size()[0],-1)

            loss_outputs = outputs[:,begin_loss_ind:]
            loss_targets = targets[:,begin_loss_ind:]

            if use_gpu:
                loss_outputs = loss_outputs.cuda()
                loss_targets = loss_targets.cuda()



            #calculate losses
            reg1_loss = 0
            if lambda1 > 0:
                reg1_loss = calculate_l1_loss(lstm_net)

            loss = mse_criterion(loss_outputs, loss_targets) + lambda1*reg1_loss


            avg_loss += loss

            batches_done += 1
            #backward prop
            loss.backward(retain_graph=False)
            if grad_clip > 0:
                clip_grad_norm_(lstm_net.parameters(), grad_clip, norm_type=2)

            #optimize
            optimizer.step()

            #zero the parameter gradients
            optimizer.zero_grad()

            #print statistics
            # running_loss += loss.item()
            # if verbose:
            #     if i % 3 == 2:
            #         print('[%d, %5d] loss: %.3f' %
            #              (epoch + 1, i + 1, running_loss / 3))
            #         running_loss = 0.0
        avg_loss = avg_loss / batches_done

        if verbose:
            print("rmse loss=", avg_loss)



        # testloader = torch.utils.data.DataLoader(tst_data, batch_size=tst_data.size()[0], shuffle=False, pin_memory=True)

        # with torch.no_grad():
        #     avg_mse = 0
        #     ct = 0
        #     for m, data in enumerate(testloader, 0):
        #         #now for mendota data
        #         #this loop is dated, there is now only one item in testloader

        #         #parse data into inputs and targets
        #         inputs = data[:,:,:-1].float()
        #         targets = data[:,:,-1].float()
        #         tmp_dates = tst_dates[:, :]

        #         if use_gpu:
        #             inputs = inputs.cuda()
        #             targets = targets.cuda()

        #         #run model
        #         h_state = None
        #         lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
        #         pred, h_state = lstm_net(inputs, h_state)
        #         pred = pred.view(pred.size()[0],-1)
        #         pred = pred[:, begin_loss_ind:]

        #         #calculate error
        #         targets = targets.cpu()
        #         loss_indices = torch.from_numpy(np.array(np.isfinite(targets.cpu()), dtype='bool_'))
        #         if use_gpu:
        #             targets = targets.cuda()
        #         inputs = inputs[:, begin_loss_ind:, :]
        #         mse = mse_criterion(pred[loss_indices], targets[loss_indices])
        #         #calculate error
        #         avg_mse += mse
        #         ct += 1
        #         # if mse > 0: #obsolete i think
        #         #     ct += 1
        #     avg_mse = avg_mse / ct

        #     if avg_mse < min_tst_rmse:
        #         min_tst_rmse = avg_mse
        #         min_tst_epoch = epoch
        #     print("test rmse: ", avg_mse, " (lowest rmse at epoch ",min_tst_epoch,": ",min_tst_rmse,")")

print("pretraining finished in " + str(epoch) +" epochs")
saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
     



#####################################################################################
####################################################3
# fine tune
###################################################33
##########################################################################################33

#####################
#params
###########################
patience = 1000
lambda1 = 0.0001
data_dir = "../../data/processed/"+lakename+"/"

#paths to save

pretrain_path = "../../models/"+lakename+"/pretrain_source_model"
save_path = "../../models/"+lakename+"/finetune_source_model_0.7"


###############################
# data preprocess
##################################
#create train and test sets

batch_size = trn_data.size()[0]



#Dataset classes
class TemperatureTrainDataset(Dataset):
    #training dataset class, allows Dataloader to load both input/target
    def __init__(self, trn_data):
        # depth_data = depth_trn
        self.len = trn_data.shape[0]
        self.x_data = trn_data[:,:,:-1].float()
        self.y_data = trn_data[:,:,-1].float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class TotalModelOutputDataset(Dataset):
    #dataset for unsupervised input(in this case all the data)
    def __init__(self, all_data, all_phys_data,all_dates):
        #data of all model output, and corresponding unstandardized physical quantities
        #needed to calculate physical loss
        self.len = all_data.shape[0]
        self.data = all_data[:,:,:-1].float()
        self.label = all_data[:,:,-1].float() #DO NOT USE IN MODEL
        self.phys = all_phys_data.float()
        helper = np.vectorize(lambda x: date.toordinal(pd.Timestamp(x).to_pydatetime()))
        dates = helper(all_dates)
        self.dates = dates

    def __getitem__(self, index):
        return self.data[index], self.phys[index], self.dates[index], self.label[index]

    def __len__(self):
        return self.len




#format training data for loading
train_data = TemperatureTrainDataset(trn_data)

#get depth area percent data

#format total y-hat data for loading
n_batches = math.floor(trn_data.size()[0] / batch_size)

#batch samplers used to draw samples in dataloaders
batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)



#load val/test data into enumerator based on batch size
testloader = torch.utils.data.DataLoader(tst_data, batch_size=tst_data.size()[0], shuffle=False, pin_memory=True)



#define LSTM model class
class myLSTM_Net(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(myLSTM_Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size = n_features, hidden_size=hidden_size, batch_first=True) #batch_first=True?
        self.out = nn.Linear(hidden_size, 1) #1?
        self.hidden = self.init_hidden()
        self.w_upper_to_lower = []
        self.w_lower_to_upper = []

    def init_hidden(self, batch_size=0):
        # initialize both hidden layers
        if batch_size == 0:
            batch_size = self.batch_size
        ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
                xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
        if use_gpu:
            item0 = ret[0].cuda(non_blocking=True)
            item1 = ret[1].cuda(non_blocking=True)
            ret = (item0,item1)
        return ret
    
    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.float()
        x, hidden = self.lstm(x, self.hidden)
        self.hidden = hidden
        x = self.out(x)
        return x, hidden

#method to calculate l1 norm of model
def calculate_l1_loss(model):
    def l1_loss(x):
        return torch.abs(x).sum()

    to_regularize = []
    # for name, p in model.named_parameters():
    for name, p in model.named_parameters():
        if 'bias' in name:
            continue
        else:
            #take absolute value of weights and sum
            to_regularize.append(p.view(-1))
    l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    l1_loss_val = l1_loss(torch.cat(to_regularize))
    return l1_loss_val


lstm_net = myLSTM_Net(n_features, n_hidden, batch_size)

pretrain_dict = torch.load(pretrain_path)['state_dict']
model_dict = lstm_net.state_dict()
pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
model_dict.update(pretrain_dict)
lstm_net.load_state_dict(pretrain_dict)

#tell model to use GPU if needed
if use_gpu:
    lstm_net = lstm_net.cuda()




#define loss and optimizer
mse_criterion = nn.MSELoss()
optimizer = optim.AdamW(lstm_net.parameters())#, weight_decay=0.01)

#training loop

min_mse = 99999
min_mse_tsterr = None
ep_min_mse = -1
best_pred_mat = np.empty(())
manualSeed = [random.randint(1, 99999999) for i in range(train_epochs)]






min_tst_rmse = 999
min_tst_epoch = -1
#############################################################
#pre- training loop
####################################################################
for epoch in range(train_epochs):
    if verbose:
        print("train epoch: ", epoch)
    torch.manual_seed(manualSeed[epoch])
    if use_gpu:
        torch.cuda.manual_seed_all(manualSeed[epoch])
    running_loss = 0.0

    #reload loader for shuffle
    #batch samplers used to draw samples in dataloaders
    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)
    trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)


    #zero the parameter gradients
    optimizer.zero_grad()
    avg_loss = 0
    # avg_unsup_loss = 0
    # avg_dc_unsup_loss = 0

    batches_done = 0
    # for i, batches in enumerate(multi_loader):
    #     #load data
    #     inputs = None
    #     targets = None
    #     depths = None
    #     unsup_inputs = None
    #     unsup_phys_data = None
    #     unsup_depths = None
    #     unsup_dates = None
#     unsup_labels = None
    for m, data in enumerate(trainloader, 0):
        inputs = data[0].float()
        targets = data[1].float()
        #cuda commands
        if(use_gpu):
            inputs = inputs.cuda()
            targets = targets.cuda()

        #forward  prop
        lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
        h_state = None
        inputs = inputs.float()
        outputs, h_state = lstm_net(inputs, h_state)
        outputs = outputs.view(outputs.size()[0],-1)

        loss_indices = torch.from_numpy(np.array(np.isfinite(targets.cpu()), dtype='bool_'))

        loss_outputs = outputs[:,begin_loss_ind:]
        loss_targets = targets[:,begin_loss_ind:]

        if use_gpu:
            loss_outputs = loss_outputs.cuda()
            loss_targets = loss_targets.cuda()



        #calculate losses
        reg1_loss = 0
        if lambda1 > 0:
            reg1_loss = calculate_l1_loss(lstm_net)

        loss = mse_criterion(outputs[loss_indices], targets[loss_indices]) + lambda1*reg1_loss


        avg_loss += loss

        batches_done += 1
        #backward prop
        loss.backward(retain_graph=False)
        if grad_clip > 0:
            clip_grad_norm_(lstm_net.parameters(), grad_clip, norm_type=2)

        #optimize
        optimizer.step()

        #zero the parameter gradients
        optimizer.zero_grad()

        #print statistics
        # running_loss += loss.item()
        # if verbose:
        #     if i % 3 == 2:
        #         print('[%d, %5d] loss: %.3f' %
        #              (epoch + 1, i + 1, running_loss / 3))
        #         running_loss = 0.0
    avg_loss = avg_loss / batches_done

    if verbose:
        print("rmse loss=", avg_loss)

    valloader = torch.utils.data.DataLoader(val_data, batch_size=tst_data.size()[0], shuffle=False, pin_memory=True)

    with torch.no_grad():
        avg_mse = 0
        ct = 0
        for m, data in enumerate(valloader, 0):
            #now for mendota data
            #this loop is dated, there is now only one item in testloader

            #parse data into inputs and targets
            inputs = data[:,:,:-1].float()
            targets = data[:,:,-1].float()
            tmp_dates = tst_dates[:, :]

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            #run model
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            pred, h_state = lstm_net(inputs, h_state)
            pred = pred.view(pred.size()[0],-1)
            pred = pred[:, begin_loss_ind:]

            #calculate error
            targets = targets.cpu()
            loss_indices = torch.from_numpy(np.array(np.isfinite(targets.cpu()), dtype='bool_'))
            if use_gpu:
                targets = targets.cuda()
            inputs = inputs[:, begin_loss_ind:, :]
            mse = mse_criterion(pred[loss_indices], targets[loss_indices])
            #calculate error
            avg_mse += mse
            ct += 1
            # if mse > 0: #obsolete i think
            #     ct += 1
        avg_mse = avg_mse / ct

        if avg_mse < min_tst_rmse:
            min_tst_rmse = avg_mse
            min_tst_epoch = epoch
            saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
        else:
            epoch_since_best += 1
            if epoch_since_best > patience:
                print("patience met")
                break
        print("val rmse: ", avg_mse, " (lowest val rmse at epoch ",min_tst_epoch,": ",min_tst_rmse,")")



print("training finished in " + str(epoch) +" epochs")
# saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)


#test model
testloader = torch.utils.data.DataLoader(tst_data, batch_size=tst_data.size()[0], shuffle=False, pin_memory=True)

with torch.no_grad():
    avg_mse = 0
    ct = 0
    for m, data in enumerate(testloader, 0):
        #now for mendota data
        #this loop is dated, there is now only one item in testloader

        #parse data into inputs and targets
        inputs = data[:,:,:-1].float()
        targets = data[:,:,-1].float()
        tmp_dates = tst_dates[:, :]

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        #run model
        h_state = None
        lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
        pred, h_state = lstm_net(inputs, h_state)
        pred = pred.view(pred.size()[0],-1)
        pred = pred[:, begin_loss_ind:]

        #calculate error
        targets = targets.cpu()
        loss_indices = torch.from_numpy(np.array(np.isfinite(targets.cpu()), dtype='bool_'))
        if use_gpu:
            targets = targets.cuda()
        inputs = inputs[:, begin_loss_ind:, :]
        mse = mse_criterion(pred[loss_indices], targets[loss_indices])
        #calculate error
        avg_mse += mse
        ct += 1
        # if mse > 0: #obsolete i think
        #     ct += 1
    avg_mse = avg_mse / ct

    if avg_mse < min_tst_rmse:
        min_tst_rmse = avg_mse
        min_tst_epoch = epoch
    print("test rmse: ", np.sqrt(avg_mse))