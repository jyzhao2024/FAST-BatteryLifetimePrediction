import torch
import glob
import os
import pickle
import random
import time
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader
from Informer_cyx_modified import *
from flash_probsparse_attention import *
from common_flash_attention import Trainer, FineTrainer
from thop import profile
from hyperparameter_optimization_1 import grid_search, random_search

# In[2]:


def seed_torch(seed=1029):
    '''
    Args:
        seed: int型

    Returns:

    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# save dict
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


# load dict
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def interp(v, q, num):
    f = interpolate.interp1d(v, q, kind='linear')
    v_new = np.linspace(v[0], v[-1], num)
    q_new = f(v_new)
    vq_new = np.concatenate((v_new.reshape(-1, 1), q_new.reshape(-1, 1)), axis=1)
    return q_new

def get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor, x=None, all_v=None,
           all_q=None):
    """
    Args:
        n_cyc (int): The previous cycles number for model input
        in_stride (int): The interval in the previous cycles number
        fea_num (int): The number of interpolation
        v_low (float): Voltage minimum for normalization
        v_upp (float): Voltage maximum for normalization
        q_low (float): Capacity minimum for normalization
        q_upp (float): Capacity maximum for normalization
        rul_factor (float): The RUL factor for normalization
        cap_factor (float): The capacity factor for normalization
    """
    A = load_obj(f'./our_data/{name}')[name]
    A_rul = A['rul']
    A_dq = A['dq']
    A_df = A['data']
    all_v = []
    all_q =[]
    All_v, All_q  = [] ,[]
    fig = plt.figure(figsize=(20, 10))
    all_idx = list(A_dq.keys())[9:]
    all_fea, rul_lbl, cap_lbl = [], [], []
    legend_labels = []
    used_label_positions = []
    for cyc in all_idx:
        tmp = A_df[cyc]
        tmp = tmp.loc[tmp['Status'].apply(lambda x: not 'discharge' in x)]

        left = (tmp['Current (mA)'] < 5000).argmax()+1
        right = (tmp['Current (mA)'] < 1090).argmax() - 2

        tmp = tmp.iloc[left:right]

        tmp_v = tmp['Voltage (V)'].values

        tmp_q = tmp['Capacity (mAh)'].values



        tmp_t = tmp['Time (s)'].values
        v_fea = interp(tmp_t, tmp_v, fea_num)
        q_fea = interp(tmp_t, tmp_q, fea_num)

        tmp_fea = np.hstack((v_fea.reshape(-1, 1), q_fea.reshape(-1, 1)))

        all_fea.append(np.expand_dims(tmp_fea, axis=0))
        rul_lbl.append(A_rul[cyc])
        cap_lbl.append(A_dq[cyc])
    all_fea = np.vstack(all_fea)
    rul_lbl = np.array(rul_lbl)
    cap_lbl = np.array(cap_lbl)

    all_fea_c = all_fea.copy()
    all_fea_c[:, :, 0] = (all_fea_c[:, :, 0] - v_low) / (v_upp - v_low)
    all_fea_c[:, :, 1] = (all_fea_c[:, :, 1] - q_low) / (q_upp - q_low)
    dif_fea = all_fea_c - all_fea_c[0:1, :, :]
    all_fea = np.concatenate((all_fea, dif_fea), axis=2)

    all_fea = np.lib.stride_tricks.sliding_window_view(all_fea, (n_cyc, fea_num, 4))
    cap_lbl = np.lib.stride_tricks.sliding_window_view(cap_lbl, (n_cyc,))
    all_fea = all_fea.squeeze(axis=(1, 2,))
    rul_lbl = rul_lbl[n_cyc - 1:]
    sample_indices = np.linspace(0, n_cyc - 1, sample_count, dtype=int)

#
    #selected_samples = your_data[sample_indices]
    all_fea = all_fea[:, (in_stride - 1)::in_stride, :, :]
    cap_lbl = cap_lbl[:, (in_stride - 1)::in_stride, ]
    #all_fea = all_fea[:, sample_indices, :, :]
    #cap_lbl = cap_lbl[:, sample_indices, ]
    print(all_fea.shape)

    all_fea_new = np.zeros(all_fea.shape)
    all_fea_new[:, :, :, 0] = (all_fea[:, :, :, 0] - v_low) / (v_upp - v_low)
    all_fea_new[:, :, :, 1] = (all_fea[:, :, :, 1] - q_low) / (q_upp - q_low)
    all_fea_new[:, :, :, 2] = all_fea[:, :, :, 2]
    all_fea_new[:, :, :, 3] = all_fea[:, :, :, 3]
    print(f'{name} length is {all_fea_new.shape[0]}',
          'v_max:', '%.4f' % all_fea_new[:, :, :, 0].max(),
          'q_max:', '%.4f' % all_fea_new[:, :, :, 1].max(),
          'dv_max:', '%.4f' % all_fea_new[:, :, :, 2].max(),
          'dq_max:', '%.4f' % all_fea_new[:, :, :, 3].max())
    rul_lbl = rul_lbl / rul_factor
    cap_lbl = cap_lbl / cap_factor

    return all_fea_new, np.hstack((rul_lbl.reshape(-1, 1), cap_lbl))

# # In[3]:

n_cyc = 40
in_stride = 2
fea_num = 100
sample_count = 20
v_low = 3.36
v_upp = 3.60
q_low = 610
q_upp = 1190
rul_factor = 3000
cap_factor = 1190
#
#
all_loader = dict()
print('----init_train----')
if os.path.exists('./all_loader.pkl'):

    all_loader = load_obj('./all_loader')
else:

    pkl_list = glob.glob('./*-*.pkl')
    train_name = []
    for name in pkl_list:
        name_ = name.split('\\')[-1][:-4]
        train_name.append(name_)
    for name in train_name:
        tmp_fea, tmp_lbl = get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor)
        all_loader.update({name:{'fea':tmp_fea,'lbl':tmp_lbl}})
        save_obj(all_loader, './all_loader')
