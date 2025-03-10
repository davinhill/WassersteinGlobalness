# %%
import os
import sys

import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

# import matplotlib; matplotlib.pyplot.switch_backend('agg')
import matplotlib; matplotlib.rc('image', cmap='gray')
import matplotlib.pyplot as plt

import time
import numpy as np
import pandas as pd

os.chdir('../../')
sys.path.append('./')
from utils.locality_utilities import *
from utils.utils import *

import pickle as pkl
from datetime import datetime

import quantus


from argparse import ArgumentParser
parser = ArgumentParser(description='AUC Experiment')

parser.add_argument('--explainer', type=str, default = 'smoothgrad')
parser.add_argument('--dataset', type=str, default = 'divorce')
parser.add_argument('--exp_norm', type=str, default = 'sphere_l2_bound')
parser.add_argument('--nUnif', type=int, default = 10000)
parser.add_argument('--abs', type=int, default = 0)
parser.add_argument('--calc_exp', type=int, default = 0)
parser.add_argument('--seed', type=int, default = 0)
parser.add_argument('--metric', type=str, default = 'euclidean')
parser.add_argument('--scale_diameter', type=str, default = 'None')
parser.add_argument('--solver', type=str, default = 'pot_slice_symmetric')
parser.add_argument('--centering', type=str, default = 'mean')
parser.add_argument('--l2_bound', type=float, default = 1)
#TODO: if not provided, l2_bound should default to explanation max.
parser.add_argument('--std', type=float, default =  0.0) # sigma for smoothed explainers
parser.add_argument('--n_samples_to_explain', type=int, default = 1000)
parser.add_argument('--normalize_global', type=int, default = 1)
parser.add_argument('--exp_save_path', type=str, default = './Files/Results/saved_explanations_1.24_1k')
parser.add_argument('--metric_save_path', type=str, default = './Files/Results/AUC')
args = parser.parse_args()

print_args(args)


# %%
# Hyperparameters
n_samples_to_explain = args.n_samples_to_explain
normalize_global = args.normalize_global
# n_samples_to_explain = 10000
make_dir(args.exp_save_path)
make_dir(args.metric_save_path)
if args.dataset == 'imagenet':
    batch_size = 2 # batch size for calculating explanations
    noise_batch_size = 10 # batch size for captum noisetunnel
else:
    batch_size = 5 # batch size for calculating explanations
    noise_batch_size = 64 # batch size for captum noisetunnel

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
nUnif = args.nUnif
std = args.std
seed = args.seed
l2_bound = args.l2_bound
_nsamp = 500 # number of samples for smoothed explanations (e.g. smoothgrad, smoothgbp)
np.random.seed(seed)
torch.manual_seed(seed)
print(device)
inc_auc = {}
inc_auc_curve = {}
pct_include = np.linspace(0,1,51).tolist()


################################################

# #locality debug
glob_output = []
wasg_time = []
npoints = 1000
nfeat_list = (np.exp(np.arange(0,9.25,0.25))*5).astype('int').tolist()
nfeat_list = [54, 784, 3*32*32, 3*224*224]
# for i in tqdm(np.arange(2,500, 20).tolist(), total = len(np.arange(2,500, 20).tolist())):
for i, d in enumerate(tqdm(nfeat_list)):
    randvec = np.random.uniform(-1,1,(npoints,d))
    if args.abs:
        abs = True
        randvec = np.abs(randvec)
    else:
        abs = False
    time_start = time.time()
    glob_output.append(wasserstein_globalness(randvec, args.metric, n_unif=nUnif, reg=0.01, abs = abs, exp_norm = args.exp_norm, seed = seed, scale_diameter = args.scale_diameter, solver = args.solver, centering = args.centering, l2_bound = l2_bound, log = False, normalize_global = normalize_global))
    wasg_time.append(time.time() - time_start)
    print('=================')

output_dict = {
    'time': wasg_time,
    'wasg': glob_output,
    'n_features': nfeat_list,
    'seed': seed,
}
filename =  './Files/Results/time/%s_%s_%s_%s_datasets.pkl' % (args.exp_norm, str(args.nUnif), str(args.seed), args.solver)
make_dir_fromfile(filename)
save_dict(output_dict, filename)
print(filename)
print('done!')