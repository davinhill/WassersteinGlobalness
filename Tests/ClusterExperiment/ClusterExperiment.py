# NEED TO RUN AUC_EXPERIMENT FIRST TO SAVE EXPLANATIONS

# %%
import os
import sys

import skimage.measure

import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

import matplotlib; matplotlib.rc('image', cmap='gray')
import matplotlib.pyplot as plt

import time
import numpy as np

os.chdir('../../')
sys.path.append('./')
from utils.locality_utilities import *
from utils.utils import *
import pickle as pkl

from argparse import ArgumentParser
parser = ArgumentParser(description='AUC Experiment')

parser.add_argument('--dataset', type=str, default = 'cifar10')
parser.add_argument('--nUnif', type=int, default = int(10000))
parser.add_argument('--n_projections', type=int, default = int(500))
parser.add_argument('--seed', type=int, default = 0)
parser.add_argument('--metric', type=str, default = 'euclidean')
parser.add_argument('--exp_save_path', type=str, default = './Files/Results/saved_explanations_1.24_1k')
parser.add_argument('--result_save_path', type=str, default = './Files/Results/cluster_experiment_1.24_1k')
parser.add_argument('--l2_bound', type=float, default = 47.402)
args = parser.parse_args()

##################################################
# Parameters
make_dir(args.result_save_path)
nUnif = args.nUnif
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
seed = args.seed
dataset = args.dataset
np.random.seed(seed)
torch.manual_seed(seed)
print(device)

##################################################
# Load Saved Data
print(f"Loading data...")
filename = os.path.join(args.exp_save_path, '%s_samples_%s.pkl' % (dataset, str(seed)))
sample_dict = load_dict(filename)
test_images, test_labels = sample_dict['test_images'], sample_dict['test_labels']
print('done!')

##################################################
# Load Saved Explanations
explanation_methods = [
    # 'sage',
    # 'deepshap',
    # 'inputxgrad',
    # 'deeplift',
    'smoothgbp',
    'smoothgrad',
    'smoothig',
    # 'randGlobal',
    ]
std_list = [100, 10, 1, 1e-1, 1e-2, 0]
std_list = list(map(float, std_list)) # convert to float

print('loading saved explanations...')
expref = {}
for explainer in explanation_methods:
    for std in std_list:
        if explainer[:6] != 'smooth' and std != 0: # skip std parameter if not a smoothed explanation
            continue
        filename = os.path.join(args.exp_save_path, '%s_%s_%s_%s.pkl' % (dataset, explainer, str(std), str(seed)))
        try:
            output = load_dict(filename)
            expref[explainer + '_%s' % str(std)] = output['explanations'][:1000,...]
        except:
            warnings.warn('File not found: %s' % filename)
            continue

expref['labels'] = test_labels
expref['images'] = test_images
print('done!')

##################################################
# Globalness Calculation

def group_classes(class_list = []):
    tmp_list = []
    for j in class_list:
        tmp_list.append(np.vstack([np.array(separated_exps[j]).reshape(n_per_class, -1)]))
    exp_samples = np.concatenate(tmp_list, axis = 0)
    return exp_samples

def wasg(class_list = []):
    return wasserstein_globalness(group_classes(class_list), n_unif=nUnif, seed = seed, l2_bound = args.l2_bound, log = False, n_projections = args.n_projections)

res_dict = {}
l2_dict = {}
fig, ax = plt.subplots()

# iterate over batches of classes instead of iterating over each class
# if classes_per_batch == 1, will iterate over all classes.
if args.dataset == 'imagenet':
    classes_per_batch = 10
elif args.dataset == 'imagenet1k':
    classes_per_batch = 100
else:
    classes_per_batch = 1

for explainer,class_shap in tqdm(expref.items(), total = len(expref.keys())):
    if explainer == 'labels' or explainer == 'images':
        continue
    print('===========================')
    print(explainer)
    
    # Now separate class_shap according to the labels, and compute globalness
    if classes_per_batch == 1:
        # no label batching
        separated_exps = {label:[] for label in test_labels.tolist()}
        for im, label in zip(class_shap, test_labels.tolist()):
            separated_exps[label].append(im)
    else:
        # batch labels
        batched_labels = torch.arange((test_labels.max()) // classes_per_batch+1)
        separated_exps = {label:[] for label in batched_labels.tolist()}
        for im, label in zip(class_shap, test_labels.tolist()):
            print('========')
            print(label)
            print('mapping: ' + str(label//classes_per_batch))
            separated_exps[label//classes_per_batch].append(im)
        
    # ensure balanced classes
    # filters out the samples in each class based on the min over all classes
    n_per_class = min([len(separated_exps[s]) for s in separated_exps])
    for k in separated_exps.keys():
        separated_exps[k] = separated_exps[k][:n_per_class]
        print(f"{k} has {len(separated_exps[k])} images")

    # sort by key
    separated_exps = dict(sorted(separated_exps.items())) 

    # calculate individual wasg
    wasg_uni = np.zeros((len(separated_exps.keys())))
    for k in separated_exps.keys():
        wasg_uni[k] = wasg([k])
    # sort wasg
    sorted_classes = np.argsort(wasg_uni)[::-1] # sort descending

    # calculate grouped wasg
    globs = []
    l2 = []
    for k in range(1,len(sorted_classes)+1):
        glob = wasg(sorted_classes[:k])
        print(k)
        print(glob)
        globs.append(glob)
        l2.append(np.linalg.norm(center(flatten(group_classes(sorted_classes[:k]), dim = 1)), axis = 1).max())
        res_dict['x_values'] = [(x + 1) * classes_per_batch for x in separated_exps.keys()]
    res_dict[explainer] = globs
    l2_dict[explainer] = l2
    res_dict['x_values'] = [(x + 1) * classes_per_batch for x in separated_exps.keys()]
    ax.plot([x + 1 for x in separated_exps.keys()][2:], globs[2:])
    ax.set_ylabel('Globalness')
    ax.set_xlabel('Number of classes')

plt.legend()
plt.savefig(os.path.join(args.result_save_path, '%s_l2bound%s_clusterExperimentResults.jpg' % (args.dataset, str(args.l2_bound))), dpi = 300, bbox_inches='tight')
res_fname = os.path.join(args.result_save_path, '%s_clusterExperimentResults_l2bound%s_seed%s.pkl' % (args.dataset, str(args.l2_bound), str(args.seed)))
save_dict(res_dict, res_fname)

res_fname = os.path.join(args.result_save_path, '%s_clusterExperimentResults_l2bound%s_seed%s_l2.pkl' % (args.dataset, str(args.l2_bound), str(args.seed)))
save_dict(l2_dict, res_fname)
print(res_fname)
print('done!')