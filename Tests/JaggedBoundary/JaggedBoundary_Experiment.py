# %%
import os
import sys
os.chdir('/work/jdy/davin/WasG/')
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.locality_utilities import *

# classifiers
from sklearn.neighbors import KNeighborsClassifier  

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import pickle
import shap
from datetime import datetime
import sklearn


from argparse import ArgumentParser
parser = ArgumentParser(description='AUC Experiment')

# Wasserstein Globalness Parameters
parser.add_argument('--nUnif', type=int, default = int(4e5))
parser.add_argument('--abs', type=int, default = 0)

# Jagged Boundary Parameters
parser.add_argument('--seed', type=int, default = 0)
parser.add_argument('--scale', type=float, default = 0)
parser.add_argument('--dimension', type=int, default = 2)
args = parser.parse_args()

# %%
# Define the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class synthNet(nn.Module):
    def __init__(self,d_in, d_out, hidden_layer_sizes=(300,300,300)):
        super().__init__()
        
        w1, w2, w3 = hidden_layer_sizes
        
        self.top = nn.Sequential(
            nn.Linear(d_in, w1),
            nn.ReLU(),
            nn.Linear(w1, w2),
            nn.ReLU(),
            nn.Linear(w2, w2),
            nn.ReLU(),
            nn.Linear(w2, w2),
            nn.ReLU(),
            nn.Linear(w2, w2),
            nn.ReLU(),
            nn.Linear(w2, w3),
            nn.ReLU(),
            nn.Linear(w3, d_out))
        
    def forward(self, x):
        
        if type(x) == list:
            x = np.array(x)
        
        if not torch.is_tensor(x):
            x = torch.Tensor(x).to(device)
        
        out = self.top(x)
        if len(out.shape) == 1:
            out = out.reshape(1,-1)
        return(out)
    
    def predict(self, x):
        return(self.forward(x).argmax(dim=1))
    
    def predict_proba(self, x):
        return(self.forward(x).cpu().detach().numpy())
    
    def predict_flattened_softmax(self, x):
        return F.softmax(self.forward(x), dim = 1).cpu().detach().numpy()
        
    def fit(self, X_train, y_train, optim='ADAM', n_epochs=20, batch_size=256, _lr=0.005):
        
        if not torch.is_tensor(X_train): X_train = torch.Tensor(X_train).to(device)
        if not torch.is_tensor(y_train): y_train = torch.Tensor(y_train)
        dataset = TensorDataset(X_train, y_train)
        trainloader = DataLoader(dataset, batch_size=batch_size)
        
        criterion = nn.CrossEntropyLoss()
        if optim.upper() == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=_lr, momentum=0.9)
        elif optim.upper() == 'ADAM':
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        # Training
        # losses = []
        for epoch in range(n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # inputs, labels = data
                inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def attribution_to_selection(attributions, k = 1):
    '''
    only works for k = 1 for now
    '''
    exps = np.argmax(attributions, axis = 1)
    exps = one_hot(exps, 2)
    return exps

def tensor2numpy(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()
    return x

def list2cuda(list):
    array = np.array(list)
    return numpy2cuda(array)

def numpy2cuda(array):
    tensor = torch.from_numpy(array)
    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

explainers = [
    'sage',
    # 'deepshap',
    # 'vanillagrad',
    # 'guidedbackprop',
    # 'integratedgradients',
    # 'ground_truth'
    ]

# std_list = [10, 1, 1e-1, 1e-2] # last one is used in cluster experiment plots. default is 0.15
# std_list = [10, 1] # last one is used in cluster experiment plots. default is 0.15
# explainers = explainers + ['smoothgrad_%s' % str(std) for std in std_list]
# explainers = explainers + ['smoothgbp_%s' % str(std) for std in std_list]
classifier = 'nn'

###########################################3
# Parameters
n, d = 3000, args.dimension
# n_bg = int(d * 1e3)
selection_method = 'auto'
jagged_boundary_threshold = args.scale
_scale = 0.3  # data scale
seed = args.seed
num_features_to_select = 1
split_params = {'test_size' : 0.5, 'random_state' : 0}
save_path = './Files/Results/JaggedBoundary/scale%s_seed%s_d%s_gt.pkl' % (str(jagged_boundary_threshold), str(seed), str(d))

# %%
###########################################3
# GENERATE DATA
# Set seed for data perturbation and model training
np.random.seed(seed)
torch.manual_seed(seed)
all_accs, all_globs, all_scores = {}, {}, {}
# X_bg, y_bg, src_bg = hyperplane_data(n_bg, d, _scale = _scale);
X,y, src_labels = hyperplane_data(n, d, _scale=_scale, logits=False)  # changed logits to False when I changed model to custom pytorch (instead of scikit)


accs, globs, scores = [], [], []
n,d = X.shape
X, y = fuzz_boundary(X, y, thresh=jagged_boundary_threshold) 
X, y = torch.Tensor(X).to(device), torch.Tensor(y).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)
# X_train, X_test = scaler.fit_transform(X_train, X_test)
src_train, src_test = train_test_split(src_labels, **split_params)

###########################################3
# Train Model
if classifier.lower() == 'knn':
    clf = KNeighborsClassifier(n_neighbors=3)
elif classifier.lower() == 'svm':
    clf = sklearn.svm.SVC(kernel='rbf', probability=True)
elif classifier.lower() == 'nn':
    clf = synthNet(d, 2, hidden_layer_sizes=(300,500,300)).to(device)
    clf.device = device
    
print(f"Training...")
clf.fit(X_train, y_train)  # use encoded version of y_train
print(f"Done training!")

###########################################3
# Generate Explanations
output_df = pd.DataFrame()
for explainer_choice in explainers:
    output_dict = {}
    output_dict['seed'] = args.seed
    output_dict['scale'] = args.scale
    output_dict['explainer'] = explainer_choice
    output_dict['data_dim'] = args.dimension
    
    print(f"Explaining...")
    # Explain
    test_images = X_test
    model = clf
    test_labels = np.ones_like(y_test) # just used for indicating which class to calculate explanations for

    if explainer_choice == 'sage':
        import sage
        print('SAGE...')
        # model_activation = nn.Sequential(model, nn.Softmax(dim=1)) # add softmax at output
        sample_shape = test_images[0:1,...].shape
        num_feats = torch.flatten(test_images, 1).shape[1]
        model_activation = model.predict_flattened_softmax
        # imputer = sage.MarginalImputer(model_activation, X_train[:50,:])
        imputer = sage.DefaultImputer(model_activation, np.zeros(num_feats))
        estimator = sage.PermutationEstimator(imputer, 'cross entropy')

        sage_values = estimator(tensor2numpy(torch.flatten(test_images, 1)), np.ones_like(test_labels), batch_size=512, thresh=0.05)

        sage_values = sage_values.values.reshape(sample_shape) # reshape to sample shape
        sage_values = sage_values.repeat(test_images.shape[0], axis = 0) # copy explanations for all samples

        exps = attribution_to_selection(sage_values) # convert to feature selection
        exp_prediction = np.argmax(exps, axis = 1) # predicted top feature

    elif explainer_choice == 'deepshap':
        # deepshap
        print('deepshap...')
        background = torch.zeros((5,2)).to(device)
        e = shap.DeepExplainer(model, background)
        time_start = time.time()
        e_fn = lambda x : e.shap_values(x)

        explainer_time = time.time() - time_start
        exps = e_fn(test_images)
        deepshaps = exps[1] # attributions w.r.t. class 1

        exps = attribution_to_selection(deepshaps) # convert to feature selection
        exp_prediction = np.argmax(exps, axis = 1) # predicted top feature

    elif explainer_choice[:10] == 'smoothgrad':
        std = float(explainer_choice.split('_')[1])
        #################################################
        print('smoothgrad %s...' % str(std))
        e_fn = exp_smoothGrad
        # avg pool
        time_start = time.time()
        if len(test_images.shape) >2:
            image = True
        else:
            image = False
        exps = e_fn(test_images, model, targets=numpy2cuda(test_labels), std = std, _nsamp = 500, image = image)

        explainer_time = time.time() - time_start
        print(np.shape(exps))
        smoothmaps_hires = np.vstack([np.expand_dims(im, 0) for im in exps])
        exps = attribution_to_selection(smoothmaps_hires) # convert to feature selection
        exp_prediction = np.argmax(exps, axis = 1) # predicted top feature

    elif explainer_choice[:9] == 'smoothgbp':
        std = float(explainer_choice.split('_')[1])
        print('smoothGBP %s...' % str(std))
        # guided smoothgrad
        e_fn = exp_guidedsmoothGrad
        image = False
        time_start = time.time()
        exps = e_fn(test_images, model, std = std, _nsamp = 500, image = image)
        explainer_time = time.time() - time_start
        print(np.shape(exps))
        gsgs_hires = np.vstack([np.expand_dims(im, 0) for im in exps])
        exps = attribution_to_selection(gsgs_hires) # convert to feature selection
        exp_prediction = np.argmax(exps, axis = 1) # predicted top feature

    elif explainer_choice == 'vanillagrad':
        print('vanillagrad...')
        e_fn = exp_vanillaGrad
        time_start = time.time()
        exps = e_fn(test_images, model, targets=numpy2cuda(test_labels))

        explainer_time = time.time() - time_start
        print(np.shape(exps))
        gradmaps_hires = np.vstack([np.expand_dims(im, 0) for im in exps])
        exps = attribution_to_selection(gradmaps_hires) # convert to feature selection
        exp_prediction = np.argmax(exps, axis = 1) # predicted top feature

    elif explainer_choice == 'ground_truth':
        print('ground truth...')
        exp_prediction = src_test.reshape(-1)
        exps = one_hot(src_test.astype('int'), 2)

    elif explainer_choice == 'guidedbackprop':
        print('guided backprop...')
        e_fn = exp_guidedbackprop
        image = False
        time_start = time.time()
        exps = e_fn(test_images, model, image = image)

        explainer_time = time.time() - time_start
        print(np.shape(exps))
        gbpmaps_hires = np.vstack([np.expand_dims(im, 0) for im in exps]) 
        exps = attribution_to_selection(gbpmaps_hires) # convert to feature selection
        exp_prediction = np.argmax(exps, axis = 1) # predicted top feature

    elif explainer_choice == 'integratedgradients':
        print('integrated gradients...')
        e_fn = exp_intGrad
        time_start = time.time()
        exps = e_fn(test_images, model)
        explainer_time = time.time() - time_start
        print(np.shape(exps))
        intgrads_hires = np.vstack([np.expand_dims(im, 0) for im in exps])
        exps = attribution_to_selection(intgrads_hires) # convert to feature selection
        exp_prediction = np.argmax(exps, axis = 1) # predicted top feature


    ###########################################3
    # Calculate Metrics
    output_dict['accy'] = sklearn.metrics.accuracy_score(src_test, exp_prediction)
    output_dict['f1'] = sklearn.metrics.f1_score(src_test, exp_prediction)
    output_dict['precision'] = sklearn.metrics.precision_score(src_test, exp_prediction)
    output_dict['recall'] = sklearn.metrics.recall_score(src_test, exp_prediction)
        
    ###########################################3
    # Wasserstein Globalness
    # glob = wasserstein_globalness(exps, 'hamming', n_unif=args.nUnif, seed = seed, l2_bound = np.sqrt(2))
    glob = wasserstein_globalness(exps, 'hamming', n_unif=args.nUnif, seed = seed, solver = 'pot', normalize_global = False)
    output_dict['wasg'] = glob

    
    ###########################################3
    # Save Results
    tmp = pd.DataFrame([output_dict])
    output_df = output_df.append(tmp, ignore_index = True)

output_df.to_pickle(save_path)
print(save_path)
print('done!')