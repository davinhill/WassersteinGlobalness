
import pickle
import numpy as np
import os
import sys
import torch
import sklearn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

def save_dict(dictionary, path):
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

def merge_dict(dict_list):
    '''
    merges a list of dictionaries into a single dictionary
    '''
    output = {}
    for d in dict_list:
        for k, v in d.items():  
            output.setdefault(k, []).append(v)
    return output

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir_fromfile(filepath):
    dirpath = os.path.dirname(filepath)
    make_dir(dirpath)

def tensor2numpy(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()
    return x

def list2cuda(list, cuda = True):
    array = np.array(list)
    return numpy2cuda(array, cuda = cuda)

def numpy2cuda(array, cuda = True):
    tensor = torch.from_numpy(array)
    return tensor2cuda(tensor, cuda = cuda)

def tensor2cuda(tensor, cuda = True):
    if torch.cuda.is_available() and cuda:
        tensor = tensor.cuda()
    return tensor

def load_model(path):
    tmp = os.path.dirname(os.path.abspath(path))
    sys.path.append(tmp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = torch.load(path, map_location=device)

    return model

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return min(lr)


def iAUC(torch_sample, ref_sample, attributions, model, pct_mask = np.linspace(0,1,20).tolist()):
    '''
    calculates iAUC for a given sample and feature ranking.

    args:
        torch_sample: tokenized sample
        attributions: np vector of length d of feature importance values
        model: pytorch model
        pct_mask: list of percentages to mask

    return:
        output_vector: np vector of length pct_mask; model output with :i features added.
    '''
    attributions = np.abs(attributions)
    ranking = np.argsort(attributions)


    output_vector = []
    num_feats = attributions.shape[1]
    for k in pct_mask:
        n_mask = min(int(num_feats * k), num_feats) # number of features to mask
        if n_mask>0:
            for idx in ranking[-n_mask:]:
                ref_sample[:,idx] = torch_sample[:,idx]

        prediction = model.predict(ref_sample)
        output_vector.append(prediction[0].item())

    return np.array(output_vector)


def exAUC(torch_samples, attributions, ref_samples, model, pct_exclude = np.linspace(0,1,21).tolist()):
    '''
    '''

    assert torch_samples.shape[0] == attributions.shape[0] and attributions.shape[0] == ref_samples.shape[0], 'torch_samples, attributions, and ref_samples must have the same number of samples (dim 0 must be the same size)'
    # get labels, i.e. model prediction with all features
    dataloader = create_dataloader(data = torch_samples, labels = None)
    labels = batch_predictions(model, dataloader)

    # preprocessing: flatten samples and attributions
    orig_shape = torch_samples.shape
    torch_samples = torch.flatten(torch_samples, 1).cpu()
    ref_samples = torch.flatten(ref_samples, 1).cpu()
    if type(attributions) == torch.Tensor: attributions = tensor2numpy(attributions)
    attributions = np.abs(attributions)
    attributions = attributions.reshape(orig_shape[0], -1)

    #iteratively mask features according to attribution ranking
    output_list = []
    for j, k in tqdm(enumerate(pct_exclude), total = (len(pct_exclude)+1)):

        num_feats = attributions.shape[1]
        n_exclude = min(int(num_feats * k), num_feats) # number of features to mask
        torch_samples_tmp = torch_samples.clone().cpu()
        for i, (torch_sample, attribution) in enumerate(zip(torch_samples, attributions)):
            ranking = np.argsort(np.abs(attribution))
            
            if n_exclude == 0:
                continue # original sample only
            else:
                torch_samples_tmp[i, ranking[-n_exclude:]] = ref_samples[i, ranking[-n_exclude:]]

        dataloader = create_dataloader(torch_samples_tmp.reshape(orig_shape), labels)
        output_list.append(calc_test_accy(model, dataloader))
            
    output_list = np.array(output_list)
    auc = sklearn.metrics.auc(x = pct_exclude, y = output_list)
    return auc, output_list

def incAUC(torch_samples, attributions, ref_samples, model, abs = True, pct_include = np.linspace(0,1,21).tolist()):
    '''
    Inclusion AUC metric (Jethani, Neil, et al. "Fastshap: Real-time shapley value estimation." ICLR 2022 (2022)) for evaluating the effects of feature permutation.

    args:
        torch_samples (n x d tensor): data samples to evaluate
        attributions (n x d tensor): feature attributions for the data samples.
        ref_samples (n x d tensor): reference samples that represent samples where all features are masked.
        model (pytorch model): pytorch model that takes the torch samples as input and ouputs a scalar (e.g. probability of a class).
        pct_include (optional, list): list of percentages of features to mask. If not provided, algorithm will iteratively mask every feature.
        abs (bool): take absolute values of attributions (does not affect order_attributions)
        
    returns:
        auc (float): Inclusion AUC
        output_list (np array): model outputs to plot AUC curve
    '''
    assert torch_samples.shape[0] == attributions.shape[0] and attributions.shape[0] == ref_samples.shape[0], 'torch_samples, attributions, and ref_samples must have the same number of samples (dim 0 must be the same size)'

    # get labels, i.e. model prediction with all features
    dataloader = create_dataloader(data = torch_samples, labels = None)
    labels = batch_predictions(model, dataloader)

    # flatten samples and attributions
    orig_shape = torch_samples.shape
    torch_samples = torch.flatten(torch_samples, 1).cpu()
    ref_samples = torch.flatten(ref_samples, 1).cpu()
    if type(attributions) == np.ndarray:
        attributions = np.abs(attributions)
        attributions = flatten(attributions, 1)
    else:
        attributions = torch.abs(attributions)
        attributions = torch.flatten(attributions, 1)

    #iteratively mask features according to attribution ranking
    output_list = []
    for j, k in tqdm(enumerate(pct_include), total = (len(pct_include)+1)):

        num_feats = attributions.shape[1]
        n_include = min(int(num_feats * k), num_feats) # number of features to mask
        ref_sample_tmp = ref_samples.clone().cpu()
        for i, (torch_sample, attribution) in enumerate(zip(torch_samples, attributions)):
            ranking = np.argsort(np.abs(attribution))
            
            if n_include == 0:
                continue # reference sample only
            elif n_include == num_feats:
                ref_sample_tmp[i,:] = torch_sample
            else:
                ref_sample_tmp[i, ranking[-n_include:]] = torch_sample[ranking[-n_include:]]

        dataloader = create_dataloader(ref_sample_tmp.reshape(orig_shape), labels)
        output_list.append(calc_test_accy(model, dataloader))
            
    output_list = np.array(output_list)
    auc = sklearn.metrics.auc(x = pct_include, y = output_list)
    return auc, output_list

def create_dataloader(data, labels, batch_size = 64):
    dataset = plain_dataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader

class plain_dataset(Dataset):
    def __init__(self, data, labels):
        """
        args:
            samples: torch tensor of samples
        """
        self.data = data
        if labels is None:
            self.labels = torch.zeros(self.data.shape[0])
        else:
            self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx,...], self.labels[idx]

def calc_test_accy(model, test_loader):
    model.eval()   # Set model into evaluation mode
    correct = 0
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = tensor2cuda(data), tensor2cuda(target)
            output = model(data)   # Calculate Output
            try:
                dim_output = output.shape[1]
            except IndexError:
                dim_output = 1
            if dim_output == 1:
                pred = (output>=0)*1
            else:
                pred = output.max(1, keepdim=True)[1]  # Calculate Predictions

            correct += pred.eq(target.view_as(pred)).sum().item()
        return (100.*correct/len(test_loader.dataset))

def batch_predictions(model, test_loader):
    model.eval()   # Set model into evaluation mode
    pred_list = []
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = tensor2cuda(data), tensor2cuda(target)
            output = model(data)   # Calculate Output
            try:
                dim_output = output.shape[1]
            except IndexError:
                dim_output = 1
            if dim_output == 1:
                pred = (output>=0)*1
            else:
                pred = output.max(1, keepdim=True)[1]  # Calculate Predictions

            pred_list.append(pred.detach().cpu())
    
    return torch.cat(pred_list).reshape(-1)

