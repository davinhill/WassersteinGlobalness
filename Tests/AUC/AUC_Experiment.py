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



from argparse import ArgumentParser
parser = ArgumentParser(description='AUC Experiment')

parser.add_argument('--explainer', type=str, default = 'inputxgrad')
parser.add_argument('--dataset', type=str, default = 'imagenet1k')
parser.add_argument('--exp_norm', type=str, default = 'sphere_l2_bound')
parser.add_argument('--nUnif', type=int, default = 5000)
parser.add_argument('--abs', type=int, default = 0)
parser.add_argument('--calc_exp', type=int, default = 1)
parser.add_argument('--seed', type=int, default = 0)
parser.add_argument('--metric', type=str, default = 'euclidean')
parser.add_argument('--scale_diameter', type=str, default = 'None')
parser.add_argument('--solver', type=str, default = 'pot_slice_symmetric')
parser.add_argument('--centering', type=str, default = 'mean')
parser.add_argument('--l2_bound', type=float, default = 1)
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
if args.dataset == 'imagenet' or args.dataset == 'imagenet1k':
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

# %%

if args.dataset == 'mnist':
    path_data = './Files/Data'
    path_model = './Files/Models/MLP_baseline.pt'

    from BlackBox_Models.MNIST.train_MNIST import NN_MNIST, load_data
    _, _, test_set, test_loader = load_data(path_data, batch_size = n_samples_to_explain)
    # _, _, _, bgloader = load_data(path_data, batch_size = n_bg)
    model_params = {
        'input_units':28*28,
        'num_classes':10,
        'hidden_units':500,
    }
    model = NN_MNIST(**model_params)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.to(device)
    model.eval()

    # reference sample for integrated gradients
    # norm = torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081], inplace=False)
    # ref_sample = tensor2cuda(norm(torch.zeros((1,1,28,28))))
    ref_sample = tensor2cuda((torch.zeros((1,1,28,28))))

elif args.dataset == 'cifar10':
    path_data = './Files/Data'
    path_model = './Files/Models/CIFAR10_baseline.pt'

    from BlackBox_Models.CIFAR10.CIFAR10_baseline import load_data, custom_resnet
    from torchvision import models
    _, _, test_set, test_loader = load_data(path_data, batch_size = n_samples_to_explain)
    # _, _, _, bgloader = load_data(path_data, batch_size = n_bg)
    model = custom_resnet(input_shape = (3,32,32), num_classes = 10, model = 'resnet18')
    model.model.load_state_dict(torch.load(path_model, map_location=device))
    model.to(device)
    model.eval()

    # reference sample for integrated gradients
    # norm = torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    # ref_sample = tensor2cuda(norm(torch.zeros((1,3,32,32))))
    ref_sample = tensor2cuda((torch.zeros((1,3,32,32))))

elif args.dataset == 'divorce':
    path_data = './Files/Data/divorce.h5'
    path_model = './Files/Models/divorce_model.pt'
    scaler = load_dict('./Files/Data/divorce_scaler.pkl')

    from BlackBox_Models.divorce.train_divorce import load_data, NN
    from torchvision import models
    _, _, test_set, test_loader = load_data(path_data, batch_size = n_samples_to_explain, scaler = scaler)
    # _, _, _, bgloader = load_data(path_data, batch_size = n_bg, scaler = scaler)
    model = NN(input_units = 54, hidden_units = 50, num_classes = 1)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.to(device)
    model.eval()

    # reference sample for integrated gradients
    ref_sample = tensor2cuda(torch.zeros((1,54)))

elif args.dataset == 'nhanes':
    path_data = './Files/Data/'
    scaler = load_dict('./Files/Data/divorce_scaler.pkl')

    from BlackBox_Models.nhanes.train_nhanes import load_data
    _, _, test_set, test_loader = load_data(path_data, batch_size = n_samples_to_explain)

    saved_filepath = './Files/Models/nhanes_tabnet.zip'
    sys.path.append('./tabnet')
    sys.path.append('./tabnet/pytorch_tabnet')
    from tab_model import TabNetClassifier
    model = TabNetClassifier()
    model.load_model(saved_filepath)
    model = model.network
    # print(model.network(test_set[0:2][0]))

    # reference sample for integrated gradients
    ref_sample = tensor2cuda(torch.zeros((1,27)))

elif args.dataset == 'spiro':
    rejection_reason = 'COUGHING'
    target = 'fev1.best'
    path_data = './Files/Data/UKBB_Spirometry_%s.pkl' % rejection_reason
    path_model = './Files/Models/best_model_visionary-shadow-37.pt'

    sys.path.append('./BlackBox_Models/spiro/')
    from BlackBox_Models.spiro.load_data import load_data
    train_loader, test_loader = load_data(path = path_data, target = target, batch_size = 512, scale = True)
    test_set = test_loader.dataset
    model = torch.load(path_model, map_location=device)
    model.to(device)
    model.eval()

    # reference sample for integrated gradients
    ref_sample = tensor2cuda(torch.zeros((1,300)))

elif args.dataset == 'copd':
    path_model = './Files/Models/COPD_model.pt'
    path_data = './Files/Data/COPD.h5'
    path_scaler = './Files/Data/copd_scaler.pkl'
    scaler = load_dict(path_scaler)

    sys.path.append('./BlackBox_Models/copd/')
    from BlackBox_Models.COPD.model import COPD, NN
    test_set = COPD(data_path = path_data, train = False, scaler = scaler)
    test_loader = DataLoader(test_set, batch_size = 128, shuffle = False, num_workers = 0)
    checkpoint = torch.load(path_model)
    model = NN(**checkpoint['params'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # reference sample for integrated gradients
    ref_sample = tensor2cuda(torch.zeros((1,1079)))

elif args.dataset == 'imagenet':

    path_data = './BlackBox_Models/INet200/data'
    path_model = './BlackBox_Models/INet200/best.ckpt'

    sys.path.append('./BlackBox_Models/INet200/')
    from openood.networks import ResNet18_224x224
    from openood.evaluation_api.datasets import get_id_ood_dataloader
    from openood.evaluation_api.preprocessor import get_default_preprocessor

    model = ResNet18_224x224(num_classes=200)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.to(device)
    model.eval()

    loader_kwargs = {
        'batch_size': n_samples_to_explain,
        'shuffle': True,
        'num_workers': 1
    }
    preprocessor = get_default_preprocessor('imagenet200')
    dataloader_dict = get_id_ood_dataloader(id_name = 'imagenet200', data_root  = path_data, preprocessor = preprocessor, **loader_kwargs)

    test_loader  = dataloader_dict['id']['test']

    ref_sample = tensor2cuda((torch.zeros((1,3,224,224))))
    

elif args.dataset == 'imagenet1k':

        


    train_path = '/work/jdy/datasets/imagenet/val'
    from torchvision.models import resnet50, ResNet50_Weights
    from torch.utils.data import RandomSampler
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.to(device)
    model.eval()
    transform = weights.transforms()

    imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
    sampler = RandomSampler(imagenet_data, replacement=True, num_samples=n_samples_to_explain)
    test_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=n_samples_to_explain,
        # shuffle=True,
        num_workers=1,
        sampler=sampler
    )
    test_set = test_loader.dataset

    ref_sample = tensor2cuda((torch.zeros((1,3,224,224))))


print(f"Loading data...")
# %% 

# Draw Random Sample
if args.dataset == 'imagenet':
    data = next(iter(test_loader))
    test_images, test_labels = data['data'], data['label']

elif args.dataset == 'imagenet1k':
    test_images, test_labels = next(iter(test_loader))
    test_labels = test_labels.type(torch.int64)

else:
    from torch.utils.data import RandomSampler
    sampler = RandomSampler(test_set, replacement=True, num_samples=n_samples_to_explain)
    test_loader =  DataLoader(test_set, sampler=sampler, batch_size=100000)
    test_images, test_labels = next(iter(test_loader))
    test_labels = test_labels.type(torch.int64)

# save samples for reference
if args.calc_exp:
    output = {'test_images': test_images, 'test_labels': test_labels}
    save_dict(output, os.path.join(args.exp_save_path, '%s_samples_%s.pkl' % (args.dataset, str(args.seed))))

# %%
if args.calc_exp:
##########################
# Calculate Explanations
    if args.explainer == 'sage':
        #################################################
        # SAGE
        import sage
        print('SAGE...')
        # model_activation = nn.Sequential(model, nn.Softmax(dim=1)) # add softmax at output
        sample_shape = test_images[0:1,...].shape
        num_feats = torch.flatten(test_images, 1).shape[1]

        # model_activation = model.predict_flattened_softmax
        model_activation = softmax_wrapper(model, test_images).flattened_softmax # output of model should be probability

        imputer = sage.DefaultImputer(model_activation, tensor2numpy(ref_sample.reshape(-1)))
        estimator = sage.PermutationEstimator(imputer, 'cross entropy')

        time_start = time.time()

        sage_values = estimator(tensor2numpy(torch.flatten(test_images, 1)), tensor2numpy(test_labels), batch_size=batch_size, thresh=0.05)

        explainer_time = time.time() - time_start

        sage_values = sage_values.values.reshape(sample_shape) # reshape to sample shape
        sage_values = sage_values.repeat(test_images.shape[0], axis = 0) # copy explanations for all samples
        print(np.shape(sage_values))

        # copy naming convention for other methods
        exps = sage_values.copy()

    elif args.explainer == 'deepshap':
        print('deepshap...')
        from captum.attr import DeepLiftShap
        e_fn = DeepLiftShap(model, multiply_by_inputs=False)
        time_start = time.time()

        # if multiclass output, use attributions w.r.t. label
        output_shape = model(test_images[:2,...].to(device)).shape
        if len(output_shape) == 1:
            target = None
        elif output_shape[1] == 1:
            target = None
        else:
            target = test_labels
            
        background = torch.zeros_like(test_images)[:5,...] # datasets are normalized with mean = 0 

        exps = batch_explanations(e_fn.attribute,batch_size = batch_size, inputs = test_images, baselines = tensor2cuda(background), target = target)
        # exps = e_fn.attribute(tensor2cuda(test_images), baselines = tensor2cuda(background), target = tensor2cuda(target))
        explainer_time = time.time() - time_start
        exps = tensor2numpy(exps)

    elif args.explainer == 'deeplift':
        print('deeplift...')
        from captum.attr import DeepLift
        e_fn = DeepLift(model)
        time_start = time.time()

        # if multiclass output, use attributions w.r.t. label
        output_shape = model(test_images[:2,...].to(device)).shape
        if len(output_shape) == 1:
            target = None
        elif output_shape[1] == 1:
            target = None
        else:
            target = test_labels
            
        exps = batch_explanations(e_fn.attribute,batch_size = batch_size, inputs = test_images, baselines = tensor2cuda(ref_sample), target = target)
        # exps = e_fn.attribute(tensor2cuda(test_images), baselines = tensor2cuda(ref_sample), target = tensor2cuda(target))
        explainer_time = time.time() - time_start
        exps = tensor2numpy(exps)

    elif args.explainer == 'inputxgrad':
        print('inputxgrad...')
        from captum.attr import InputXGradient
        e_fn = InputXGradient(model)
        time_start = time.time()

        # if multiclass output, use attributions w.r.t. label
        output_shape = model(test_images[:2,...].to(device)).shape
        if len(output_shape) == 1:
            target = None
        elif output_shape[1] == 1:
            target = None
        else:
            target = test_labels
            
        exps = batch_explanations(e_fn.attribute,batch_size = batch_size, inputs = test_images, target = target)
        # exps = e_fn.attribute(tensor2cuda(test_images), target = tensor2cuda(target))
        explainer_time = time.time() - time_start
        exps = tensor2numpy(exps)

    #################################################
    elif args.explainer == 'smoothgrad':
        from captum.attr import NoiseTunnel
        from captum.attr import Saliency

        # if multiclass output, use attributions w.r.t. label
        output_shape = model(test_images[:2,...].to(device)).shape
        if len(output_shape) == 1:
            target = None
        elif output_shape[1] == 1:
            target = None
        else:
            target = test_labels
            
        print('smoothgrad %s...' % str(std))
        time_start = time.time()
        if std == 0:
            e_fn = Saliency(model)
            exps = batch_explanations(e_fn.attribute,batch_size = batch_size, inputs = test_images, target = target)
        else:
            e_fn = NoiseTunnel(Saliency(model))
            exps = batch_explanations(e_fn.attribute,batch_size = batch_size, inputs = test_images, target = target, nt_samples = _nsamp, nt_samples_batch_size = noise_batch_size, stdevs = float(std), nt_type = 'smoothgrad')
        explainer_time = time.time() - time_start
        exps = tensor2numpy(exps)

    #################################################
    elif args.explainer == 'smoothgbp':
        print('smoothGBP %s...' % str(std))
        # guided smoothgrad
        e_fn = exp_guidedsmoothGrad
        if len(test_images.shape) >2:
            image = True
        else:
            image = False
        time_start = time.time()

        if std == 0: _nsamp = 1
        exps = batch_explanations(e_fn, inputs = test_images, model = model, std = std, _nsamp = _nsamp, image = image, return_tensor = True)

        # exps = e_fn(test_images, model, std = std, _nsamp = _nsamp, image = image)

        explainer_time = time.time() - time_start
        print(np.shape(exps))
        exps = np.vstack([np.expand_dims(im, 0) for im in exps])

    #################################################
    elif args.explainer == 'smoothig':
        from captum.attr import NoiseTunnel
        from captum.attr import IntegratedGradients

        print('smoothig %s...' % str(std))

        # if multiclass output, use attributions w.r.t. label
        output_shape = model(test_images[:2,...].to(device)).shape
        if len(output_shape) == 1:
            target = None
        elif output_shape[1] == 1:
            target = None
        else:
            target = test_labels
            
        time_start = time.time()
        if std == 0:
            e_fn = IntegratedGradients(model, multiply_by_inputs=False)
            exps = batch_explanations(e_fn.attribute,batch_size = batch_size, inputs = test_images, baselines = tensor2cuda(ref_sample), target = target, internal_batch_size = batch_size)
        else:
            e_fn = NoiseTunnel(IntegratedGradients(model, multiply_by_inputs=False))
            exps = batch_explanations(e_fn.attribute,batch_size = batch_size, inputs = test_images, baselines = tensor2cuda(ref_sample), target = target, nt_samples = _nsamp, nt_samples_batch_size = noise_batch_size, stdevs = float(std), internal_batch_size = batch_size)
        # exps = e_fn.attribute(tensor2cuda(test_images), baselines = tensor2cuda(ref_sample), target = tensor2cuda(target))
        explainer_time = time.time() - time_start
        exps = tensor2numpy(exps)

    #################################################
    elif args.explainer == 'randGlobal':
        # Random Global
        print('Random Global...')
        time_start = time.time()
        exps = np.random.uniform(-1,1, size = (test_images[0:1,...]).shape).repeat(test_images.shape[0], axis = 0) # random vector
        explainer_time = time.time() - time_start
        
    #################################################
    elif args.explainer == 'randLocal':
        print('Random Local...')
        d = test_images[0,...].reshape(-1).shape[0]
        npoints = test_images.shape[0]
        time_start = time.time()
        # randvec = sample_spherical(npoints, d, quadrant = None)
        exps = tensor2numpy(sample_sphere_interior(d = d, n1 = npoints, r =  l2_bound))
        explainer_time = time.time() - time_start

    # save explanations
    output = {
    'explainer_time': explainer_time,
    'explanations': exps,
    'explainer': args.explainer,
    'std': args.std,
    'dataset': args.dataset,
    'l2_norm': np.linalg.norm(center(flatten(exps, dim = 1)), ord = 2, axis = 1).max(),
    }
    # change this
    # save_dict(output, '/work/jdy/davin/WasG/Files/Results/explanations_for_visualization/%s_%s_%s_%s' % (args.dataset, args.explainer, str(std), str(seed)))
    filename = os.path.join(args.exp_save_path, '%s_%s_%s_%s.pkl' % (args.dataset, args.explainer, str(std), str(seed)))
    save_dict(output, filename)
    print(filename)
    sys.exit()


##########################################
# Load Explanations
else:
    print('loading saved explanations...')
    filename = os.path.join(args.exp_save_path, '%s_%s_%s_%s.pkl' % (args.dataset, args.explainer, str(std), str(seed)))
    output = load_dict(filename)

    explainer_time = output['explainer_time']
    exps = output['explanations']
    print('done!')
    if args.n_samples_to_explain < exps.shape[0]:
        exps = exps[:args.n_samples_to_explain,...]

##########################################
# Wasserstein Globalness
if args.abs:
    abs = True
    exps = np.abs(exps)
else:
    abs = False
print(np.shape(exps))

time_start = time.time()
exps = exps.reshape(exps.shape[0], -1)
glob, log_dict = wasserstein_globalness(exps, args.metric, n_unif=nUnif, reg=0.01, abs = abs, exp_norm = args.exp_norm, seed = seed, scale_diameter = args.scale_diameter, solver = args.solver, centering = args.centering, l2_bound = l2_bound, log = True, normalize_global = normalize_global)
wasg_time = time.time() - time_start

##########################################
# Metrics: Exclusion AUC
input_shape = list(test_images.shape)
input_shape[1:] = [-1]*len(input_shape[1:])
auc_exclude,auc_curve_exclude = exAUC(
    torch_samples = test_images,
    attributions = exps,
    ref_samples = ref_sample.expand(input_shape),
    model = model,
    pct_exclude = pct_include,
)

##########################################
# Metrics: Infidelity

calc_infidelity = False
if calc_infidelity:
    sys.path.append('./saliency_evaluation/')
    from infid_sen_utils import get_exp_infid

    max_samples = 50 # maximum number of samples for which to calculate infidelity
    if test_images.shape[0] > max_samples:
        # downsample
        idx = np.random.choice(test_images.shape[0], size = max_samples, replace = False)
        tmp_images, tmp_labels, tmp_exps = test_images[idx,...], test_labels[idx], exps[idx,...]
    else:
        tmp_images, tmp_labels, tmp_exps = test_images, test_labels, exps


    if args.dataset in ['divorce', 'copd']:
        # 1d output
        model_activation = softmax_wrapper(model, tmp_images).softmax_sc2mc
    elif args.dataset == 'spiro':
        model_activation = model
        tmp_labels = torch.zeros_like(tmp_labels) # dummy labels for regression
    else:
        model_activation = softmax_wrapper(model, tmp_images).softmax
    infid_list = []
    for i in tqdm(range(tmp_images.shape[0])):
        pdt = tensor2numpy(model_activation(ref_sample))[0,int(tmp_labels[i].item())]

        infid_list.append(get_exp_infid(tmp_images[i:i+1,...], model_activation, tmp_exps[i:i+1,...], tmp_labels[i].int(), pdt, binary_I=False, pert = 'Gaussian'))
    infid_list = np.array(infid_list)
    infid = infid_list.mean()



##########################################
# Metrics: Inclusion AUC
input_shape = list(test_images.shape)
input_shape[1:] = [-1]*len(input_shape[1:])
auc_include,auc_curve_include = incAUC(
    torch_samples = test_images,
    attributions = exps,
    ref_samples = ref_sample.expand(input_shape),
    model = model,
    pct_include = pct_include,
)

##########################################
# Metrics: Exclusion AUC
auc_exclude,auc_curve_exclude = exAUC(
    torch_samples = test_images,
    attributions = exps,
    ref_samples = ref_sample.expand(input_shape),
    model = model,
    pct_exclude = pct_include,
)


output_dict = {}
output_dict['dataset'] = args.dataset
output_dict['explainer'] = args.explainer
output_dict['std'] = args.std
output_dict['wasg'] = glob
output_dict['unnorm_wasg'] = log_dict['unnorm_wasg']
output_dict['inc_auc'] = auc_include
output_dict['ex_auc'] = 100-auc_exclude
if calc_infidelity: output_dict['infidelity'] = infid
output_dict['explainer_time'] = explainer_time
output_dict['wasg_time'] = wasg_time
output_dict['nUnif'] = args.nUnif
output_dict['abs'] = args.abs
output_dict['l2_bound'] = args.l2_bound
output_dict['exp_norm'] = args.exp_norm
output_dict['solver'] = args.solver
output_dict['date'] = str(datetime.now())
output_dict['seed'] = args.seed

output = pd.DataFrame(output_dict, index = [0])

#################################################
filepath = os.path.join(args.metric_save_path, 'solver%s_scale%s_centering%s_dataset%s_explainer%s_expnorm%s_nUnif%s_abs%s_seed%s_bound%s_std%s_wasg.pkl' % (args.solver, args.scale_diameter, args.centering, args.dataset, args.explainer, args.exp_norm, str(args.nUnif), str(args.abs), str(args.seed), str(args.l2_bound), str(args.std)))
make_dir_fromfile(filepath)
output.to_pickle(filepath)
print(filepath)
#################################################
