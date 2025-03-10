## Locality utilities

import numpy as np
import torch
import ot
import ot.plot
import time
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import warnings
from matplotlib import cm
from sklearn import metrics
import os
import sys
import torch.nn.functional as F
from tqdm import tqdm

from matplotlib.colors import ListedColormap


from ot import wasserstein_1d
from ot.backend import get_backend
from ot.utils import list_to_array
from ot.sliced import get_random_projections

# from integrated_gradients import IntegratedGradients
sys.path.append('./')
from utils.explainers import IntegratedGradients, VanillaGrad, GuidedBackprop

def grayscale(explanation):
    if type(explanation) != np.ndarray:
        explanation = np.array(explanation)
    if len(explanation.shape) > 2:
        if explanation.shape[2] == 3:
            return np.abs(np.sum(explanation, axis=2))
    
    return explanation

def exp_intGrad_image(images, model):
    e = IntegratedGradients(model)
    exps = [e.explain(input=x.unsqueeze(0), postproc=np.abs, baseline=0) for x in images]
    exps = [exp.transpose(2, 0, 1) for exp in exps]
    return(exps)

def exp_intGrad(images, model):
    e = IntegratedGradients(model)
    exps = [e.explain(input=torch.Tensor(x), postproc=np.abs, baseline=0, image=False) for x in images]
    return(exps)

def exp_vanillaGrad_image(images, model, target=None, smooth=False, return_tensor = False):
    e = VanillaGrad(model)
    exps = [e.explain(input=torch.Tensor(x).unsqueeze(0), image=True, target=t) for x,t in zip(images,target)]

    if return_tensor:
            exps = torch.cat([numpy2cuda(exp).permute(2,0,1).unsqueeze(0) for exp in exps], dim = 0).cpu()

    else:
        exps = [exp.transpose(2, 0, 1) for exp in exps]
    return exps

def exp_vanillaGrad(images, model, target=None, return_tensor = False):
    e = VanillaGrad(model)
    if target is not None:
        exps = [e.explain(input=torch.Tensor(x), image=False, target=t) for x,t in zip(images,target)]
    else:
        exps = [e.explain(input=torch.Tensor(x), image=False, target=None) for x in images]

    if return_tensor:
        exps = torch.cat([numpy2cuda(exp).unsqueeze(0) for exp in exps], dim = 0).cpu()

    return(exps)

def exp_guidedsmoothGrad(images, model, _nsamp=250, std=0.15, image = True, return_tensor = True):
    e = GuidedBackprop(model)
    exps = [e.get_smoothed_explanation(input=torch.Tensor(x).unsqueeze(0), samples=_nsamp, std=std, image = image) for x in images]
    
    e.unhook()
    if return_tensor:
        if image:
            exps = torch.cat([exp.permute(2,0,1).unsqueeze(0) for exp in exps], dim = 0).cpu()
        else:
            exps = torch.cat([exp.unsqueeze(0) for exp in exps], dim = 0).cpu()
    else:
        if image:
            exps = [tensor2numpy(exp).transpose(2, 0, 1) for exp in exps]
        else:
            exps = [tensor2numpy(exp) for exp in exps]
    return(exps)      

def exp_smoothGrad(images, model = None, _nsamp=250, target=None, std=0.15, image=True, squared = True, return_tensor = False):
    e = VanillaGrad(model)
    exps = [e.get_smoothed_explanation(torch.Tensor(x).unsqueeze(0), target=t, samples=_nsamp, std=std,image=image, squared = squared) for x,t in zip(images,target)]
    e.unhook()
    if return_tensor:
        if image:
            exps = torch.cat([exp.permute(2,0,1).unsqueeze(0) for exp in exps], dim = 0).cpu()
        else:
            exps = torch.cat([exp.unsqueeze(0) for exp in exps], dim = 0).cpu()
    else:
        if image:
            exps = [tensor2numpy(exp).transpose(2, 0, 1) for exp in exps]
        else:
            exps = [tensor2numpy(exp) for exp in exps]
    return(exps)     

def exp_guidedbackprop(images, model, image = True):
    e = GuidedBackprop(model)
    exps = []
    for i in range(images.shape[0]):
        exps.append(e.explain(input = images[i:i+1,...]))
    e.unhook()
    exps = [np.squeeze(exp, 0) for exp in exps]
    return(exps)

def get_explainer(explainer, model, background, true_label, image=True):
    # Instantiate explainer 
    if explainer.lower() in ['expectedgradients', 'expgrad']:  # EXPECTED GRADIENTS
        model.np = False
        _nsamp = 100 
        e = shap.GradientExplainer(model, background)
        e_fn = lambda x, model : e.shap_values(x, nsamples = _nsamp, rseed = 0)[true_label]
    elif explainer.lower() in ['vanillagrad', 'vanilla']:
        if image: 
            e_fn = exp_vanillaGrad_image
        else: 
            e_fn = exp_vanillaGrad
    elif explainer.lower() in ['guidedbackprop']:
        e_fn = exp_guidedbackprop
    elif explainer.lower() in ['smoothgrad']:
        e_fn = exp_smoothGrad
    elif explainer.lower() in ['guidedsmoothgrad']:
        e_fn = exp_guidedsmoothGrad
    elif explainer.lower() in ['integrated gradients']:
        if image: 
            e_fn = exp_intGrad_image
        else:
            e_fn = exp_intGrad
        
    elif explainer.lower() in ['gradcam']:
        e = GradCam(model)
        e_fn = lambda x: e.get_mask(input=x, target_class=true_label)
    elif explainer.lower() in ['deepshap']:
        model.np = False
        if type(background) == np.ndarray:
            background = torch.Tensor(background).to(model.device)
        e = shap.DeepExplainer(model, background)
        e_fn = lambda x, model: e.shap_values(x)#[true_label]
        exp_args = {}
    elif explainer.lower() in ['kernel', 'kernelshap']:
        # If this runs out of CUDA memory very quick, decrease size of background 
        np_bg = np.array([im.reshape(-1) for im in background.cpu().detach().numpy()])
        
        if not type(all_test_images) == np.ndarray:
            all_test_images = all_test_images.cpu().detach().numpy()
            
        e = shap.KernelExplainer(model.predict_flattened, np_bg[:20]) 
        
        e_fn = lambda x: np.array([ exp.reshape(-1, 32, 32) for exp in e.shap_values(np.array([im.reshape(-1) for im in x]), nsamples=1000)[true_label] ]) # "The “auto” setting uses nsamples = 2 * X.shape[1] + 2048. This is too big!
    elif explainer.lower() in ['lime']:
        model.np = True
        e = lime_image.LimeImageExplainer(kernel_width=1.0)
        if not type(all_test_images) == np.ndarray:
            all_test_images = all_test_images.cpu().detach().numpy()
        e_fn = lambda x: lime_exp_fn(net.predict_channel_end, e, x, true_label, nsamples=15000)
    else:
        print(f"WARNING: explainer not recognized!")
        return(-1)  
    return e_fn

def lime_exp_fn(pred_fn, e, x, label, nsamples=5000):

    if torch.is_tensor(label):
        label = label.item()
    batch_size, channel_size, imheight, imwidth = x.shape    
    if channel_size == 1:
        # MNIST
        explanations = [e.explain_instance(gray2rgb(img).squeeze(), 
            pred_fn, top_labels=10, num_samples=nsamples, hide_color=0) for img in x]  # progress_bar=False gives unexpected keyword argument for some reason...
    elif channel_size == 3:
        # e.g. CIFAR
        # Have to move channel to last dim.
        explanations = [e.explain_instance(np.moveaxis(img, source=0, destination=-1), 
            pred_fn, top_labels=10, num_samples=nsamples) for img in x]  # progress_bar=False gives unexpected keyword argument for some reason...
    dics = [exp.local_exp for exp in explanations]
    segs = [exp.segments for exp in explanations]

    exps = []
    for dic, seg in zip(dics, segs):
        assert label in dic.keys(), 'GT label missing from explanation'
        exp_im = np.zeros(np.shape(seg))

        for superpix_ind, shap_val in dic[label]:
            exp_im[np.where(seg == superpix_ind)] = shap_val
        exps.append(exp_im)
    
    return(exps)


def make_meshgrid(x, y, step=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    return xx, yy

def plot_contours(ax, clf, xx, yy, fill = True,**params):
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if torch.is_tensor(Z):
        Z = Z.cpu().detach().numpy()
    if len(Z.shape) > 1:  # unfortunately this is necessary - using many different models
        # if Z.shape[1] == 2:  # logits - take argmax
        #     Z = np.argmax(Z, axis=1)
        Z = scipy.special.softmax(Z, axis = 1)
        Z = Z[:,1]
    Z = Z.reshape(xx.shape)
    if fill:
        out = ax.contourf(xx, yy, Z, **params)
    else:
        out = ax.contour(xx, yy, Z, **params)
    return out

def hyperplane_data(n,d,_scale=0.1, n_clusters=None, logits=False):
    '''
    Create Synthetic Data for Jagged Boundary Experiment

    args:
        n: number of samples
        d: number of dimensions (equal to number of clusters if n_clusters is not provided)
        _scale: scale (std dev) for gaussian distribution
        n_clusters: number of clusters

    '''
    
    if n_clusters is None:
        n_clusters = d  # number of clusters

    k = n//n_clusters  # samples per cluster
    n = k * n_clusters  # in case n is not divisible by n_clusters

    ctrs = np.zeros((n_clusters,d))

    # i'th cluster will use the i'th dimension to generate labels,
    # and will be centered at e_j^i = 0 \forall j != i, e_i^i = 1 
    for ii in range(n_clusters):
        ctrs[ii, ii] = 1
    
    # generate clusters
    X = np.vstack(([np.random.normal(loc=ctr, scale=_scale, size=(k,d)) for ctr in ctrs]))
    
    src_labels = np.vstack(([np.ones((k,)).reshape(-1, 1)*j for j in range(n_clusters)]))

    if logits:
        # generate labels
        y = np.zeros((n,2), dtype=int)
        for j in range(n_clusters):
            bot, top = j * k, (j+1) * k
            y[bot:top, 0] = [int(s) for s in (X[bot:top, j] > 1)]
            y[bot:top, 1] = [int(s) for s in (X[bot:top, j] <= 1)]
    else:
        # generate labels
        y = np.empty((n,), dtype=int)
        for j in range(n_clusters):
            bot, top = j * k, (j+1) * k
            y[bot:top] = X[bot:top, j] > 1

    return(X,y,src_labels)

def fuzz_boundary(X, y, thresh = 0.075):
    # thresh is size of groups
    if thresh>0:
        X = X.copy()
        D = metrics.pairwise_distances(X, X, metric='euclidean')
        for ind,row in enumerate(D):
            nearby = np.abs(row) < thresh
            if np.any(y[nearby] != y[ind]):
                # flip points close to boundary
                y[nearby] = y[ind]

    return(X,y)


def sample_spherical(npoints, _d=3, quadrant=None):
    randvec = np.random.randn(npoints//2, _d)
    randvec = np.concatenate((randvec, -randvec), axis = 0)
    randvec /= np.linalg.norm(randvec, axis=1, keepdims = True)
    
    if quadrant==1:
        randvec = np.abs(randvec)
    else:
        assert quadrant is None, f'unrecognized quadrant argument: {quadrant}'
    
    return randvec

def sample_cube(npoints, _d=3, u_min = -1, u_max = 1):
    '''
    sample uniform distribution over hypercube.

    args:
        n_points: number of samples
        _d: number of dimensions
        u_min (int or np vector): minimum of uniform dist'n.
        u_max (int or np vector): maximum of uniform dist'n.

    return:
        matrix of samples
    
    '''
    # Generates random points in the interval [u_min, u_max]^d
    if type(u_min) == np.ndarray or type(u_max) == np.ndarray:
        # separate min / max for each dimension
        assert type(u_min) == np.ndarray and type(u_max) == np.ndarray # both must be vectors
        assert (u_max - u_min < 0).sum() == 0 # check max > min
        randvec = np.random.uniform(0, 1, size=(npoints, _d))
        randvec = randvec * (u_max - u_min) + u_min

    else:
        assert u_max > u_min # check max > min
        randvec = np.random.uniform(u_min, u_max, size=(npoints, _d))

    return randvec

def sample_permutation(n, d, seed = 0):
    np.random.seed(seed)
    samples = np.zeros((n,d))
    for i in range(n):
        samples[i,:] = np.random.permutation(d)
    return samples

def sample_sphere_interior(d, n1, r=1):
    # scale radius by 1/d
    assert r >= 0, f"radius must be nonnegative"
    if r > 0:
        MVN_Unit_Samples          = torch.normal(mean = torch.zeros(d * n1), std = torch.ones(d * n1 ))
        MVN_Unit_Samples          = MVN_Unit_Samples.view(n1, d)
        Sphere_Uniform_Samples    = MVN_Unit_Samples / MVN_Unit_Samples.norm(dim = 1).unsqueeze(dim = 1) 
        Radii_Uniform_Samples     = (torch.rand((n1,1)) ** (1/d)) * r
        Ball_Uniform_Samples      = Sphere_Uniform_Samples * Radii_Uniform_Samples
    else:
        Ball_Uniform_Samples = torch.zeros((n1, d))
    return Ball_Uniform_Samples

def get_codes(d):
    codes = []
    for i in range(1<<d):
        code = [0 for j in range(d)]
        binary = bin(i).split('b')[-1]
        code[-len(binary):] = [int(char) for char in binary]
        codes.append(code)
    return(codes)

def sample_hypercube(ndim):
    # Generates an array of hypercube points
    vec = get_codes(ndim)
    return vec

def calc_range(explanations):
    distances = metrics.pairwise_distances(explanations,metric = 'euclidean')
    return distances.max()


def wasserstein_globalness(shap_values, metric = 'euclidean', reg=.15, n_unif=5000, abs=False, numItermax=2000, exp_norm = 'sphere_l2_bound', seed = 0, scale = 1, scale_diameter = None, solver = 'pot_slice_symmetric', centering = 'mean', l2_bound = 5, log = False, normalize_global = True, n_projections = 500):
    '''
    returns wasserstein globalness, approximated using sinkhorn

    args:
        shap_values: attributions
        metric (str): cosine, euclidean, or hamming
        reg (float): regularization for sinkhorn approximation
        n_unif (int): number of samples for approximating uniform distn
        abs(bool): set this to True if explanations are all non-negative.
        exp_norm (str): type of normalization for explanations (for debugging).
            'sphere': normalize each explanation to be length 1 (l2 norm).
            'cube': scale all explanations to be within 0 and 1
            'cube_ind': independently scale each dimension of the explanations to be within 0 and 1
            'cube_scale': no normalization. Sets the min and max of the uniform samples to match the explanations for each dimension.

    return:
        wasserstein globalness
    '''
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    metric = metric.lower()
    explanations = np.array(shap_values).squeeze()
    unnorm_wasg = 0 # for logging, if normalize_global == True

    if len(np.shape(explanations)) > 2:
        explanations = explanations.reshape(explanations.shape[0], -1) # flatten
    elif len(explanations.shape) == 1:
        explanations = explanations.reshape(-1,1)

    if abs:
        quadrant = 1
        explanations = np.abs(explanations)
    else:
        quadrant = None

    n_p, d = np.shape(explanations)
    
    if metric == 'cosine' or metric == 'euclidean' or metric == 'angular':
        if centering in ['mean', 'median', 'midrange']:
            explanations = center(explanations, method = centering)
        if exp_norm == 'nonorm':
            x_u = sample_spherical(n_unif, _d=d, quadrant=quadrant)
        elif exp_norm == 'sphere':
            explanations += (explanations == 0) * 1e-10 # numerical stability
            explanations = explanations / np.linalg.norm(explanations, axis = 1, keepdims = True)
            explanations = explanations * (1/scale)
            x_u = sample_spherical(n_unif, _d=d, quadrant=quadrant)
        elif exp_norm == 'sphereinterior':
            max_norm = np.linalg.norm(explanations, axis = 1).max()
            if max_norm != 0: explanations /= max_norm
            explanations = explanations * (1/scale)
            x_u = sample_sphere_interior(d = d, n1 = n_unif, r = 1)
        elif exp_norm == 'sphere_scale':
            max_norm = np.linalg.norm(explanations, axis = 1).max()
            x_u = sample_sphere_interior(d = d, n1 = n_unif, r = max_norm)
        elif exp_norm == 'sphere_l2_bound':
            x_u = sample_sphere_interior(d = d, n1 = n_unif, r = l2_bound)

    elif metric == 'hamming':
        x_u = sample_hypercube(d)
    elif metric == 'kendalltau':
        explanations = np.argsort(-explanations, axis = 1) # convert explanations to ranking (largest to smallest)
        x_u = sample_permutation(n = n_unif, d = d)
        metric = kendalltau_distance # predefined distance function
    n_u, _ = np.shape(x_u)

    if solver == '2d':
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment

        d = cdist(explanations, np.array(x_u))
        assignment = linear_sum_assignment(d)
        d /=d.max()
        output = d[assignment].sum() / explanations.shape[0]
    elif solver == 'pot':
        # sklearn parallelization
        if metric == 'cosine' or metric == 'euclidean' or metric == 'angular':
            n_jobs = None
        else:
            n_jobs = -1
        D = metrics.pairwise_distances(explanations, np.array(x_u), metric=metric, n_jobs = n_jobs)
        D /= D.max()
        D = np.nan_to_num(D, nan = 0.0) # replace nan with zero

        plan = ot.bregman.sinkhorn(np.ones((n_p,))/n_p, np.ones((n_u,))/n_u, D, reg, numItermax=numItermax)
        output = np.sum(plan * D)
        
        # normalize global explanation = 1
        if normalize_global:
            unnorm_wasg = output.copy()

            # calculate normalization factor
            D = metrics.pairwise_distances(explanations[0:1,:], np.array(x_u), metric=metric, n_jobs = n_jobs)
            D /= D.max()
            D = np.nan_to_num(D, nan = 0.0) # replace nan with zero
            plan = ot.bregman.sinkhorn(np.ones((1,)), np.ones((n_u,))/n_u, D, reg, numItermax=numItermax)
            normalization_factor = np.sum(plan * D)
            output /= normalization_factor

    elif solver == 'pot_slice':
        output = ot.sliced_wasserstein_distance(X_s = np.array(x_u), X_t = explanations, n_projections = n_projections, seed = seed, p=2)

    elif solver == 'pot_slice_symmetric':
        output, log_dict = sliced_wasserstein_uniformsphere(X_unif = np.array(x_u), X_t = explanations, n_projections = n_projections, seed = seed, p=2, normalize_global = normalize_global, log = True)
        unnorm_wasg = log_dict['unnorm_wasg']

    if log:
        return output, {'exp': explanations, 'x_u': np.array(x_u), 'unnorm_wasg': unnorm_wasg}
    else:
        return output

def sliced_wasserstein_uniformsphere(X_unif, X_t, a=None, b=None, n_projections=50, p=2,
                                normalize_global = True, projections=None, seed=None, log=False):
    r"""

    Modification of ot.sliced_wasserstein_distance that assumes X_unif is a uniform distribution with support defined over the interior of a d-dimensional ball.


    Parameters
    ----------
    X_unif : ndarray, shape (n_samples_a, dim)
        samples in the source domain, must be uniformly distributed over the interior of a d-dimensional ball.
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional =
        Power p used for computing the sliced Wasserstein
    normalize_global: normalize by global measure
    projections: shape (dim, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    log : dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45
    """
    # from .lp import wasserstein_1d

    X_unif, X_t = list_to_array(X_unif, X_t)

    if a is not None and b is not None and projections is None:
        nx = get_backend(X_unif, X_t, a, b)
    elif a is not None and b is not None and projections is not None:
        nx = get_backend(X_unif, X_t, a, b, projections)
    elif a is None and b is None and projections is not None:
        nx = get_backend(X_unif, X_t, projections)
    else:
        nx = get_backend(X_unif, X_t)

    n = X_unif.shape[0]
    m = X_t.shape[0]

    if X_unif.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_unif and X_t must have the same number of dimensions {} and {} respectively given".format(X_unif.shape[1], X_t.shape[1]))

    if a is None:
        a = nx.full(n, 1 / n, type_as=X_unif)
    if b is None:
        b = nx.full(m, 1 / m, type_as=X_unif)

    d = X_unif.shape[1]

    if projections is None:
        projections = get_random_projections(d, n_projections, seed, backend=nx, type_as=X_unif)
    else:
        n_projections = projections.shape[1]

    # project over one direction and copy over all directions (symmetry)
    if type(X_unif) == np.ndarray:
        X_unif_projections = nx.dot(X_unif, projections[:,:1])
        X_unif_projections = X_unif_projections.repeat(n_projections, axis = 1)
    elif type(X_unif) == torch.Tensor:
        X_unif_projections = nx.dot(X_unif, projections[:,:1])
        X_unif_projections = X_unif_projections.expand(-1,n_projections)
    else:
        X_unif_projections = nx.dot(X_unif, projections)
    
    # Check if explanation is global.
    # Global explanations can be calculated efficiently due to symmetry
    if nx.abs(X_t).sum() == 0:
        glob_exp = True
    else:
        glob_exp = False

    # Calculate global WasG for normalization
    # The global explanation has equal 1d wasserstein for each projection. Therefore this can be calculated with a single projection
    if normalize_global or glob_exp:
        X_unif_projections_glob = X_unif_projections[:,:1]
        X_t_projections_glob = nx.zeros((1, 1))
        projected_emd_glob = wasserstein_1d(X_unif_projections_glob, X_t_projections_glob, a, nx.full(1, 1, type_as=X_unif), p=p)
        res_glob = nx.sum(projected_emd_glob) ** (1.0 / p)
        res = res_glob # set explanation WasG to be global WasG (overwritten below if explanations are not global)
    else:
        res_glob = 0
    
    # calculate WasG for non-global explanations
    if not glob_exp:
        X_t_projections = nx.dot(X_t, projections)
        projected_emd = wasserstein_1d(X_unif_projections, X_t_projections, a, b, p=p)
        res = (nx.sum(projected_emd) / n_projections) ** (1.0 / p)

    if normalize_global: res /= res_glob # normalize by global WasG

    if log:
        return res, {"unnorm_wasg": res * res_glob}
    return res


def plot_point(X,y,clf=None, pts=None,feature_names=None,i=None,j=None,title=None):
    h = .02  # mesh coarseness

    n,d = X.shape

    if i is None and j is None:
        i,j = 2,3  # axes to plot  (need to slice 4D data to 2D)

    x_min, x_max = X[:, i].min() - 1, X[:, i].max() + 1
    y_min, y_max = X[:, j].min() - 1, X[:, j].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    
    # new
    if clf is not None:
        z_sep = []
        for k in range(d):
            if k == i:
                z_sep.append(xx.ravel().reshape(-1, 1))
            elif k == j:
                z_sep.append(yy.ravel().reshape(-1, 1))
            else:
                z_sep.append(np.ones(xx.ravel().reshape(-1,1).shape)*X[:,k].mean())  # replace missing axis with mean value 
        Z = clf.predict(np.hstack(z_sep))
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z)


    # Plot the training points
    plt.scatter(X[:, i], X[:, j], c=y)
    if pts is not None:
        plt.scatter(pts[:, i], pts[:, j], c='k', s=30)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    if title is not None:
        plt.title(title)
    if feature_names is not None:
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])

    plt.show()

def get_neighborhood(ctr, clf, var, _n=1000):
    d = ctr.shape[1]
    
    # sanity check
    ctr = ctr.squeeze()
    var = var.squeeze()
    assert ctr.shape == var.shape

    neighborhood_data = np.vstack(( ctr, np.array([np.random.normal(loc=ctr, scale=var).squeeze() for i in range(_n)]) ))  # check this after pulling changes on lambda machine 

    # print(f"neighborhood_data shape: {neighborhood_data.shape}")
    preds = clf.predict_proba(neighborhood_data)
    if torch.is_tensor(preds): 
        neighborhood_labels = preds.cpu().detach().numpy()
    else:
        neighborhood_labels = preds
    # print(f"neighborhood_labels shape: {neighborhood_labels.shape}")

    return(neighborhood_data, neighborhood_labels)

def kendalltau_distance(x, y):
    '''
    returns kendall-distance between ordinal vectors x, y
    '''
    n = x.size
    kt_stat, _ = stats.kendalltau(x, y) # kendall-tau rank correlation
    kt_dist = (1-kt_stat) * (n * (n-1)) * 0.25 # convert to kendall-tau distance
    return kt_dist

def disp_exp(exp, feature_names):
    for tuple in exp:
        print(f"{feature_names[tuple[0]]} --- {tuple[1]:0.3f}")
    
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


def auto2cuda(obj):
    # Checks object type, then calls corresponding function
    if type(obj) == list:
        return list2cuda(obj)
    elif type(obj) == np.ndarray:
        return numpy2cuda(obj)
    elif type(obj) == torch.Tensor:
        return tensor2cuda(obj)
    else:
        raise ValueError('input must be list, np array, or pytorch tensor')

def create_grid(grid_min, grid_max, dim, gridsize = 10):
    '''
    create evenly spaced grid.
    
    args:
        grid_min: minimum value
        grid_max: maximum value
        dim: number of dimensions
        gridsize: (int) number of grid elements in each dimension
    return:
        grid: (np matrix) size gridsize^d x d
    '''
    
    linspace_list = []
    for feat_idx in range(dim):
        linspace_list.append(np.linspace(grid_min, grid_max, gridsize))
        
    xx_list = np.meshgrid(*linspace_list)
    grid = np.vstack(tuple(map(np.ravel, xx_list)))

    return grid.transpose()

########################################################
########################################################
########################################################


def cosine_similarity(x,y):
    return np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))
def angular_distance(x,y):
    return np.arccos(1-cosine_similarity(x,y)) / np.pi

def center(explanations, method = 'mean'):
    rounding = 10 # numerical precision for global explanations
    if method == 'mean':
        return np.round(explanations - explanations.mean(axis = 0, keepdims = True), rounding)
    elif method == 'midrange':
        distances = metrics.pairwise_distances(explanations,metric = 'euclidean')
        width = distances.max()
        row, col = np.unravel_index(np.argmax(distances), distances.shape)
        a = explanations[row,:]
        b = explanations[col,:]
        midpoint = 0.5*(b-a) + a
        return np.round(explanations - midpoint.reshape(1,-1), rounding)
    elif method == 'median':
        from geom_median.torch import compute_geometric_median
        points = [numpy2cuda(explanation, cuda = False) for explanation in explanations]
        output = compute_geometric_median(points, weights=None)
        return np.round(explanations - tensor2numpy(output.median).reshape(1,-1), rounding)


def normalized_kendall_tau_distance(a,b):
    '''
    Kendall Tau Distance, normalized to [0,1] (divided by #permutations)

    Let A = Number of Pairwise Rank Agreement
    Let B = Number of Pairwise Rank Disagreement

    Corr = (A-B)/(A+B)
    Normalized Distance = B/(A+B) = (1-Corr)/2

    '''
    stat, _ = stats.kendalltau(a,b)
    return (1-stat)/2

def batch_explanations(explainer_func, inputs, batch_size = 5, target = None, verbose = True, **kwargs):
    batches = torch.split(torch.arange(inputs.shape[0]), batch_size)
    exp_list = []

    iterator = batches
    if verbose: iterator = tqdm(iterator, total = len(batches))
    for input_batch in iterator:
        x = tensor2cuda(inputs[input_batch,...])
        if target != None:
            y = tensor2cuda(target[input_batch])
            exp_list.append(explainer_func(x, target = y, **kwargs).detach().cpu())
        else:
            exp_list.append(explainer_func(x, **kwargs).detach().cpu())
    output =torch.cat(exp_list, dim = 0) 
    return output


class softmax_wrapper():
    '''
    wrapper for pytorch model
    '''
    def __init__(self, model, orig_x, tabnet = False):
        '''
        args:
            model: pytorch model
            orig_x: x samples, to get input shape for the model. The first dimension of orig_x should be the number of samples.
        '''
        self.input_shape = orig_x.shape
        # if using tabnet, only use model output and not loss
        if tabnet:
            self.model = tabnet_wrapper(model)
        else:
            self.model = model

        
        # get dimension of model output
        output_shape = self.model(auto2cuda(orig_x[:1,...])).shape # get shape of model output
        if len(output_shape) == 1:
            self.output_dim = 1
        else:
            self.output_dim = output_shape[1]

    def flattened_softmax(self,x):
        '''
        assumes x is flattened
        '''

        with torch.no_grad(): output = self.model(auto2cuda(x).reshape((-1,) + self.input_shape[1:]))

        if self.output_dim == 1:
            output = F.sigmoid(output)
        else:
            output = F.softmax(output, dim = 1)

        return tensor2numpy(output)

    def softmax(self,x):
        '''
        assumes x is same shap as orig_x
        '''

        input_type = type(x)
        with torch.no_grad(): output = self.model(auto2cuda(x))

        if self.output_dim == 1:
            output = F.sigmoid(output)
        else:
            output = F.softmax(output, dim = 1)

        if input_type == np.ndarray:
            return tensor2numpy(output)
        else:
            return output.detach().cpu()

    def softmax_sc2mc(self, x):
        '''
        assumes x is the same as orig_x

        for binary classifiers with scalar output, convert to 2-dimension output (1-p, p)
        '''

        assert self.output_dim == 1, 'this function can only be applied for binary classifiers'
        input_type = type(x)
        with torch.no_grad(): output = self.model(auto2cuda(x))

        output = F.sigmoid(output).reshape(-1,1)
        output = torch.hstack((1-output, output))
        return output
