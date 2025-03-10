# Wasserstein Globalness (WG)

Implementation of Wasserstein Globalness, a method to quantify the *globalness* of an explainer based on the distribution of its explanations over a dataset.

![Image](https://github.com/davinhill/WassersteinGlobalness/blob/main/Figures/fig1.png?raw=true)

For more details, please see our full paper:

**Axiomatic Explainer Globalness via Optimal Transport**  
Davin Hill*, Josh Bone*, Aria Masoomi, Max Torop, Jennifer Dy  
*Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS) 2025*  
[[Paper]](https://arxiv.org/pdf/2411.01126)

## Tutorial and Examples
**Under Construction**: Will be updated by the AISTATS conference in May 2025.


## Implementation

* The function wasserstein_globalness() in ./utils/locality_utils.py calculates wasserstein globalness.
* An example implementation of Wasserstein Globalness on CIFAR10 samples is provided in ./Example



## Experiments

Below we detail source code from the manuscript experiments.

**Datasets and Black-Box Models:**
The black-box models evaluated in the experiments section are trained using the code in the [Models/blackbox_model_training](https://github.com/davinhill/GPEC/tree/main/Tests/Models/blackbox_model_training) directory. Datasets are not included in the repository due to file size, however all datasets are publicly available with sources listed in the paper supplement.


* **AUC_Experiment** contains the code for the experiment in Section 5.1 (incAUC/excAUC/Infidelity comparison)
* **ClusterExperiment** contains the code for the experiment in Section 5.3 (group experiment)
* **JaggedBoundary** contains the code for the experiment in Section 5.2 (synthetic dataset)
* **time** contains the code for estimating computation time for varying number of features (App. D.1)
* **Ablation** contains the code for the ablation experiment (App. D.1).


