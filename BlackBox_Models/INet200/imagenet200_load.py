from openood.networks import ResNet18_224x224
import torch
import os
from openood.evaluation_api.datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor

net = ResNet18_224x224(num_classes=200)
net.load_state_dict(
            torch.load('best.ckpt', map_location='cpu'))
net.eval() 


loader_kwargs = {
    'batch_size': 200,
    'shuffle': True,
    'num_workers': 1
}

preprocessor = get_default_preprocessor('imagenet200')
dataloader_dict = get_id_ood_dataloader(id_name = 'imagenet200', data_root  = "./data/",
                                        preprocessor = preprocessor, **loader_kwargs)
