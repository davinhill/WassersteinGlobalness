# from msilib.schema import Error
from atexit import register
import torch
from torch import Tensor
import numpy as np
import torch.nn.functional as F
from utils.utils import tensor2cuda

class BaseExplainer(object):
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.hooks = [] 
        self.gradient = None
        
    def explain(self, input, target=None):
        print(f"WARNING: parent class called - this function should be implemented by child")
        pass
    
    def unhook(self):
        for hook in self.hooks: hook.remove()
        

class VanillaGrad(BaseExplainer):
    
    def __init__(self, model):
        # super(VanillaGrad, self).__init__(model)
        super().__init__(model)
        
    def explain(self, input, target=None, image = False):
        input = Tensor(input).clone()
        input.requires_grad = True
        input.retain_grad()
        
        outputs = self.model(input)  # Logits from the prediction model
        if target is not None:
            if len(target.shape) == 0:
                target = target.repeat(outputs.shape[0])
            
            try:
                output_dim = outputs.shape[1]
            except:
                output_dim = 1

            if output_dim > 1:
                onehot = F.one_hot(target.long(), num_classes = outputs.shape[1])
                self.model.zero_grad()
                outputs.backward(tensor2cuda(onehot))
            else:
                self.model.zero_grad()
                outputs.backward()
        else:
            if len(outputs.shape) > 1:
                onehot = F.one_hot(outputs.argmax(dim = 1), num_classes = outputs.shape[1])
                self.model.zero_grad()
                outputs.backward(tensor2cuda(onehot))
            else:
                self.model.zero_grad()
                outputs.backward()

        # onehot = torch.zeros_like(outputs)
        # onehot[target if target is not None else outputs.argmax()] = 1

        
        if image:
            return np.moveaxis(input.grad.detach().cpu().numpy()[0], 0, -1)
        else:
            return input.grad.detach().cpu().numpy()
        
    def get_smoothed_explanation(self, input, target=None, samples=50, std=0.15, squared = False, image=True):
        
        input = tensor2cuda(input)  # should already be there, but just in case
        
        # scale noise amount by the image size
        # noise_amt = std * (input.max() - input.min()).cpu().detach().numpy()
        noise_amt = std
        if squared:
            process = lambda x: x**2
        else:
            process = lambda x: x
        
        if image:
            _, num_channels, width, height = input.size()
            cum_grad = torch.zeros((width, height, num_channels))  # accumulate the gradients into this variable
            for i in range(samples):
                noise = tensor2cuda(torch.normal(0, noise_amt, size=input.size()))
                # noisey_img = input + noise
                cum_grad += process(self.explain(input + noise, target, image=image)) / samples
        else:
            # print(f"INPUT SIZE: {input.size()}")
            cum_grad = torch.zeros(self.explain(input, target, image=True).shape)
            for i in range(samples):
                noise = tensor2cuda(torch.normal(0, noise_amt, size=input.size()))
                cum_grad += process(self.explain(input + noise, target, image=True)) / samples
        return cum_grad

class IntegratedGradients(VanillaGrad):
    
    def explain(self, input, target=None, image=True, baseline=0, steps=100, postproc=lambda x: x):
        
        if baseline == 0:
            baseline_img = torch.zeros_like(input)
        else:
            baseline_img = torch.ones_like(input)
        diff = input - baseline_img
        
        if image:
            _, num_channels, width, height = input.size()
            cum_grad = np.zeros((width, height, num_channels))
        else:
            # print(f"Size of input: {input.size()}") 
            d = len(input)
            cum_grad = np.zeros((d,))
            
        for alpha in np.linspace(0,1,steps):
            cur_im = alpha * diff + baseline_img
            grad = postproc(super(IntegratedGradients, self).explain(cur_im, target, image=image))  # Get gradient
            cum_grad += grad
            
        if image:
            # swapaxis to move channel axis to end
            return cum_grad * np.moveaxis(diff.detach().cpu().numpy()[0], 0, -1) / steps  
        else:
            return cum_grad * diff.detach().cpu().numpy()
        

class GuidedBackprop(VanillaGrad):
    
    def __init__(self, model):
        super().__init__(model)
        self.activation_inputs = []
        self.deconv()
        
    def deconv(self):
        
        # Hook functions
        def clip_grad(block, grad_in, grad_out):
            input = self.activation_inputs.pop()
            out = grad_out[0]
            out *= (grad_out[0] > 0).float()
            out *= (input > 0).float()
            return(out,)
        
        def register_input(block, inputs, outputs):
            self.activation_inputs.append(inputs[0])
        
        # Update the ReLUs
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.ReLU):
                self.hooks.append(layer.register_forward_hook(register_input))
                self.hooks.append(layer.register_backward_hook(clip_grad))