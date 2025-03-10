
#%% Import Libraries ===============================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torchvision.models.resnet import conv3x3
from torch.autograd import Variable
from datetime import datetime
from tqdm import tqdm
import os




def calc_test_accy(model, test_loader):
    model.eval()   # Set model into evaluation mode
    correct = 0
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(
                device)  # Move data to GPU
            output = model(data)   # Calculate Output
            pred = output.max(1, keepdim=True)[1]  # Calculate Predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

        return (100.*correct/len(test_loader.dataset))


# Define function for calculating predictions on training data
# num_batches indicates how many training observations to use for training accuracy calculation
# This reduces computation time, instead of using the entire training set.
def calc_train_accy(model, dataloader, num_batches, batch_size):
    model.eval()
    correct = 0
    with torch.no_grad():
        data_iterator = iter(dataloader)
        for i in range(num_batches):  # iterate for the specified number of batches
            try:
                data, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                data, target = next(data_iterator)

            data, target = data.to(device), target.to(
                device)  # Move data to GPU
            output = model(data)   # Calculate Output
            pred = output.max(1, keepdim=True)[1]  # Calculate Predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    return (100.*correct/(num_batches * batch_size))

def load_data(path_data = './', batch_size = 32):

    # Define Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue = 0.05),   # Change brightness, contrast, and saturation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),

        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
    }


    # Training Data
    train_set = datasets.CIFAR10(
        root=path_data, train=True, download=True, transform=data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # Test Data
    test_set = datasets.CIFAR10(
        root=path_data, train=False, download=True, transform=data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_set, train_loader, test_set, test_loader


class BasicBlock(nn.Module):
    # custom basicblock adapted from https://github.com/pytorch/captum/issues/378#
    # changes relu name to fix issues with captum package

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # Added another relu here
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        # Modified to use relu2
        out = self.relu2(out)

        return out



class custom_resnet(nn.Module):
    def __init__(self, input_shape = (3,32,32), num_classes = 10, model = 'resnet18'):
        super(custom_resnet, self).__init__()
        if model == 'resnet18':
            # self.model = models.resnet18(pretrained=False)
            self.model = models.resnet._resnet(BasicBlock, [2, 2, 2, 2], weights = None, progress = False) # use custom basicblock to fix captum package issue
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.input_shape = input_shape
            self.np = False # output as numpy
            self.softmax = False

    def forward(self, x):
        if self.softmax:
        
            return F.softmax(self.model(x), dim = -1)
        else:
            return self.model(x)

    def top_class(self, x):
        if self.np:
            return np.argmax(self.forward(x), axis=1)
        else: 
            return torch.argmax(self.forward(x), axis=1)

    def predict_flattened(self, x):
        '''
        This function takes FLATTENED inputs -
            so need to reshape before passing to forward.
        (It's for shap kernel explainer)
        '''
        # print(f"Shape of x in flattened pred fn: {np.shape(x)}")

        if torch.is_tensor(x):x = tensor2numpy(x)

        outs = self.forward([im.reshape(self.input_shape) for im in x])

        return outs.detach().numpy()

    def predict_flattened_softmax(self, x):
        '''
        takes flattened, numpy input and returns numpy output after softmax
        
        for SAGE explainer
        '''
        # print(f"Shape of x in flattened pred fn: {np.shape(x)}")

        output = self.forward(numpy2cuda(x).reshape((-1,) + self.input_shape))
        output = F.softmax(output, dim = 1)

        return tensor2numpy(output)


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



if __name__ == '__main__':
    #%%  Model Parameters ===================================================
    load_checkpoint = False
    num_epochs = 80
    batch_size = 128
    model_params = {
        'num_classes': 10
    }
    optimizer_params = {
        'lr': 0.001,
        'weight_decay': 0
    }
    scheduler_params = {
        'step_size': 60,
        'gamma': 0.1
    }

    # Set path for model and data locations
    path_data = './Files/Data'
    path_model = './Files/Models'


    #%%  Load Data =================================================
    train_set, train_loader, test_set, test_loader = load_data(path_data, batch_size = batch_size)



    #%%  Run Model ==================================================

    # Initialize Model
    device = torch.device("cuda:0" if torch.cuda.is_available()
                        else "cpu")  # Use GPU if available
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    # Initialize Optimizer
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    # Initialize Other Trackers
    StartTime = datetime.now()  # track training time
    # numpy array to track training/test accuracy by epoch
    save_accy = np.zeros((num_epochs + 1, 2))
    np.random.seed(1)
    torch.manual_seed(1)
    # Epoch to start calculations (should be zero unless loading previous model checkpoint)
    init_epoch = 0

    # If we are starting from a loaded checkpoint, load previous model paramters
    if load_checkpoint:
        checkpoint = torch.load(os.path.join(path_model, 'checkpoint.pt'))
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        init_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.to(device)


    for epoch in range(init_epoch, num_epochs):

        epoch_time_start = datetime.now()
        model.train()
        for batch_ID, (data, target) in tqdm(enumerate(train_loader), total = len(train_loader)):
            data, target = data.to(device), target.to(device)  # Move training data to GPU
            pred = model(data)      # Calculate Predictions
            loss = criterion(pred, target)  # Calculate Loss
            loss.backward()  # Calculate Gradients

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # Calculate Train/Test Accuracy
        test_accy = calc_test_accy(model, test_loader)
        train_accy = calc_train_accy(model, train_loader, 100, batch_size)
        # Save training / test accuracy
        save_accy[epoch, :] = [train_accy, test_accy]
        epoch_time = datetime.now() - epoch_time_start
        print("Epoch: %d; Loss: %f; Train_Accy: %f; Test_Accy: %f; Time: %s" %
            (epoch, loss, train_accy, test_accy, str(epoch_time)))

    #%% Evaluate Final Model ==================================================

    test_accy = calc_test_accy(model, test_loader)
    train_accy = calc_test_accy(model, train_loader)
    save_accy[-1, :] = [train_accy, test_accy]

    total_time = datetime.now() - StartTime  # Calculate total training / test time
    print("Train Accuracy: %f; Test Accuracy: %f; Total Time: %s; Epochs: %d" %
        (train_accy, test_accy, total_time, epoch))
    # Save final model to disk
    # torch.save(model, os.path.join(path_model, 'MLP_baseline_CIFAR10.pt'))
    torch.save(model.state_dict(), os.path.join(path_model, 'CIFAR10_baseline.pt'))
    np.savetxt(os.path.join(path_model, 'model_accuracy.csv'), save_accy,
            delimiter=",")  # Write training/test accuracy to disk
