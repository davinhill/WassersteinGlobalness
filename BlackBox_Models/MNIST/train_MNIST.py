


#%% Import Libraries ===============================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.autograd import Variable
from datetime import datetime
from tqdm import tqdm
import os



class NN_MNIST(nn.Module):
    # Define ResNet

    def __init__(self, input_units, num_classes, hidden_units):
        super(NN_MNIST, self).__init__()

        self.np = False # output preds in np type
        self.softmax = False
        self.input_units = input_units
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 200, kernel_size = 6, stride = 2, padding = 0) # 12 x 12 x 300
        self.conv2 = nn.Conv2d(in_channels = 200, out_channels = 200, kernel_size = 6, stride = 2, padding = 0) # 4 x 4 x 500

        self.bn1 = nn.BatchNorm2d(200)
        self.fc1 = nn.Linear(3200, self.num_classes)

        self.a1 = nn.ReLU()
        self.a2 = nn.ReLU()

    def forward(self, x):
        if type(x) == list:
            x = numpy2cuda(np.array(x))
        x = self.a1(self.bn1(self.conv1(x)))
        x = self.a2(self.conv2(x))

        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)
        if self.softmax:
            x = F.softmax(x, dim = -1)

        return x

    def rgb_predict(self, x):
        from skimage.color import rgb2gray
        x = rgb2gray(x)
        if len(np.shape(x)) < 4:
            x = np.expand_dims(x, 1)
        return self.forward(x)

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

        if torch.is_tensor(x):
            x = x.detach().numpy()

        outs = self.forward([im.reshape(1, 28, 28) for im in x])

        return outs.detach().numpy()

    def predict_flattened_softmax(self, x):
        '''
        takes flattened, numpy input and returns numpy output after softmax
        
        for SAGE explainer
        '''
        # print(f"Shape of x in flattened pred fn: {np.shape(x)}")

        output = self.forward(numpy2cuda(x).reshape(-1,1, 28, 28))
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


# Define function for calculating predictions on test data

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
# n_batches indicates how many training observations to use for training accuracy calculation
# This reduces computation time, instead of using the entire training set.
def calc_train_accy(model, dataloader, n_batches, batch_size):
    model.eval()
    correct = 0
    with torch.no_grad():
        data_iterator = iter(dataloader)
        for i in range(n_batches):  # iterate for the specified number of batches
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

    return (100.*correct/(n_batches * batch_size))


def load_data(path_data = './', batch_size = 32):

    # Define Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ]),

        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
    }


    # Training Data
    train_set = datasets.MNIST(
        root=path_data, train=True, download=True, transform=data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # Test Data
    test_set = datasets.MNIST(
        root=path_data, train=False, download=True, transform=data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_set, train_loader, test_set, test_loader


if __name__ == "__main__":

    # Set path for model and data locations
    path_data = './Files/Data'
    path_model = './Files/Models'

    #%%  Model Parameters ===================================================
    load_checkpoint = False
    num_epochs = 10
    batch_size = 32
    optimizer_params = {
        'lr': 0.01,
        'weight_decay': 0,
        'momentum': 0
    }
    scheduler_params = {
        'step_size': 8,
        'gamma': 0.1
    }
    model_params = {
        'input_units':28*28,
        'num_classes':10,
        'hidden_units':500,
    }


    #%%  Load Data =================================================
    train_set, train_loader, test_set, test_loader = load_data(path_data, batch_size = batch_size)


    #%%  Run Model ==================================================

    # Initialize Model
    device = torch.device("cuda:0" if torch.cuda.is_available()
                        else "cpu")  # Use GPU if available
    model = NN_MNIST(**model_params).to(device)

    # Initialize Optimizer
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
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
    n_batches = np.int(np.floor((len(train_set)*1.)/batch_size))

    #~~~~~~~~~~~~~~~~~~~~~~
    # If we are starting from a loaded checkpoint, load previous model paramters
    if load_checkpoint:
        checkpoint = torch.load(os.path.join(path_model, 'checkpoint.pt'))
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        init_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.to(device)
    #~~~~~~~~~~~~~~~~~~~~~~~

    for epoch in range(init_epoch, num_epochs):

        epoch_time_start = datetime.now()
        model.train()
        for batch_ID, (data, target) in tqdm(enumerate(train_loader), total = n_batches):
            data, target = data.to(device), target.to(device)  # Move training data to GPU
            pred = model(data)      # Calculate Predictions
            loss = criterion(pred, target)  # Calculate Loss
            loss.backward()  # Calculate Gradients

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # Calculate Train/Test Accuracy
        test_accy = calc_test_accy(model, test_loader)
        train_accy = calc_train_accy(model, train_loader, np.minimum(25, n_batches), batch_size)
        # Save training / test accuracy
        save_accy[epoch, :] = [train_accy, test_accy]
        epoch_time = datetime.now() - epoch_time_start
        print("Epoch: %d || Loss: %f || Train_Accy: %f || Test_Accy: %f || Time: %s" %
            (epoch, loss, train_accy, test_accy, str(epoch_time)))

        # Save Checkpoint
        if epoch % 5 == 0:
            checkpoint = {'epoch': epoch,
                        'model': NN_MNIST(**model_params),
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                        }

            # Write Checkpoint to disk
            torch.save(checkpoint, os.path.join(path_model, 'checkpoint.pt'))
            np.savetxt(os.path.join(path_model, 'model_accuracy.csv'),
                    save_accy, delimiter=",")  # Write training/test accuracy to disk

    #%% Evaluate Final Model ==================================================

    test_accy = calc_test_accy(model, test_loader)
    train_accy = calc_test_accy(model, train_loader)
    save_accy[-1, :] = [train_accy, test_accy]

    total_time = datetime.now() - StartTime  # Calculate total training / test time
    print("Train Accuracy: %f; Test Accuracy: %f; Total Time: %s; Epochs: %d" %
        (train_accy, test_accy, total_time, epoch))
    # Save final model to disk
    # torch.save(model, os.path.join(path_model, 'MLP_baseline.pt'))
    torch.save(model.state_dict(), os.path.join(path_model, 'MLP_baseline.pt'))
    np.savetxt(os.path.join(path_model, 'model_accuracy.csv'), save_accy,
            delimiter=",")  # Write training/test accuracy to disk
