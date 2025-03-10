


#%% Import Libraries ===============================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.autograd import Variable
from datetime import datetime
from torch.utils.data import Dataset
import os
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pickle

#%% Define Classes ===============================================


class NN(nn.Module):
    # Define ResNet

    def __init__(self, input_units, hidden_units, num_classes):
        super(NN, self).__init__()

        self.input_units = input_units
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.input_units, self.hidden_units) 
        self.a1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.a2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_units, self.num_classes)
        self.np = False
        self.softmax = False

    def forward(self, x):
        if type(x) == np.ndarray:
            x = numpy2cuda(x)
        # x = F.relu(self.fc1(x))
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.fc3(x)
        if self.softmax:
            x = F.sigmoid(x)

        return x

    def top_class(self, x):
        if self.softmax:
            threshold = 0.5
        else:
            threshold = 0
        return (self.forward(x)>=threshold)*1

    def predict_flattened(self, x):
        '''
        This function takes FLATTENED inputs -
            so need to reshape before passing to forward.
        (It's for shap kernel explainer)
        '''
        # print(f"Shape of x in flattened pred fn: {np.shape(x)}")

        outs = self.forward(x)
        return outs.detach().numpy()

    def predict_flattened_softmax(self, x):
        '''
        takes flattened, numpy input and returns numpy output after softmax
        
        for SAGE explainer
        '''
        # print(f"Shape of x in flattened pred fn: {np.shape(x)}")

        output = self.forward(x)
        output = F.sigmoid(output)

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

def save_dict(dictionary, path):
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary




class nhanes(Dataset):

    def __init__(self, data_path = './Files/Data/', train = True):
        import h5py


        if train:
            self.data = torch.tensor(np.loadtxt(os.path.join(data_path, 'nhanes_x_train.csv'), delimiter = ',')).float()
            self.labels = torch.tensor(np.loadtxt(os.path.join(data_path, 'nhanes_y_train.csv'), delimiter = ',')).int()

        else:
            self.data = torch.tensor(np.loadtxt(os.path.join(data_path, 'nhanes_x_test.csv'), delimiter = ',')).float()
            self.labels = torch.tensor(np.loadtxt(os.path.join(data_path, 'nhanes_y_test.csv'), delimiter = ',')).int()


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx, :], self.labels[idx]
# Define function for calculating predictions on test data

def calc_test_accy(model, test_loader):
    model.eval()   # Set model into evaluation mode
    epoch_accy = 0
    counter = 0
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(
                device)  # Move data to GPU
            pred = model(data)   # Calculate Output
            prediction = pred >= 0.0
            truth = target >= 0.5
            acc = prediction.reshape(-1).eq(truth).sum().cpu().data.numpy()
            epoch_accy += acc
            counter += target.shape[0]
        epoch_accy /= counter
        return epoch_accy 

def load_data(path_data = './', batch_size = 32):

    train_set = nhanes(data_path = path_data, train = True)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=len(train_set), shuffle=True, num_workers=0)

    test_set = nhanes(data_path = path_data, train = False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=len(test_set), shuffle=False, num_workers=0)

    return train_set, train_loader, test_set, test_loader


if __name__ == '__main__':
    # Set path for model and data locations
    os.chdir('/work/jdy/davin/WasG/')
    path_data = './Files/Data/'
    path_model = './Files/Models'

    #%%  Model Parameters ===================================================
    load_checkpoint = False
    num_epochs = 100
    batch_size = 100
    optimizer_params = {
        'lr':1e-1,
        'weight_decay': 0.5,
        'momentum': 0
    }
    scheduler_params = {
        'step_size': 20,
        'gamma': 0.1
    }
    model_params = {
        'input_units':27,
        'hidden_units':50,
        'num_classes':1
    }


    #%%  Load Data =================================================

    train_set, train_loader, test_set, test_loader = load_data(path_data, batch_size = batch_size)


    #%%  Run Model ==================================================

    # Initialize Model
    device = torch.device("cuda:0" if torch.cuda.is_available()
                        else "cpu")  # Use GPU if available
    model = NN(**model_params).to(device)

    # Initialize Optimizer
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    criterion = nn.BCEWithLogitsLoss()
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
        for batch_ID, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).double()  # Move training data to GPU

            pred = model(data)      # Calculate Predictions
            loss = criterion(pred, target.unsqueeze(-1))  # Calculate Loss
            loss.backward()  # Calculate Gradients

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # Calculate Train/Test Accuracy
        test_accy = calc_test_accy(model, test_loader)
        train_accy = calc_test_accy(model, train_loader)
        # Save training / test accuracy
        save_accy[epoch, :] = [train_accy, test_accy]
        epoch_time = datetime.now() - epoch_time_start
        print("Epoch: %d || Loss: %f || Train_Accy: %f || Test_Accy: %f || Time: %s" %
            (epoch, loss, train_accy, test_accy, str(epoch_time)))

        # Save Checkpoint
        if epoch % 5 == 0:
            checkpoint = {'epoch': epoch,
                        'model': NN(**model_params),
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
    print('Train AUROC: %s' % str(metrics.roc_auc_score(train_set.labels.numpy(), model.predict_flattened_softmax(train_set.data))))
    print('Test AUROC: %s' % str(metrics.roc_auc_score(test_set.labels.numpy(), model.predict_flattened_softmax(test_set.data))))
    # Save final model to disk
    # torch.save(model, os.path.join(path_model, 'nhanes_model.pt'))
    # torch.save(model.state_dict(), os.path.join(path_model, 'nhanes_model.pt'))
