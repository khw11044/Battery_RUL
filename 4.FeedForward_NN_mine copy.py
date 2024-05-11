import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

sns.set()

# Hyperparameters
batch_size = 32
input_size = 3
hidden_size = 10
num_classes = 1             # 회귀 문제 
learning_rate = 0.001
epochs = 50

class Pipeline:
    def __init__(self, scalar):
        self.scalar = scalar
    
    def fit(self, X, y):
        X = self.scalar.fit_transform(X)
        return X, y
    
    def transform(self, X, y):
        X = self.scalar.transform(X)
        return X, y

class BatteryDataSet(Dataset):

    def __init__(self, dataset):
        # Data loading
        self.x = torch.from_numpy(dataset[:, :-1])
        self.y = torch.from_numpy(dataset[:, [-1]])
        # X = dataset.drop(['RUL'], axis=1)
        # y = dataset['RUL']
        # self.x = torch.from_numpy(X)
        # self.y = torch.from_numpy(y)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # len(Dataset)
        return self.n_samples


def classifyer(train_dataset, valid_dataset, test_dataset, batch_size, shuffle_dataset=False):

    # get the dataset size
    train_dataset_len = len(train_dataset)
    valid_dataset_len = len(valid_dataset)
    test_dataset_len = len(test_dataset)

    # get the indices
    train_indices = list(range(train_dataset_len))
    valid_indices = list(range(valid_dataset_len))
    test_indices = list(range(test_dataset_len))

    # percentage share of data set
    # train:        ~ 70 %
    # test:         ~ 30 %

    # shuffle the dataset
    if shuffle_dataset:
        np.random.seed()
        np.random.shuffle(train_indices)
        np.random.shuffle(valid_indices)
        np.random.shuffle(test_indices)

    # set train dataset ot samplers and loader
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    # set valid dataset ot samplers and loader
    valid_sampler = SubsetRandomSampler(valid_indices)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    
    # set test dataset ot samplers and loader
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    return (train_loader, valid_loader, test_loader)


# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.l2 = nn.Linear(hidden_size, 50)
#         self.relu = nn.ReLU()
#         self.l3 = nn.Linear(50, 20)
#         self.relu = nn.ReLU()
#         self.l4 = nn.Linear(20, 5)
#         self.relu = nn.ReLU()
#         self.l5 = nn.Linear(5, num_classes)

#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         out = self.relu(out)
#         out = self.l3(out)
#         out = self.relu(out)
#         out = self.l4(out)
#         out = self.relu(out)
#         out = self.l5(out)
#         return out

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.l5 = nn.Linear(16, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l5(out)
        return out

# Training function
def train_loop(epoch, train_loader, model, loss_fn, optimizer):
    size = len(train_loader)
    with tqdm(train_loader) as pbar :
        for batch, (features, RUL) in enumerate(pbar):
            # Forward path
            outputs = model(features)           # ([32, 6])
            loss = loss_fn(outputs, RUL)

            # Backwards path
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            loss, current = loss.item(), batch*len(features)
            pbar.set_postfix({'loss' : "{loss:>7f}"})

    print('===> Epoch [{}] : loss : {:.5}'.format(epoch,  loss))

# Test function
def val_loop(epoch, dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    diff_list = []
    targets_list = []
    pred_list = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Difference between prediction and target
            diff = abs(y - pred) / y
            diff = diff.numpy()
            mean_diff = np.mean(diff)
            diff_list.append(mean_diff)

            # # Target vs prediction
            pred_np = pred.squeeze().tolist()
            target_np = y.squeeze().tolist()

            try:
                for i in pred_np:

                    pred_list.append(i)
                for i in target_np:
                    targets_list.append(i)
            except:
                pass

    # Average loss
    test_loss /= num_batches
    scheduler.step(test_loss)
    
    # Average difference
    difference_mean = np.mean(diff_list)

    # Print the average difference and average loss
    print(f"Test: \n Avg Difference: {(100*difference_mean):>0.2f}%, Avg loss: {test_loss:>8f} \n")

    # Minimum difference and its epoch
    min_diff_dict[epoch+1] = (difference_mean*100)
    min_diff_value = min(min_diff_dict.items(), key=lambda x:x[1])
    print("LOWEST DIFFERENCE AND EPOCH:")
    print(f"Epoch: {min_diff_value[0]}, diff: {min_diff_value[1]:>0.2f}%")

    # PLOT Target vs Prediction
    # if t % 10 == 0:

    # plt.rcParams["figure.dpi"] = 600
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.scatter(targets_list, pred_list)
    plt.xlabel('Target', fontsize=10)
    plt.ylabel('Prediction', fontsize=10)
    plt.ylim(0, 1300)
    plt.title(f"Epoch {epoch+1}", fontsize=13)
    # plt.show()


    # PLOT Difference
    plt.subplot(2,1,2)
    plt.scatter(epoch, difference_mean*100)
    plt.ylim(0, 70)
    plt.xlabel('Epoch')
    plt.ylabel('Target-Pred Difference (%)')
    plt.scatter(epoch, test_loss)
    plt.savefig(f'./result/training_{epoch+1}.png')


if __name__ == "__main__":

    # Import data
    path = "data/Battery_RUL.csv"
    data = pd.read_csv(path)
    data=data.drop(['Cycle_Index','Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Time constant current (s)','Charging time (s)'],axis=1)
    X = data.drop(['RUL'], axis=1)
    y = data['RUL']
    input_size = X.shape[1]
    scaler = RobustScaler()
    pipeline = Pipeline(scaler)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle =False)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.2, shuffle =False)
    
    X_train_s, y_train_s = pipeline.fit(X_train, y_train)
    X_val_s, y_val_s = pipeline.transform(X_val, y_val)
    X_test_s, y_test_s = pipeline.transform(X_test, y_test)
    
    train_dataset = pd.DataFrame(X_train_s).join(y_train_s).to_numpy(dtype=np.float32)
    valid_dataset = pd.DataFrame(X_val_s).join(y_val_s).to_numpy(dtype=np.float32)
    test_dataset = pd.DataFrame(X_test_s).join(y_test_s).to_numpy(dtype=np.float32)
    
    # Load dataset
    train_dataset = BatteryDataSet(train_dataset)
    valid_dataset = BatteryDataSet(valid_dataset)
    test_dataset = BatteryDataSet(test_dataset)

    # Train and test loader
    train_loader, valid_loader, test_loader = classifyer(train_dataset, valid_dataset, test_dataset, 
                                           batch_size=batch_size, shuffle_dataset=True)
    # Init model
    model = NeuralNet(input_size, hidden_size, num_classes)

    # Loss function
    loss_fn = nn.MSELoss()  # nn.L1Loss()       # nn.MSELoss()->L2loss

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # Auxiliary dictionary to store epochs and difference values:
    min_diff_dict = {}

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train_loop(epoch, train_loader, model, loss_fn, optimizer)

        val_loop(epoch, valid_loader, model, loss_fn)

    print("Fertig!")


# Save model
# torch.save(NeuralNet.state_dict(), os.getcwd() + '/Datasets/FF_Net_1.pth')
            
            

