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
from sklearn.metrics import * 
import warnings
warnings.filterwarnings("ignore")

sns.set()

# Hyperparameters
batch_size = 64
input_size = 3
hidden_size = 16
num_classes = 1             # 회귀 문제 
learning_rate = 0.1
epochs = 100

def RUL_metric(y_valid, y_pred, threshold=10):
    error = y_valid.reshape(-1) - y_pred.reshape(-1)
    per = np.where(error<=threshold, 1, 0)
    return sum(per) / len(per)

class Pipeline:
    def __init__(self, scalar):
        self.scalar = scalar
    
    def fit(self, X, y):
        X = self.scalar.fit_transform(X)
        return X, y.to_numpy(dtype=np.float32)
    
    def transform(self, X, y):
        X = self.scalar.transform(X)
        return X, y.to_numpy(dtype=np.float32)

class BatteryDataSet(Dataset):

    def __init__(self, X,y):
        # Data loading
        # self.x = torch.from_numpy(X)
        # self.y = torch.from_numpy(y)
        self.x = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # len(Dataset)
        return self.n_samples


def classifyer(train_dataset, valid_dataset, test_dataset, batch_size, shuffle_dataset=False):

    # get the dataset size
    # get the indices
    train_indices = list(range(len(train_dataset)))
    valid_indices = list(range(len(valid_dataset)))
    test_indices = list(range(len(test_dataset)))

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


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l5 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l5(out)
        return out

# Training function
def train_loop(epoch, model, criterion, optimizer):
    size = len(X_train_s)
    optimizer.zero_grad()
    outputs = model(X_train_s)
    loss = criterion(outputs, y_train_s)
    loss.backward()
    optimizer.step()
    print(f'Epoch : {epoch} / AVG train loss : {loss.item()/size}')
    train_loss_list.append(loss.item()/size)
    
# Test function
def val_loop(epoch, model, scheduler):
    
    with torch.no_grad():
        size = len(X_val_s)
        pred = model(X_val_s)
        test_loss = criterion(pred, y_val_s)
        
        test_loss_list.append(test_loss.item()/size)
        pred_np = pred.squeeze().tolist()
        target_np = y_val_s.squeeze().tolist()
        
        try:
            for i,j in zip(pred_np,target_np):
                pred_list.append(i)
                targets_list.append(j)
        except:
            pass
        
        pred = pred.detach().cpu().numpy()
        y = y_val_s.detach().cpu().numpy()
    
        RMSE = mean_squared_error(y, pred, squared=False)
        MAE = mean_absolute_error(y, pred)
        MAPE = mean_absolute_percentage_error(y, pred)
        R2 = r2_score(y, pred)
        RUL_score = RUL_metric(y, pred, threshold=100)
        msg = f"Test: \n Avg loss: {test_loss.item()/size:>8f}, \n"
        print(msg) 
        msg = f"- RMSE: {(RMSE):>0.2f}, - MAE: {(MAE):>0.2f}, - MAPE: {(MAPE):>0.2f},  - R2: {(R2):>0.2f},  - RUL_metric: {(RUL_score):>0.2f}, \n"
        print(msg)
        print('---'*20)
        
        scheduler.step(test_loss)

    epoch_list.append(epoch)
    MAE_list.append(MAE)
    RUL_score_list.append(RUL_score*100)
    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.scatter(targets_list, pred_list)
    plt.xlabel('Target', fontsize=10)
    plt.ylabel('Prediction', fontsize=10)
    plt.ylim(0, 1300)
    plt.title(f"Epoch {epoch+1}", fontsize=13)

    plt.subplot(3,1,2)
    # plt.scatter(epoch_list, MAE_list)
    plt.plot(MAE_list, label='RUL_score', marker = '.')
    plt.ylim(0, 1000)
    plt.xlabel('Epoch')
    plt.ylabel('Target-Pred MAE')

    # PLOT Difference
    plt.subplot(3,1,3)
    # plt.scatter(epoch_list, RUL_score_list)
    plt.plot(RUL_score_list, label='RUL_score', marker = '.')
    plt.ylim(0, 100)
    plt.xlabel('Epoch')
    plt.ylabel('Target-Pred RUL_score (%)')
    # plt.scatter(epoch, test_loss)
    plt.tight_layout()

    plt.savefig(f'./result/training_{epoch}.png')


if __name__ == "__main__":

    # Import data
    path = "data/Battery_RUL.csv"
    data = pd.read_csv(path)
    data=data.drop(['Cycle_Index','Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Time constant current (s)','Charging time (s)'],axis=1)
    # RUL이 0인 값은 비정상 데이터로 빼버린다.
    data = data[data['RUL']!=0]
    X = data.drop(['RUL'], axis=1)
    y = data['RUL']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2023, shuffle =True)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.2, random_state=2023, shuffle =True)

    scaler = RobustScaler()
    pipeline = Pipeline(scaler)
    
    X_train_s, y_train_s = pipeline.fit(X_train, y_train)
    X_val_s, y_val_s = pipeline.transform(X_val, y_val)
    X_test_s, y_test_s = pipeline.transform(X_test, y_test)

    X_train_s = torch.FloatTensor(X_train_s)
    y_train_s = torch.FloatTensor(y_train_s).unsqueeze(-1)

    X_val_s = torch.FloatTensor(X_val_s)
    y_val_s = torch.FloatTensor(y_val_s).unsqueeze(-1)

    X_test_s = torch.FloatTensor(X_test_s)
    y_test_s = torch.FloatTensor(y_test_s).unsqueeze(-1)

    input_size =  X_train_s.shape[1] #num of columns
    hidden_size = 16
    num_classes = 1
    model = NeuralNet(input_size, hidden_size, num_classes)

    # Loss function
    criterion = nn.MSELoss()  # nn.L1Loss()       # nn.MSELoss()->L2loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    train_loss_list = []
    test_loss_list = []
    min_diff_dict = {}

    targets_list = []
    pred_list = []
    epoch_list = []
    MAE_list= []
    RUL_score_list = []

    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train_loop(epoch, model, criterion, optimizer)

        val_loop(epoch, model, scheduler)

    print("Fertig!")


# Save model
# torch.save(NeuralNet.state_dict(), os.getcwd() + '/Datasets/FF_Net_1.pth')
            
            

