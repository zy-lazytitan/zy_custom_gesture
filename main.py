import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
# from dtw import *
import argparse
import os
from tqdm import tqdm
import time

import matplotlib.pyplot as plt

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

dtype = torch.float32

def my_dtw(a, b):
    len1 = a.shape[0]
    len2 = b.shape[0]

    # 初始化距离矩阵
    distance_matrix = torch.tensor(np.zeros((len1, len2)), dtype=dtype, device=device)
    
    for i in range(len1):
        for j in range(len2):
            cost = torch.norm(a[i, :]-b[j, :], p=2)
            if i == 0 and j == 0:
                distance_matrix[i, j] = cost
            elif i == 0:
                distance_matrix[i, j] = cost + distance_matrix[i, j-1]
            elif j == 0:
                distance_matrix[i, j] = cost + distance_matrix[i-1, j]
            else:
                distance_matrix[i, j] = cost + min(distance_matrix[i-1, j], distance_matrix[i, j-1], distance_matrix[i-1, j-1])
    
    return distance_matrix[-1, -1]

def get_cos(a, b):
    angle_cos = a.dot(b) / (torch.sqrt(a.dot(a)) * torch.sqrt(b.dot(b)))
    if angle_cos < -1:
        return torch.tensor(-1, device=device, dtype=dtype)
    if angle_cos > 1:
        return torch.tensor(1, device=device, dtype=dtype)
    return angle_cos

def get_bone_distance(data1, data2):
    cosine_sim = torch.nn.functional.cosine_similarity(data1, data2, dim=1)
    # print(cosine_sim)
    cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
    # print(cosine_sim)
    squared_cosine_sim = 1 - cosine_sim ** 2
    # print(squared_cosine_sim)
    result = torch.sqrt(torch.sum(squared_cosine_sim))
    # print(result)
    return result

class Mydata(Dataset):
    def __init__(self, data_path, label_path):
        self.angles = torch.tensor(np.load(data_path), dtype=dtype, device=device)
        print(self.angles.shape)
        self.N = self.angles.shape[0]
        self.labels = torch.tensor(np.load(label_path), dtype=torch.int32, device=device)
        print(self.labels.shape)

    def __getitem__(self, item):
        x = self.angles[item]
        label = self.labels[item]
        return x

    def __len__(self):
        return self.N
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()

        # 创建隐藏层
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.hidden_layers(x)
    
class Udf_loss(nn.Module):
    def __init__(self):
        super(Udf_loss, self).__init__()
    
    def forward(self, predictions, data):
        N = predictions.shape[0]
        loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        num = 0
        # print(loss)
        for i in range(N-1):
            for j in range(i, N):
                distance = torch.dist(predictions[i], predictions[j], p=2)
                # dis = torch.sqrt(torch.abs(torch.prod((data[i] - data[j]) / 10)))
                # dis = torch.dist(data[i], data[j], p=2)
                dis = get_bone_distance(data[i], data[j])
                # print("distance:", distance)
                # print("dis:", dis)
                # print(self.alpha)
                # print(self.alpha * distance - dis)
                # print(torch.abs(self.alpha * distance - dis))
                loss = loss + torch.abs(self.alpha * distance - dis)
                # print("loss = ", loss)
                # print(loss)
                num += 1    
        loss /= num
     
        return loss
        
def train(model, optimizer, train_data, val_data, arg, myLoss, epochs=3, print_every=100, batch_size=64):
    losses = []
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model = model.to(device=device)
    for epoch in range(epochs):
        sum_loss = 0
        num = 0
        for batch in tqdm(train_loader, desc="loading"):
            # print(batch.shape)
            data_input = batch.view(batch.shape[0], -1)
            # print(data_input.shape)
            output = model(data_input)
            # print(output.shape)
            
            # print(output)
            
            loss = myLoss(output, batch)
            sum_loss += loss.item()
            
            # print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # return
            num += 1
        
            # losses.append(loss.item())
            
        mean_loss = sum_loss / num
        losses.append(mean_loss) 
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {mean_loss:.4f}')
        
        plt.figure()
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xlabel('epochs')
        plt.ylabel('loss')
        
        plt.plot(losses)
        plt.title('loss curve')
        plt.savefig('./runs/' + arg.benchmark + '/' + arg.name + "_loss_curve" + ".png")
        
        plt.close()
    
        state={}
        state['model_state'] = model.state_dict()
        state['optimizer'] = optimizer.state_dict()
        torch.save(state, './runs/' + arg.benchmark + '/' + arg.name + ".pth")

if __name__=="__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='28') # 14, 28
    parser.add_argument('--name', default='model.pth')

    arg = parser.parse_args()

    train_data_path = os.path.join('./data/', '{}/train_bone_data_no_adjust.npy'.format(arg.benchmark))
    # train_data_path = os.path.join('./data/', '{}/train_angle.npy'.format(arg.benchmark))
    train_label_path = os.path.join('./data/', '{}/train_label.npy'.format(arg.benchmark))
    train_data = Mydata(train_data_path, train_label_path)
    
    val_data_path = os.path.join('./data/', '{}/train_bone_data_no_adjust.npy'.format(arg.benchmark))
    val_label_path = os.path.join('./data/', '{}/val_label.npy'.format(arg.benchmark))
    val_data = Mydata(val_data_path, val_label_path)
    
    model = MLP(63, 512, 4, 10)
    # state = torch.load('./runs/28/no_adjust_bone_10_mlp_Adam_1e-4_batchsize_32.pth')
    # model.load_state_dict(state["model_state"])
    
    learning_rate = 1e-4    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    
    loss = Udf_loss().to(device=device)
    
    train(model, optimizer, train_data, val_data, arg, loss, epochs=100, batch_size=32)