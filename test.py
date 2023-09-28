import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler

import argparse
import os
from tqdm import tqdm
import time

from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from fastdtw import fastdtw

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

dtype = torch.float32
    
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
    
class MLP2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, hidden_size)
        self.fc10 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x))
        return self.output(x)
    
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
    
def generate_hull(model, train_data_path, train_label_path):
    train_data = torch.tensor(np.load(train_data_path), dtype=dtype, device=device)
    print(train_data.shape)
    train_data = train_data.view(train_data.shape[0], -1)
    print(train_data.shape)
    train_labels = np.load(train_label_path)
    print(train_labels.shape)
    
    output = model(train_data)
    print(output.shape)
    
    d = {}
    hulls = {}
    for i in tqdm(range(train_labels.shape[0])):
        label = train_labels[i]
        feature = output[i].cpu().detach().numpy()
        if label in d.keys():
            d[label].append(feature)
        else:
            d[label] = [feature]
            
    print(d.keys())      
            
    for label in d.keys():
        vectors = np.array(d[label], dtype=np.float32)
        print(vectors.shape)
        hull = ConvexHull(vectors)
        hulls[label] = hull

    print("generate over")
    return d, hulls    

def test(hulls, val_data_path, val_label_path, output_path):
    val_data = torch.tensor(np.load(val_data_path), dtype=dtype, device=device)
    print(val_data.shape)
    val_data = val_data.view(val_data.shape[0], -1)
    print(val_data.shape)
    val_labels = np.load(val_label_path)
    print(val_labels.shape)
    
    output = model(val_data)
    print(output.shape)
    
    file = open(output_path, 'w')
    
    num = val_labels.shape[0]
    pass_num = 0
    in_num = 0
    for i in tqdm(range(num)):
        val_label = val_labels[i]
        feature = output[i].cpu().detach().numpy()
        
        output_label = []
        for label in hulls.keys():
            hull = hulls[label]
            if all(hull.equations.dot(np.append(feature, 1.0)) <= 0):
                output_label.append(label)
        
        if val_label in output_label:
            in_num += 1
            if len(output_label) == 1:
                pass_num += 1
        file.write(str(val_label) + " " + ','.join(map(str, output_label)) + "\n")
           
    print(num, pass_num, pass_num / num, in_num, in_num / num)   
    
def visualize_hull(hulls, d, labels):
    # 创建一个 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    for label in labels:
        points = np.array(d[label], dtype=np.float32)
        hull = hulls[label]
        fourth_dimension = points[:, 3]

        # 使用颜色映射表示第四个维度
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[label], cmap=cm.jet, marker='o', label='Data Points')

        # 添加颜色映射的 colorbar
        cbar = plt.colorbar(sc)

        # 绘制凸包的边界
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')

    # 设置图的标题和标签
    ax.set_title('Convex Hull Visualization (3D with Color Mapping)')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # 显示图例
    ax.legend(loc='best')

    # 显示 3D 图形
    plt.show()
    
def visualize_first_frame(train_data_path, train_label_and_frame_path, model, start, kinds):
    train_data = torch.tensor(np.load(train_data_path), dtype=dtype, device=device)
    train_data = train_data.view(train_data.shape[0], -1)
    print(train_data.shape)
    train_labels = np.load(train_label_and_frame_path)
    
    d = {}
    index = start
    for i in tqdm(range(train_labels.shape[0])):
        label = train_labels[i, 0]
        data = train_data[index, :].view(-1)
        # print(data.shape)
        output = model(data).cpu().detach().numpy()
        if label in d.keys():
            d[label].append(output)
        else:
            d[label] = [output]
        frame = train_labels[i, 1]
        index += frame
    print(index)
    print(d.keys())
            
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = ['r', 'b', 'g']
    for label in kinds:
        points = np.array(d[label], dtype=np.float32)

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c[kinds.index(label)], cmap=cm.jet, marker='o', label='Data Points')

    # 设置图的标题和标签
    ax.set_title('Convex Hull Visualization (3D with Color Mapping)')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # 显示图例
    ax.legend(loc='best')

    # 显示 3D 图形
    plt.show()

# from sklearn.svm import SVC


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def test_with_knn(train_data_path, train_label_path, val_data_path, val_data_label, model):
    train_data = torch.tensor(np.load(train_data_path), dtype=dtype, device=device)
    print(train_data.shape)
    train_data = train_data.view(train_data.shape[0], -1)
    print(train_data.shape)
    train_labels = np.load(train_label_path)
    print(train_labels.shape)
    
    output = model(train_data)
    output = output.cpu().detach().numpy()
    print(output.shape)
    

    knn_classifier = KNeighborsClassifier(n_neighbors=100)  # 创建K最近邻分类器，设置邻居数量

    print("knn begin")
    # 训练分类器
    knn_classifier.fit(output, train_labels)  # X_train是训练数据集的特征，y_train是对应的标签
    print("knn over")
    
    val_data = torch.tensor(np.load(val_data_path), dtype=dtype, device=device)
    print(val_data.shape)
    val_data = val_data.view(val_data.shape[0], -1)
    print(val_data.shape)
    val_labels = np.load(val_data_label)
    print(val_labels.shape)
    
    val_output = model(val_data)
    val_output = val_output.cpu().detach().numpy()
    print(val_output.shape)
    
    print("predict begin")
    y_pred = knn_classifier.predict(val_output)  # X_test是测试数据集的特征
    print("predict over")
    accuracy = accuracy_score(val_labels, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
def test_time_line_knn(train_data_path, train_label_path, val_data_path, val_label_and_frame, model, output_path):
    train_data = torch.tensor(np.load(train_data_path), dtype=dtype, device=device)
    print(train_data.shape)
    train_data = train_data.view(train_data.shape[0], -1)
    print(train_data.shape)
    train_labels = np.load(train_label_path)
    print(train_labels.shape)
    
    output = model(train_data)
    output = output.cpu().detach().numpy()
    print(output.shape)
    

    knn_classifier = KNeighborsClassifier(n_neighbors=30)  # 创建K最近邻分类器，设置邻居数量

    print("knn begin")
    # 训练分类器
    knn_classifier.fit(output, train_labels)  # X_train是训练数据集的特征，y_train是对应的标签
    print("knn over")
    
    val_data = torch.tensor(np.load(val_data_path), dtype=dtype, device=device)
    print(val_data.shape)
    val_data = val_data.view(val_data.shape[0], -1)
    print(val_data.shape)
    val_labels = np.load(val_label_and_frame)
    print(val_labels.shape)
    
    val_output = model(val_data)
    val_output = val_output.cpu().detach().numpy()
    print(val_output.shape)
    
    index = 0
    accurate_count = 0
    file = open(output_path, 'w')
    for i in tqdm(range(val_labels.shape[0])):
        val_label = val_labels[i, 0]
        frame = val_labels[i, 1]
        # print(val_label)
        # print(frame)
        d = {}
        for j in range(index, index + frame):
            distances, indices = knn_classifier.kneighbors([val_output[j, :]], n_neighbors=30)
            labels = set([train_labels[i] for i in indices[0]])
            for label in labels:
                if label in d.keys():
                    d[label] += 1
                else:
                    d[label] = 1
        # print(d)
        pred_label = max(d, key=d.get)
        d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
        file.write(str(val_label) + " " + str(frame) + ": " + "pred = " + str(pred_label) + "\n")
        file.write(str(d))
        file.write("\n")
        # print(pred_label)
        if pred_label == val_label:
            accurate_count += 1
        index += frame       
        # break
    file.close()
    print(f"Accuracy: {(accurate_count / val_labels.shape[0] * 100):.2f}%")
    
def time_line_distance(x, y):
    x_frame = int(x[-1])
    y_frame = int(y[-1])
    # print(x_frame, y_frame)
    xx = x[:x_frame*3].reshape(x_frame, 3)
    yy = y[:y_frame*3].reshape(y_frame, 3)
    distance, path = fastdtw(xx, yy)
    return distance
    
def test_global_knn(train_global_path, train_label_path, val_global_path, val_label_path, output_path):
    train_global = np.load(train_global_path)
    print(train_global.shape)
    train_labels = np.load(train_label_path)
    print(train_labels.shape)
    
    train_data = np.zeros((train_global.shape[0], 180*3+1))
    for i in range(train_global.shape[0]):
        frame = train_labels[i, 1]
        global_data = train_global[i]
        train_data[i, :-1] = global_data.flatten()
        train_data[i, -1] = frame
    print(train_data.shape)
    knn_classifier = KNeighborsClassifier(n_neighbors=10, metric=time_line_distance)  # 创建K最近邻分类器，设置邻居数量

    print("knn begin")
    # 训练分类器
    knn_classifier.fit(train_data, train_labels) 
    print("knn over")
    
    val_global = np.load(val_global_path)
    print(val_global.shape)
    val_labels = np.load(val_label_path)
    print(val_labels.shape)
    
    val_data = np.zeros((val_global.shape[0], 180*3+1))
    for i in range(val_global.shape[0]):
        frame = val_labels[i, 1]
        global_data = val_global[i]
        val_data[i, :-1] = global_data.flatten()
        val_data[i, -1] = frame
    print(val_data.shape)
    
    print("predict begin")
    y_pred = knn_classifier.predict(val_data)  # X_test是测试数据集的特征
    print("predict over")
    accuracy = accuracy_score(val_labels[:, 0], y_pred[:, 0])
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./runs/28/no_adjust_bone_10_mlp_Adam_1e-4_batchsize_32.pth') # 14, 28
    parser.add_argument('--benchmark', default='28') # 14, 28
    parser.add_argument('--output_path', default='runs/28/no_adjust_bone_10_mlp_Adam_1e-4_batchsize_32.txt') # 14, 28

    arg = parser.parse_args()
    
    val_data_path = os.path.join('./data/', '{}/val_bone_data_no_adjust.npy'.format(arg.benchmark))
    val_label_path = os.path.join('./data/', '{}/val_label_and_frame.npy'.format(arg.benchmark))
    
    train_data_path = os.path.join('./data/', '{}/train_bone_data_no_adjust.npy'.format(arg.benchmark))
    train_label_path = os.path.join('./data/', '{}/train_label.npy'.format(arg.benchmark))
    
    state = torch.load(arg.model_path)
    
    model = MLP(63, 512, 4, 10)
    # model = MLP2(63, 512, 4)
    model.load_state_dict(state["model_state"])
    model = model.to(device=device)
    
    # train_data = Mydata(train_data_path, train_label_path)
    
    # train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    # model = model.to(device=device)
    # for batch in tqdm(train_loader, desc="loading"):
    #     # print(batch.shape)
    #     data_input = batch.view(batch.shape[0], -1)
    #     print(data_input.shape)
    #     output = model(data_input)
    #     # print(output)
    #     # break
    
    # d, hulls = generate_hull(model, train_data_path, train_label_path)
    
    # test(hulls, val_data_path, val_label_path, arg.output_path)
    # visualize_hull(hulls, d, range(14))
    
    # visualize_first_frame(train_data_path, train_label_path, model, 30, [18, 22])
    # test_with_knn(train_data_path, train_label_path, val_data_path, val_label_path, model)
    
    test_time_line_knn(train_data_path, train_label_path, val_data_path, val_label_path, model, arg.output_path)
    # test_global_knn(os.path.join('./data/', '{}/train_root_dert.npy'.format(arg.benchmark)), 
    #                 os.path.join('./data/', '{}/train_label_and_frame.npy'.format(arg.benchmark)), 
    #                 os.path.join('./data/', '{}/val_root_dert.npy'.format(arg.benchmark)),
    #                 os.path.join('./data/', '{}/val_label_and_frame.npy'.format(arg.benchmark)), 
    #                 "runs/28/root_dert.txt")