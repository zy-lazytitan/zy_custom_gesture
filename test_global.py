import numpy as np

train_global = np.load('./data/14/train_global_dert.npy')
train_label = np.load('./data/14/train_label_and_frame.npy')

val_global = np.load('./data/14/val_global_dert.npy')
val_label = np.load('./data/14/val_label_and_frame.npy')

from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from tqdm import tqdm

d = {}
hulls = {}
for i in tqdm(range(train_global.shape[0])):
    label = train_label[i][0]
    frame = train_label[i][1]
    data = train_global[i, :frame-1, :3]
    if label in d.keys():
        for j in range(frame-1):
            d[label].append(data[j])
    else:
        d[label] = []
        for j in range(frame-1):
            d[label].append(data[j])

for label in d.keys():
    vectors = np.array(d[label], dtype=np.float32)
    print(vectors.shape)
    hull = ConvexHull(vectors)
    hulls[label] = hull

# 创建一个 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
for label in d.keys():
    points = np.array(d[label], dtype=np.float32)
    hull = hulls[label]

    print(colors[label])
    # 使用颜色映射表示第四个维度
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[label], cmap=cm.jet, marker='o', label='Data Points')

    # 添加颜色映射的 colorbar
    cbar = plt.colorbar(sc)

    # 绘制凸包的边界
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')
        
    if label == 2:
        break

# 设置图的标题和标签
ax.set_title('Convex Hull Visualization (3D with Color Mapping)')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# 显示图例
ax.legend(loc='best')

# 显示 3D 图形
plt.show()