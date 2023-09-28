import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

data_path1 = '../SHREC2017/gesture_1/finger_2/subject_1/essai_1/skeletons_world.txt'
data_path2 = '../SHREC2017/gesture_9/finger_2/subject_1/essai_1/skeletons_world.txt'



paris = ((2, 1),
        (3, 1), (4, 3), (5, 4), (6, 5),
        (7, 2), (8, 7), (9, 8), (10, 9),
        (11, 2), (12, 11), (13, 12), (14, 13),
        (15, 2), (16, 15), (17, 16), (18, 17),
        (19, 2), (20, 19), (21, 20), (22, 21))

def draw_hand(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for d in data:
        # 绘制散点图
        ax.scatter(d[0, 0], d[0, 1], d[0, 2], c='g', marker='o')
        ax.scatter(d[1:, 0], d[1:, 1], d[1:, 2], c='b', marker='o')

        # 连接某些点
        for pair in paris:
            x = d[pair[0]-1, :]
            point1 = d[pair[1]-1, :]
            connect_points = [(x[0], x[1], x[2]), (point1[0], point1[1], point1[2])]  # 连接第一个点和第四个点
            for point in connect_points:
                ax.plot([x[0], point[0]], [x[1], point[1]], [x[2], point[2]], c='r', linestyle='--')

    # 设置坐标轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 显示图形
    plt.show()
    
def get_rotate_matrix(pre_vector, new_vector):
     # 计算旋转轴
    rotation_axis = np.cross(pre_vector, new_vector)
    rotation_axis /= np.linalg.norm(rotation_axis)

    # 计算旋转角度
    angle = np.arccos(np.dot(pre_vector, new_vector))

    # 创建旋转矩阵
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = rotation_axis

    rotation_matrix = np.array([
        [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
        [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
        [t * x * z - y * s, t * y * z + x * s, t * z * z + c]
    ])
    
    return rotation_matrix
    

def adjust(data):
    new_frame = np.zeros((22, 3), dtype=np.float64)
    new_frame[0, :] = data[0, :]
    
    new_frame[1, :2] = data[0, :2]
    new_frame[1, 2] = data[0, 2] + np.linalg.norm(data[1, :] - data[0, :])
    
    palm_rotate_matrix = get_rotate_matrix(data[1, :] - data[0, :], new_frame[1, :] - new_frame[0, :])    
    for i in range(2, 22):
        new_frame[i, :] = new_frame[0, :] + np.dot(palm_rotate_matrix, (data[i, :] - data[0, :]))
        # new_frame[i, :] = new_frame[1, :] + data[i, :] - data[1, :]
        
    # new_frame[2, :] = new_frame[1, :] + data[2, :] - data[1, :]
    # thumb_rotate_matrix = get_rotate_matrix(data[2, :] - data[0, :], new_frame[2, :] - new_frame[0, :])
    # for i in range(3, 6):
    #     new_frame[i, :] = new_frame[0, :] + np.dot(thumb_rotate_matrix, (data[i, :] - data[0, :]))
        
    return new_frame
    
if __name__=="__main__":
    data1 = np.loadtxt(data_path1, dtype=np.float64)
    data1 = np.reshape(data1, (-1, 22, 3)) # (num_frame, num_joint, 3)
    frame1 = data1[20]

    data2 = np.loadtxt(data_path2, dtype=np.float64)
    data2 = np.reshape(data2, (-1, 22, 3)) # (num_frame, num_joint, 3)
    frame2 = data2[20]
    
    new_frame1 = adjust(frame1)
    new_frame2 = adjust(frame2)
    draw_hand([frame2, new_frame2])