import argparse
import pickle
from tqdm import tqdm
import os

gesture = ['Grab',
           'Tap',
           'Expand',
           'Pinch',
           'RotationClockwise',
           'RotationCounterClockwise',
           'SwipeRight',
           'SwipeLeft',
           'SwipeUp',
           'SwipeDown',
           'SwipeX',
           'Swipe+',
           'SwipeV',
           'Shake']
max_body_true = 1
max_body_kinect = 1
num_joint = 22
max_frame = 180

import numpy as np
import os

mid_joints = ((3, 4), 
              (7, 8), 
              (11, 12), 
              (15, 16),
              (19, 20))
first_joints = (3, 7, 11, 15, 19)
base_joints = (2, 6, 10, 14, 18)

# (num, 2) -> label and frame
def gen_label_and_frame(data_path, out_path, benchmark, part):
    sample_label = []
    idx_filename = os.path.join(data_path, "{}_gestures.txt".format('test' if part=='val' else part))
    data_idx = np.loadtxt(idx_filename, dtype=np.int32)
    # get data folder and label
    for id_gesture, id_finger, id_subject, id_essai, label_14, label_28, size_sequence in data_idx:
        filename = os.path.join(data_path, 'gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt'.format(
            id_gesture, id_finger, id_subject, id_essai))
        data = np.loadtxt(filename, dtype=np.float64)
        data = np.reshape(data, (-1, 22, 3)) # (num_frame, num_joint, 3)
        num_frame = data.shape[0]
        if benchmark == '14':
            sample_label.append([label_14-1, num_frame])
        else:
            sample_label.append([label_28-1, num_frame])
    
    label = np.array(sample_label, dtype=np.int32)
    print(label.shape)
    np.save('{}/{}_label_and_frame.npy'.format(out_path, part), label)

def get_angle(a, b):
    cos_angle = a.dot(b) / (np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))
    if cos_angle >= 1 or cos_angle <= -1:
        print(a, b, cos_angle, "error")
    angle = np.arccos(cos_angle) / np.pi * 180
    return angle

def get_angle_15(data):
    num_frame = data.shape[0]
    output = np.zeros((num_frame, 15), dtype=np.float64)
    for i in range(num_frame):
        for finger_idx in range(len(mid_joints)):
            base_joint = base_joints[finger_idx]
            angle = get_angle((data[i, 1, :] - data[i, base_joint, :]), 
                              (data[i, base_joint + 1, :] - data[i, base_joint, :]))
            output[i, 3 * finger_idx] = angle
            
            mid_joint_pair = mid_joints[finger_idx]
            angle = get_angle((data[i, mid_joint_pair[0] - 1, :] - data[i, mid_joint_pair[0], :]), 
                                (data[i, mid_joint_pair[0] + 1, :] - data[i, mid_joint_pair[0], :]))
            output[i, 3 * finger_idx + 1] = angle    

            angle = get_angle((data[i, mid_joint_pair[1] - 1, :] - data[i, mid_joint_pair[1], :]), 
                    (data[i, mid_joint_pair[1] + 1, :] - data[i, mid_joint_pair[1], :]))
            output[i, 3 * finger_idx + 2] = angle  
    return output 

# (all frame, 15) -> 5 fingers, angle from base joint to root joint, first joint angle, second joint angle
def gen_angle_15(data_path, out_path, benchmark, part, feature):
    idx_filename = os.path.join(data_path, "{}_gestures.txt".format('test' if part=='val' else part))
    data_idx = np.loadtxt(idx_filename, dtype=np.int32)

    f = []

    sum_frame = 0
    for i, (id_gesture, id_finger, id_subject, id_essai, label_14, label_28, size_sequence) in enumerate(tqdm(data_idx)):
        filename = os.path.join(data_path, 'gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt'.format(
            id_gesture, id_finger, id_subject, id_essai))
        data = np.loadtxt(filename, dtype=np.float64)
        data = np.reshape(data, (-1, 22, 3)) # (num_frame, num_joint, 3)
        num_frame = data.shape[0]
        output = get_angle_15(data)
        #data: (frame, feature)
        sum_frame += num_frame
        f.append(output)

    f_angles = np.zeros((sum_frame, 15), dtype=np.float64)
    start = 0
    for angle in f:
        frame = angle.shape[0]
        f_angles[start:start + frame, :] = angle
        start += frame
    
    print(f_angles.shape)
    np.save('{}/{}_angle_15.npy'.format(out_path, part), f_angles)
 
def get_cos(a, b):
    return a.dot(b) / (np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))
    
def get_cos_angle_25(data):
    num_frame = data.shape[0]
    output = np.zeros((num_frame, 25), dtype=np.float64)
    for i in range(num_frame):
        for finger_idx in range(len(mid_joints)):
            first_joint = first_joints[finger_idx]
            output[i, 5 * finger_idx: 5 * finger_idx + 3] = data[i, first_joint, :] - data[i, 1, :]
            
            mid_joint_pair = mid_joints[finger_idx]
            angle = get_cos((data[i, mid_joint_pair[0] - 1, :] - data[i, mid_joint_pair[0], :]), 
                                (data[i, mid_joint_pair[0] + 1, :] - data[i, mid_joint_pair[0], :]))
            output[i, 5 * finger_idx + 3] = angle    

            angle = get_cos((data[i, mid_joint_pair[1] - 1, :] - data[i, mid_joint_pair[1], :]), 
                    (data[i, mid_joint_pair[1] + 1, :] - data[i, mid_joint_pair[1], :]))
            output[i, 5 * finger_idx + 4] = angle      
    return output 

# 3d vector + angle + angle, perhaps have bug
# (all frame, 25) -> 5 fingers, 3d base joint - root joint, 1d first joint angle's cos, 1d second joint angle's cos
def gen_cos_angle_25(data_path, out_path, benchmark, part):
    idx_filename = os.path.join(data_path, "{}_gestures.txt".format('test' if part=='val' else part))
    data_idx = np.loadtxt(idx_filename, dtype=np.int32)

    f = []

    sum_frame = 0
    for i, (id_gesture, id_finger, id_subject, id_essai, label_14, label_28, size_sequence) in enumerate(tqdm(data_idx)):
        filename = os.path.join(data_path, 'gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt'.format(
            id_gesture, id_finger, id_subject, id_essai))
        data = np.loadtxt(filename, dtype=np.float64)
        data = np.reshape(data, (-1, 22, 3)) # (num_frame, num_joint, 3)
        num_frame = data.shape[0]
        output = get_cos_angle_25(data)
        #data: (frame, feature)
        sum_frame += num_frame
        f.append(output)

    f_angles = np.zeros((sum_frame, 25), dtype=np.float64)
    start = 0
    for angle in f:
        frame = angle.shape[0]
        f_angles[start:start + frame, :] = angle
        start += frame
    
    print(f_angles.shape)
    np.save('{}/{}_cos_angle_25.npy'.format(out_path, part), f_angles)

def get_vector_45(data):
    num_frame = data.shape[0]
    output = np.zeros((num_frame, 45), dtype=np.float64)
    for i in range(num_frame):
        for finger_idx in range(len(mid_joints)):
            first_joint = first_joints[finger_idx]
            output[i, 9 * finger_idx : 9 * finger_idx + 3] = data[i, first_joint, :] - data[i, 1, :]
            
            mid_joint_pair = mid_joints[finger_idx]
            output[i, 9 * finger_idx + 3 : 9 * finger_idx + 6] = data[i, mid_joint_pair[0] + 1, :] - data[i, mid_joint_pair[0] - 1, :]  
            output[i, 9 * finger_idx + 6 : 9 * finger_idx + 9] = data[i, mid_joint_pair[1] + 1, :] - data[i, mid_joint_pair[1] - 1, :]      
    return output 

# (all frame, 45) -> 5 fingers, base joint - root joint, second joint - base joint, fingertip - first joint
def gen_vector_45(data_path, out_path, benchmark, part):
    idx_filename = os.path.join(data_path, "{}_gestures.txt".format('test' if part=='val' else part))
    data_idx = np.loadtxt(idx_filename, dtype=np.int32)

    f = []

    sum_frame = 0
    for i, (id_gesture, id_finger, id_subject, id_essai, label_14, label_28, size_sequence) in enumerate(tqdm(data_idx)):
        filename = os.path.join(data_path, 'gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt'.format(
            id_gesture, id_finger, id_subject, id_essai))
        data = np.loadtxt(filename, dtype=np.float64)
        data = np.reshape(data, (-1, 22, 3)) # (num_frame, num_joint, 3)
        num_frame = data.shape[0]
        output = get_vector_45(data)
        #data: (frame, feature)
        sum_frame += num_frame
        f.append(output)

    f_angles = np.zeros((sum_frame, 45), dtype=np.float64)
    start = 0
    for angle in f:
        frame = angle.shape[0]
        f_angles[start:start + frame, :] = angle
        start += frame
    
    print(f_angles.shape)
    np.save('{}/{}_vector_45.npy'.format(out_path, part), f_angles)

def getQuaternion(fromVector, toVector):
        fromVector_e = fromVector / np.linalg.norm(fromVector)
 
        toVector_e = toVector / np.linalg.norm(toVector)
 
        cross = np.cross(fromVector_e, toVector_e)
 
        cross_e = cross / np.linalg.norm(cross)
 
        dot = np.dot(fromVector_e, toVector_e)
 
        angle = np.arccos(dot)
 
 
        # if angle == 0 or angle == np.pi:
        #     print("两个向量处于一条直线")
        #     return False
        # else:
        return [cross_e[0]*np.sin(angle/2), cross_e[1]*np.sin(angle/2), cross_e[2]*np.sin(angle/2), np.cos(angle/2)]
    
def get_global_dert(data):
    num_frame = data.shape[0]
    output = np.zeros((num_frame -1, 7), dtype=np.float64)
    for i in range(num_frame-1):
        output[i, :3] = data[i+1, 0, :] - data[i, 0, :]
        
        output[i, 3:] = getQuaternion(data[i, 1, :] - data[i, 0, :], 
                                      data[i+1, 1, :] - data[i+1, 0, :])
    # print(output)
    return output
    
# (num, frame, 7) -> 3d root position offset and 4d root bone rotation, i frame to i+1 frame
def gen_global_dert(data_path, out_path, benchmark, part):
    idx_filename = os.path.join(data_path, "{}_gestures.txt".format('test' if part=='val' else part))
    data_idx = np.loadtxt(idx_filename, dtype=np.int32)
    global_dert = np.zeros((data_idx.shape[0], 180, 7), dtype=np.float64)
    # get data folder and label
    for i, (id_gesture, id_finger, id_subject, id_essai, label_14, label_28, size_sequence) in enumerate(tqdm(data_idx)):
        filename = os.path.join(data_path, 'gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt'.format(
            id_gesture, id_finger, id_subject, id_essai))
        data = np.loadtxt(filename, dtype=np.float64)
        data = np.reshape(data, (-1, 22, 3)) # (num_frame, num_joint, 3)
        global_dert[i, :data.shape[0]-1, :] = get_global_dert(data)
        # break
    
    print(global_dert.shape)
    np.save('{}/{}_global_dert.npy'.format(out_path, part), global_dert)
    
pairs = ((2, 1),
        (3, 1), (4, 3), (5, 4), (6, 5),
        (7, 2), (8, 7), (9, 8), (10, 9),
        (11, 2), (12, 11), (13, 12), (14, 13),
        (15, 2), (16, 15), (17, 16), (18, 17),
        (19, 2), (20, 19), (21, 20), (22, 21))

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
        
    return new_frame
  
def get_bone_data(data):
    bone = np.zeros((21, 3), dtype=np.float64)
    for i in range(len(pairs)):
        pair = pairs[i]
        bone[i, :] = data[pair[0]-1, :] - data[pair[1]-1, :]
    return bone

# adjust the hand to z axis, perhaps have bug
# (all frame, 21, 3) -> 21 bone, 3d rotation = pair[1].position - pair[0].position     
def gen_bone(data_path, out_path, benchmark, part):
    idx_filename = os.path.join(data_path, "{}_gestures.txt".format('test' if part=='val' else part))
    data_idx = np.loadtxt(idx_filename, dtype=np.int32)
    
    bones = []
    for i, (id_gesture, id_finger, id_subject, id_essai, label_14, label_28, size_sequence) in enumerate(tqdm(data_idx)):
        filename = os.path.join(data_path, 'gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt'.format(
            id_gesture, id_finger, id_subject, id_essai))
        data = np.loadtxt(filename, dtype=np.float64)
        data = np.reshape(data, (-1, 22, 3)) # (num_frame, num_joint, 3)
        for frame in range(data.shape[0]):
            reg_data = adjust(data[frame])      
            output = get_bone_data(reg_data)
            bones.append(output)

    bone_data = np.array(bones, dtype=np.float64)
    print(bone_data.shape)
    np.save('{}/{}_bone_data.npy'.format(out_path, part), bone_data)

# (all frame, 21, 3) -> 21 bone, 3d rotation = pair[1].position - pair[0].position   
def gen_bone_no_adjust(data_path, out_path, benchmark, part):
    idx_filename = os.path.join(data_path, "{}_gestures.txt".format('test' if part=='val' else part))
    data_idx = np.loadtxt(idx_filename, dtype=np.int32)
    
    bones = []
    for i, (id_gesture, id_finger, id_subject, id_essai, label_14, label_28, size_sequence) in enumerate(tqdm(data_idx)):
        filename = os.path.join(data_path, 'gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt'.format(
            id_gesture, id_finger, id_subject, id_essai))
        data = np.loadtxt(filename, dtype=np.float64)
        data = np.reshape(data, (-1, 22, 3)) # (num_frame, num_joint, 3)
        for frame in range(data.shape[0]):   
            output = get_bone_data(data[frame])
            bones.append(output)

    bone_data = np.array(bones, dtype=np.float64)
    print(bone_data.shape)
    np.save('{}/{}_bone_data_no_adjust.npy'.format(out_path, part), bone_data)

#（num, frame, 3）-> i+1 frame root position - i frame root position     
def gen_root_dert(data_path, out_path, benchmark, part):
    idx_filename = os.path.join(data_path, "{}_gestures.txt".format('test' if part=='val' else part))
    data_idx = np.loadtxt(idx_filename, dtype=np.int32)
    root_dert = np.zeros((data_idx.shape[0], 180, 3), dtype=np.float64)
    # get data folder and label
    for i, (id_gesture, id_finger, id_subject, id_essai, label_14, label_28, size_sequence) in enumerate(tqdm(data_idx)):
        filename = os.path.join(data_path, 'gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt'.format(
            id_gesture, id_finger, id_subject, id_essai))
        data = np.loadtxt(filename, dtype=np.float64)
        data = np.reshape(data, (-1, 22, 3)) # (num_frame, num_joint, 3)
        for frame in range(data.shape[0] - 1):
            root_dert[i, frame, :] = data[frame+1, 0, :] - data[frame, 0, :]
        # break
    
    print(root_dert.shape)
    np.save('{}/{}_root_dert.npy'.format(out_path, part), root_dert)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHREC17 Data Converter.')
    parser.add_argument('--data_path', default='../SHREC2017/')
    parser.add_argument('--out_folder', default='./data/')

    benchmark = ['14', '28']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gen_root_dert(
                arg.data_path,
                out_path,
                benchmark=b,
                part=p)
