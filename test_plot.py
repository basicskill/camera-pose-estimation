import torch
from plotting import plotXYZ
from dataLoader import DataGetter, Euler2Rot, euler_angles_from_rotation_matrix
from copy import deepcopy
import numpy as np

# data_dir = 'D:\data_odometry_gray\dataset'
data_dir = 'D:\data_odometry_color\dataset'
# data_dir = 'C:/Users/DELL/Documents/Python/PSI ML/dataset'
batch_size = 1
folder_num = 6

data = DataGetter(data_dir, batch_size, folder_num, folder_num, sampling=1, randomize_data=False) 

running_R = torch.eye(3, dtype=torch.float)
running_t = torch.zeros((1, 3), dtype=torch.float)
positions = torch.tensor([[0, 0, 0]], dtype=torch.float)


for (_, _, Ojler, t), R in zip(data, data.pos_dataset.rot_matrix):
    
    # O = torch.tensor(euler_angles_from_rotation_matrix(R))
    O = Ojler.numpy().flatten()
    Rot = torch.tensor(Euler2Rot(O), dtype=torch.float)
    # R = torch.tensor(R, dtype=torch.float)
    t = t.reshape((3,1))
    running_t += (torch.transpose(running_R, 0, 1) @ t).reshape((1, 3))
    running_R = Rot @ running_R
    positions = torch.cat((positions, running_t), dim=0)

print(positions.shape)
plotXYZ(positions[1:], folder_num)