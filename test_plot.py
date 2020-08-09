import torch
from plotting import plotXYZ
from dataLoader import DataGetter, Euler2Rot

data_dir = 'D:\data_odometry_gray\dataset'
batch_size = 1

data = DataGetter(data_dir, batch_size, 1, 1, sampling=1, randomize_data=False)

running_R = torch.eye(3)
running_t = torch.zeros((1, 3), dtype=torch.float)
positions = torch.tensor([[0, 0, 0]], dtype=torch.float)

for _, _, Ojler, t in data:
    R = Euler2Rot(Ojler)
    t = t.reshape((3,1))
    running_R = R @ running_R
    running_t += (running_R @ t).reshape((1, 3))
    positions = torch.cat((positions, running_t), dim=0)

print(positions.shape)
plotXYZ(positions[1:], 1)