import torch
from model import MyNet
from dataLoader import DataGetter, Euler2Rot
import numpy as np
from plotting import plotXYZ
from tkinter import filedialog as fd 

# data_dir = 'D:/data_odometry_gray/dataset'
data_dir = 'D:/data_odometry_color/dataset'
folder_num = 6
batch_size = 1

model_name = fd.askopenfilename()
model = MyNet()
model.load_state_dict(torch.load(model_name))

getter = DataGetter(data_dir, batch_size, folder_num, folder_num, randomize_data=False)

if torch.cuda.is_available():
    device = 'cuda'
    model.cuda()
else:
    device = 'cpu'
model.eval()


running_R = torch.eye(3, dtype=torch.float)
running_t = torch.zeros((1, 3), dtype=torch.float)
positions = torch.tensor([[0, 0, 0]], dtype=torch.float)


for img_batch1, img_batch2, _, _ in getter:
    img_batch1 = img_batch1.to(device)
    img_batch2 = img_batch2.to(device)

    # TODO: swap outputs
    t, Ojler = model(img_batch1, img_batch2)
    t = t.cpu()
    t = t.detach()
    Ojler = Ojler.cpu()
    Ojler = Ojler.detach()

    # Reshaping
    O = Ojler.numpy().flatten()
    Rot = torch.tensor(Euler2Rot(O), dtype=torch.float)
    t = t.reshape((3,1))

    running_t += (torch.transpose(running_R, 0, 1) @ t).reshape((1, 3))
    running_R = Rot @ running_R
    positions = torch.cat((positions, running_t), dim=0)

print(positions.shape)
plotXYZ(positions[1:], folder_num)

