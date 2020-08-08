import torch
from model import MyNet
from dataLoader import DataGetter
import numpy as np
from plotting import plotXYZ

model_name = './models/1596840425.779855.model'
data_dir = 'D:/data_odometry_gray/dataset'
folder_num = 0

getter = DataGetter(data_dir, 4, folder_num, folder_num)
model = MyNet()
model.load_state_dict(torch.load(model_name))

if torch.cuda.is_available():
    device = 'cuda'
    model.cuda()
else:
    device = 'cpu'

model.eval()

# ground_truth = torch.tensor([]).to('cpu')
# solution = torch.tensor([]).to('cpu')
gt = np.array([[0, 0, 0]])
sol = np.array([[0, 0, 0]])

for img_batch1, img_batch2, _, transitions in getter:
    img_batch1 = img_batch1.to(device)
    img_batch2 = img_batch2.to(device)
    transitions = transitions.to(device)

    #t_out, _ = model(img_batch1, img_batch2) 
    # print(t_out.shape)
    #t_out = t_out.cpu()
    transitions = transitions.cpu()
    #sol = np.concatenate([sol, t_out.detach().numpy()], axis=0)
    gt = np.concatenate([gt, transitions.detach().numpy()], axis=0)

sol = torch.tensor(sol[1:])
gt = torch.tensor(gt[1:])

gt = torch.add(gt, 5)
plotXYZ(gt, folder_num)
plotXYZ(sol, folder_num)

