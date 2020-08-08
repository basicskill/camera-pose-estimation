import torch
from model import MyNet
from dataLoader import DataGetter

model_name = './models/1596840425.779855.model'
data_dir = './dataset/'
folder_num = 4

getter = DataGetter(data_dir, 4, folder_num, folder_num)
model = torch.load(model_name)
if torch.cuda.is_available():
    device = 'cuda'
    model.cuda()
else:
    device = 'cpu'

model.eval()

ground_truth = torch.tensor([])
solution = torch.tensor([])

for img_batch1, img_batch2, _, transitions in getter:
    img_batch1 = img_batch1.to(device)
    img_batch2 = img_batch2.to(device)
    transitions = transitions.to(device)

    t_out, _ = model(img_batch1, img_batch2) 

    solution = torch.cat([solution, t_out], dim=0)
    ground_truth = torch.cat([ground_truth, transitions], dim=0)

print(solution.shape)
print(ground_truth.shape)

