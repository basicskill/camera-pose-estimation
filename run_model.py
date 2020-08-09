import torch
import torch.optim as optim
from model import MyNet
import time
import copy
from dataLoader import DataGetter
import matplotlib.pyplot as plt
import pickle

from train import train_model

if __name__ == "__main__":

    model = MyNet()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    # Data loader init
    data_dir = 'D:\data_odometry_gray\dataset'
    batch_size = 8

    trainData = DataGetter(data_dir, batch_size, 1, 6, sampling=5)
    valData = DataGetter(data_dir, batch_size, 8, 10, sampling=4)

    model, metrics = train_model(model, optimizer, trainData, valData, num_epochs=10)

    # Save model and results
    name = time.ctime(time.time()).replace(' ', '_').replace(':', '_')
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    model.eval()
    torch.save(model.state_dict(), 'model_' + name + '.pkl')