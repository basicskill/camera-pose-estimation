import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

def plotXYZ(xyz,folder_num,sampling = 1):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ground_truth = GetGroudnTruth(folder_num)
    # for i in range(1,len(xyz)):
    #     xyz[i] += xyz[i-1]
    ax.plot3D(ground_truth[:,0], ground_truth[:,1], ground_truth[:,2], 'blue')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(xyz[:,0], xyz[:,1], xyz[:,2], 'red')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    plt.show()
    # torch.save(xyz, 'plot/'+str(folder_num)+'plot'+ datetime.now().strftime("%m%d%Y%H%M%S")+'.pickle'+'.pt')
    #pickle.dump(fig, open('plot'+ datetime.now().strftime("%m%d%Y%H%M%S")+'.pickle', 'wb'))
    #plt.savefig('plot'+ datetime.now().strftime("%m%d%Y%H%M%S")+ '.png')

def GetGroudnTruth(i):
    positioning_path = 'D:/data_odometry_gray/dataset/poses/'
    positioning = open(positioning_path + str(i).zfill(2)+'.txt',"r")
    positioning_3x4 = positioning.readlines()
    positioning.close()
    positioning_temp = []
    transitions = []
    for pos in positioning_3x4:
        pos = pos.split()
        for j in range(len(pos)):
            #word = word[1:-1]
            pos[j] = float(pos[j])
        transitions.append([pos[3],pos[7],pos[11]])
        
    transitions = torch.Tensor(transitions)
    return transitions
"""
n = 10
for i in range(n):
    plotXYZ(GetGroudnTruth(i),1)
    """