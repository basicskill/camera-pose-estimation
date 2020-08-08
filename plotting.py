import torch
import numpy as np
import matplotlib.pyplot as plt


def plotXYZ(xyz):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax = plt.axes(projection='3d')
    print(xyz.shape)
    ax.plot3D(xyz[:,0], xyz[:,1], xyz[:,2], 'blue')
    xyz = torch.add(xyz, 5)
    ax.plot3D(xyz[:,0], xyz[:,1], xyz[:,2], 'red')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

def GetGroudnTruth(i):
    positioning_path = 'C:/Users/DELL/Documents/Python/PSI ML/dataset/poses/'
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

n = 10
for i in range(n):
    plotXYZ(GetGroudnTruth(i))