import torch
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog as fd 

for i in range(10):
    name= fd.askopenfilename() 
    xyz = torch.load(name)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xyz[:,0], xyz[:,1], xyz[:,2], 'blue')
    xyz = torch.add(xyz, 5)
    ax.plot3D(xyz[:,0], xyz[:,1], xyz[:,2], 'red')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    fig.show()