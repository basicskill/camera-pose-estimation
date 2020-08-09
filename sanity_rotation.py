import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from dataLoader import DataGetter

main_path = 'C:/Users/DELL/Documents/Python/PSI ML/dataset/'
data = DataGetter(main_path,1,0,0,1)
pos_dataset = data.pos_dataset