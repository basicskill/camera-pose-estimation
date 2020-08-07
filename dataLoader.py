from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np


class CustomDataSet(Dataset):
    def __init__(self, main_dir, curr_index, batch_size):
        self.main_dir = main_dir
        self.image_dir = '/sequences/'
        self.curr_index = curr_index
        self.batch_size = batch_size
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs
        all_imgs = os.listdir(self.main_dir+self.image_dir+str(self.curr_index).zfill(2)+'/image_0/')
        self.total_imgs = all_imgs
        
    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc1 = os.path.join(self.main_dir+self.image_dir+str(self.curr_index).zfill(2)+'/image_0/', self.total_imgs[idx])
        img_loc2 = os.path.join(self.main_dir+self.image_dir+str(self.curr_index).zfill(2)+'/image_0/', self.total_imgs[idx+1])
        image1 = Image.open(img_loc1)
        image2 = Image.open(img_loc2)
        tensor_image1 = transforms.ToTensor()(image1)
        tensor_image2 = transforms.ToTensor()(image2)
        if tensor_image1.shape[0] == 1:
            tensor_image1 = torch.cat([tensor_image1, tensor_image1, tensor_image1], dim=0)
            tensor_image2 = torch.cat([tensor_image2, tensor_image2, tensor_image2], dim=0)
        return [tensor_image1, tensor_image2]

class PositioningDataset():
    def __init__(self, main_dir,curr_index, batch_size):
        self.main_dir = main_dir
        self.pos_dir = '/poses/'
        self.curr_index = curr_index
        self.batch_size = batch_size
        self.positioning = []
        self.make_dataset()

    def __len__(self):
        return len(self.positioning)

    def __getitem__(self, idx):
        #position = self.positioning[idx]
        return self.positioning[idx]
        
    def make_dataset(self):
        #data part
        positioning_path = self.main_dir + self.pos_dir
        positioning = open(positioning_path + str(self.curr_index).zfill(2)+'.txt',"r")
        positioning_3x4 = positioning.readlines()
        positioning.close()
        self.positioning = []
        positioning_temp = []
        self.transitions = []
        self.quaternions = []
        for pos in positioning_3x4:
            pos = pos.split()
            for j in range(len(pos)):
                #word = word[1:-1]
                pos[j] = float(pos[j])
            self.transitions+=[[pos[3],pos[7],pos[11]]]
            #quaternions
            """
            qw = np.sqrt(1+pos[0]+pos[5]+pos[10])/2 # matrix diagonal
            qx = pos[9] - pos[6] / (4*qw)
            qy = pos[2] - pos[8] / (4*qw)
            qz = pos[4] - pos[1] / (4*qw)
            self.positioning += [[qw,qx,qy,qz,pos[3],pos[7],pos[11]]]
            """
            self.positioning += [pos]
            positioning_temp += [[[pos[0],pos[1],pos[2]],
                                [pos[4],pos[5],pos[6]],
                                [pos[8],pos[9],pos[10]]]]
        for i in range(len(positioning_temp)-1):
            positioning_temp[i] = positioning_temp[i+1]@ np.transpose(positioning_temp[i])
        positioning_temp = np.array(positioning_temp).reshape(len(positioning_temp), 9)
        #print(positioning_temp)
        for pos in positioning_temp:
            qw = np.sqrt(1+pos[0]+pos[4]+pos[8])/2 # matrix diagonal
            qx = pos[7] - pos[5] / (4*qw)
            qy = pos[2] - pos[6] / (4*qw)
            qz = pos[3] - pos[1] / (4*qw)
            self.quaternions+=[[qw,qx,qy,qz]]

        self.transitions = np.concatenate((np.diff(self.transitions, axis = 0),[[0,0,0]]), axis=0)
        
        self.positioning = np.concatenate((self.quaternions, self.transitions), axis=1)

        self.positioning = torch.Tensor(self.positioning)
            
        """qw= √(1 + m00 + m11 + m22) /2
        qx = (m21 - m12)/( 4 *qw)
        qy = (m02 - m20)/( 4 *qw)
        qz = (m10 - m01)/( 4 *qw)"""

class DataGetter():
    def __init__(self, main_dir, batch_size, start_index, end_index):
        self.main_dir = main_dir
        self.curr_index = start_index - 1
        self.end_index = end_index
        self.index = 0
        self.pos_dir = '/poses/'
        self.batch_size = batch_size
        self.image_dataset = None
        self.train_loader = None
        self.train_loader_iterator1 = None
        self.pos_dataset = None
        self.pos_loader = None
        self.pos_loader_iterator = None
        self.make_datasets()

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        img_batches = None
        try:
            img_batches = next(self.train_loader_iterator1)
        except:
            if self.curr_index == self.end_index:
                raise StopIteration
            self.make_datasets()
            img_batches = next(self.train_loader_iterator1)

        quaternion_batch=0
        transitions_batch=0

        all_data = next(self.pos_loader_iterator)

        quaternion_batch = all_data[:,:4]
        transitions_batch = all_data[:,4:]

        return img_batches[0],img_batches[1], quaternion_batch, transitions_batch

    def make_datasets(self):
        self.curr_index += 1
        self.image_dataset = CustomDataSet(self.main_dir, self.curr_index, self.batch_size)
        self.train_loader = DataLoader(self.image_dataset , batch_size=self.batch_size, shuffle=False)
        self.train_loader_iterator1 = iter(self.train_loader)
        self.pos_dataset = PositioningDataset(self.main_dir, self.curr_index, self.batch_size)
        self.pos_loader = DataLoader(self.pos_dataset , batch_size=self.batch_size, shuffle=False)
        self.pos_loader_iterator = iter(self.pos_loader)

    def __iter__(self):
        return self
    
    def __next__(self):
        self.index +=1
        return self[self.index]

"""
#image part
img_folder_path = 'D:/data_odometry_gray/dataset/sequences/00/image_0'
batch_size = 64
my_dataset = CustomDataSet(img_folder_path, transform =False)
train_loader = DataLoader(my_dataset , batch_size=batch_size, shuffle=False)
train_loader_iterator = iter(train_loader)
#end image part

#data part
positioning_path = 'D:/data_odometry_gray/dataset/poses/'
positioning = open(positioning_path+'00.txt',"r")
positioning_3x4 = positioning.readlines()
positioning_final = [[]]
transition = []
quaternion = []
for pos in positioning_3x4:
    pos = pos.split()
    for j in range(len(pos)):
        #word = word[1:-1]
        pos[j] = float(pos[j])
    #quaternions
    qw = np.sqrt(1+pos[0]+pos[5]+pos[10])/2 # matrix diagonal
    qx = pos[9] - pos[6] / (4*qw)
    qy = pos[2] - pos[8] / (4*qw)
    qz = pos[4] - pos[1] / (4*qw)
    quaternion += [[qw,qx,qy,qz]]
    transition += [[pos[3], pos[7], pos[11]]]
    qw= √(1 + m00 + m11 + m22) /2
qx = (m21 - m12)/( 4 *qw)
qy = (m02 - m20)/( 4 *qw)
qz = (m10 - m01)/( 4 *qw)
    
    positioning_final.append(pos)
#print([words for segments in positioning_3x4 for words in segments.split()])
#positioning_3x4 = np.reshape(positioning_3x4, len(positioning_3x4)/12, 12)
print(positioning_final)
print('+++++++++++++++++++++++++++++++++++')
print(quaternion)
print('+++++++++++++++++++++++++++++++++++')
print(transition)


not_done = True
i = 0
while not_done:
    batch = next(train_loader_iterator)
    if(len(batch)<64):
        break
        not_done = False
    dataset1 = batch.narrow(0,0,batch_size-2)
    dataset2 = batch.narrow(0,1,batch_size-1)
    print(len(batch))
    print(i)
    i+=1
    #print(batch)"""

### Primer kako radi

if __name__ == "__main__":
    main_dir = 'D:\\data_odometry_gray\\dataset'
    batch_size = 32
    all_data = DataGetter(main_dir, batch_size, 0, 0)
    i = 0
    for img_batch1, img_batch2, quaternions, transitions in all_data:
        print(str(len(img_batch1)) + str(len(quaternions)+ len(transitions)))
        print(i)
        print(img_batch1[0,0,0,0])
        print(img_batch2[0,0,0,0])
        i+=1

