import torch
import torch.optim as optim
from model import MyNet
import time
import copy
from dataLoader import DataGetter
import matplotlib.pyplot as plt
import pickle
import os
import os.path as path
# from torch.utils.tensorboard import SummaryWriter

def ATEpos(arr_truth, arr_estim):
    return torch.sqrt(3 * torch.mean(((arr_truth - arr_estim) ** 2)))


def train_model(model, optimizer, trainGetter, valGetter, num_epochs=25, name='model_'):

    start_time = time.time()

    # Make folder for model saving
    if not path.exists('runs/'):
        os.mkdir('runs')
    if name == 'model_':
        name += str(start_time) 
    
    name = 'runs/' + name + '/'
    os.mkdir(name)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if torch.cuda.is_available():
        model.cuda()
        device = 'cuda'
        model.cuda()
    else:
        device = 'cpu'
    
    metrics = {
        'train_loss'    : [],
        'train_acc'     : [],
        'val_loss'      : [],
        'val_acc'       : [],
    }
    # writer = SummaryWriter()

    for epoch in range(num_epochs):
        
        print('-' * 10)
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        trainGetter.refresh()
        valGetter.refresh()

        for phase in ['train', 'val']:
            print(phase + " in progress...")
            if phase == 'train':
                phase = 'train'
                model.train()
                data_loader = trainGetter

            else:
                model.eval()
                data_loader = valGetter

            running_loss = 0.0
            running_corrects = 0
            epoch_size = 0

            # Iterate over data.
            for img_batch1, img_batch2, YPR, transitions in data_loader:
                img_batch1 = img_batch1.to(device)
                img_batch2 = img_batch2.to(device)
                YPR = YPR.to(device)
                transitions = transitions.to(device)
                epoch_size += img_batch1.size(0)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    t_out, q_out = model(img_batch1, img_batch2)
                    loss = model.loss(t_out, q_out, transitions, YPR)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                    # statistics
                    running_loss = loss.item()  #  * img_batch1.size(0)
                    running_corrects += ATEpos(transitions, t_out)
            
                    metrics[phase + '_loss'].append(running_loss)
                    metrics[phase + '_acc'].append(running_corrects)
                    
                # print(f'{time.time() - bacth_start} s epoha')
                bacth_start = time.time()

            epoch_loss = running_loss / epoch_size
            epoch_acc = running_corrects / epoch_size

            print(f'Epoch size: {epoch_size}')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            

            # writer.add_scalar(phase + ' Train', epoch_loss, epoch)
            # writer.add_scalar(phase + ' Train', epoch_loss, epoch)
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        with open(name + '/epoch' + str(epoch) + '.pickle', 'wb') as f:
            pickle.dump(metrics, f)

        torch.save(model.state_dict(), name + '/epoch' + str(epoch) + '.model')



    time_elapsed = time.time() - start_time
    print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)

    return model, metrics
