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

    # writer = SummaryWriter()

    start_time = time.time()

    # Make folder for model saving
    if not path.exists('runs/'):
        os.mkdir('runs')
    if name == 'model_':
        name += time.ctime(start_time).replace(' ', '').replace(':', '_')
    
    name = 'runs/' + name + '/'
    os.mkdir(name)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_error = 0.0

    if torch.cuda.is_available():
        model.cuda()
        device = 'cuda'
        model.cuda()
    else:
        device = 'cpu'
    
    metrics = {
        'train_loss'    : [],
        'train_error'     : [],
        'val_loss'      : [],
        'val_error'       : [],
    }

    for epoch in range(num_epochs):
        
        print('-' * 10)
        print(f'Epoch {epoch+1}/{num_epochs}')
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
            running_error = 0.0
            epoch_size = 0.0

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
                    _loss = loss.item()
                    _error = ATEpos(transitions, t_out)
            
                    running_loss += _loss
                    running_error += _error

                    metrics[phase + '_loss'].append(_loss)
                    metrics[phase + '_error'].append(_error)

            epoch_loss = running_loss
            epoch_error = running_error

            print(f'Epoch size: {epoch_size}')

            print(f'{phase} Loss: {epoch_loss:.4f} Error: {epoch_error:.4f}')
            
            # writer.add_scalar(phase + ' Train', epoch_loss, epoch)
            # writer.add_scalar(phase + ' Train', epoch_loss, epoch)
            
            # deep copy the model
            if phase == 'val' and epoch_error > best_error:
                best_error = epoch_error
                best_model_wts = copy.deepcopy(model.state_dict())

        with open(name + '/epoch' + str(epoch) + '.pickle', 'wb') as f:
            pickle.dump(metrics, f)

        torch.save(model.state_dict(), name + '/epoch' + str(epoch) + '.model')


    time_elapsed = time.time() - start_time
    print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Error: {best_error:4f}')

    model.load_state_dict(best_model_wts)

    return model, metrics
