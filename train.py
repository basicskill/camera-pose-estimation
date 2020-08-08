import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from model import MyNet
import time
import copy
from dataLoader import DataGetter
import matplotlib.pyplot as plt
import pickle




def ATEpos(arr_truth, arr_estim):
    # result = 0
    # for i in range(arr_estim):
    #     for j in range(3):
    #         result += np.power((arr_estim[i][j]-arr_truth[i][j]),2)
    # result = np.sqrt(result/len(arr_estim))
    
    return torch.sqrt(3 * torch.mean(((arr_truth - arr_estim) ** 2)))


def train_model(model, optimizer, data_dir, num_epochs=25):
    start_time = time.time()

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
    batch_size = 4
    # writer = SummaryWriter()

    for epoch in range(num_epochs):
        
        print('-' * 10)
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            print(phase + " in progress...")
            if phase == 'train':
                phase = 'train'
                model.train()
                data_loader = DataGetter(data_dir, batch_size, 0, 5, sampling=2)

            else:
                model.eval()
                data_loader = DataGetter(data_dir, batch_size, 6, 6)

            running_loss = 0.0
            running_corrects = 0
            epoch_size = 0

            # Iterate over data.
            # TODO: Break data into train and val subsections
            for img_batch1, img_batch2, quaternions, transitions in data_loader:
                img_batch1 = img_batch1.to(device)
                img_batch2 = img_batch2.to(device)
                quaternions = quaternions.to(device)
                transitions = transitions.to(device)

                epoch_size += img_batch1.size(0)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    t_out, q_out = model(img_batch1, img_batch2)
                    loss = model.loss(t_out, q_out, transitions, quaternions)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                    # statistics
                    running_loss += loss.item() * img_batch1.size(0)
                    running_corrects += ATEpos(transitions, t_out)

            epoch_loss = running_loss / epoch_size
            epoch_acc = running_corrects / epoch_size

            print(f'Epoch size: {epoch_size}')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            metrics[phase + '_loss'].append(epoch_loss)
            metrics[phase + '_acc'].append(epoch_acc)

            # writer.add_scalar(phase + ' Train', epoch_loss, epoch)
            # writer.add_scalar(phase + ' Train', epoch_loss, epoch)
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())



    time_elapsed = time.time() - start_time
    print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    plt.plot(metrics['train_loss'])
    plt.plot(metrics['train_acc'])

    plt.plot(metrics['val_loss'])
    plt.plot(metrics['train_acc'])
    plt.show()

    model.load_state_dict(best_model_wts)

    return model, metrics

if __name__ == "__main__":

    model = MyNet()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    # Data loader init
    data_dir = './dataset/'

    model, metrics = train_model(model, optimizer, data_dir, num_epochs=10)

    # Save model and results
    name = str(time.time())
    pickle.dump(metrics, name + '.pickle')
    torch.save(model.state_dict(), name + '.model')