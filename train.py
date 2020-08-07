import torch
from model import MyNet
import time
import copy
from dataLoader import DataGetter
import torch.optim as optim

def train_model(model, optimizer, data_loader, num_epochs=25):
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    metrics = {'train_loss' : 0}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # for phase in ['train', 'val']:
            # if phase == 'train':
        phase = 'train'
        model.train()
        # else:
        #     model.eval()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        # TODO: Break data into train and val subsections
        for img_batch, quaternions, transitions in data_loader:
            print("NEW batch")

            img_batch = img_batch.to(device)
            quaternions = quaternions.to(device)
            transitions = transitions.to(device)

            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                # TODO: Request 2 image tensors from dataloader
                t_out, q_out = model(img_batch, img_batch)
                loss = model.loss(t_out, q_out, transitions, quaternions)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item()
        metrics



    time_elapsed = time.time() - start_time
    print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, metrics

if __name__ == "__main__":
    model = MyNet()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    # Data loader init
    data_dir = './dataset/'
    batch_size = 16
    data_loader = DataGetter(data_dir, batch_size, 0, 0)

    train_model(model, optimizer, data_loader, num_epochs=1)