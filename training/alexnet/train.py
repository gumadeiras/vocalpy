#import os
import copy
import time
#import logging
import numpy             as     np
<<<<<<< HEAD:training/train.py
#import matplotlib.pyplot as     plt
=======
import matplotlib.pyplot as     plt
from   tqdm              import tqdm
>>>>>>> master:training/alexnet/train.py

import torch
import torch.nn                 as     nn
import torch.optim              as     optim
from   torch.optim              import lr_scheduler as lrs
from   torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

#import torchvision
from   torchvision       import datasets, models, transforms

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step(1-best_acc)
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
<<<<<<< HEAD:training/train.py
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size#[phase]
            epoch_acc = running_corrects.double() / dataset_size#[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'resnet50_trained.pth')
=======
            with tqdm(total=len(dataloaders[phase].dataset)*0.9/128) as t:
                for itr, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train' or (itr+1)%4 ==0):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train' or (itr+1)%4 ==0:
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    running_avg_loss = running_loss/((itr+1)*128)
                    t.set_postfix(loss='{:05.3f}'.format(running_avg_loss))
                    t.update()

                if phase == 'train':
                    epoch_loss = running_loss / (len(dataloaders[phase].dataset)*0.9)
                    epoch_acc  = running_corrects.double() / (len(dataloaders[phase].dataset)*0.9)
                else:
                    epoch_loss = running_loss / (len(dataloaders[phase].dataset)*0.1)
                    epoch_acc  = running_corrects.double() / (len(dataloaders[phase].dataset)*0.1)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'alexnet_trained.pth')
>>>>>>> master:training/alexnet/train.py

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # -- training parameters
<<<<<<< HEAD:training/train.py
    plt.ion()
    dataset_dir      = 'Z:\\Dietrich_Server\\Gustavo\\dataset\\18-november-dirty-augment'
    batch_size       = 32
=======
    dataset_dir      = '../dataset/18-november-dirty-augment'
    batch_size       = 128
>>>>>>> master:training/alexnet/train.py
    validation_split = .2 # -- split training set into train/val sets
    epochs           = 100
    print('training params:\n dataset dir: {}\n batch size: {}\n val split: {}\n epochs: {}'.format(dataset_dir, batch_size, validation_split, epochs))

    # -- transforms to use
    trans         = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    # -- create dataset
    image_dataset = datasets.ImageFolder(dataset_dir, transform=trans)
    class_names   = image_dataset.classes
    num_classes   = len(class_names)
    dataset_size  = len(image_dataset)
    print('dataset has {} images'.format(dataset_size))
    print('dataset has {} classes:'.format(num_classes))
    print(class_names)

    # -- split dataset
    indices       = list(range(dataset_size))
    split         = int(np.floor(validation_split*dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # -- create dataloaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    dataloaders   = {
        'train': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler),
          'val': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, num_workers=4, sampler=valid_sampler)
    }

    # -- load pre-trained resnet50
    model    = models.alexnet(pretrained=True)

    # -- replace FC layer to match number of classes
    # model.fc = nn.Sequential(nn.Linear(2048, 512),
    #                          nn.ReLU(inplace=True),
    #                          nn.Dropout(0.2),
    #                          nn.Linear(512, num_classes),
    #                          nn.LogSoftmax(dim=1))
    # criterion        = nn.NLLLoss()
    # model.fc  = nn.Sequential(nn.Linear(4096, 512),
    #                           nn.ReLU(inplace=True),
    #                           nn.Dropout(0.2),
    #                           nn.Linear(512, num_classes))
    model.classifier[6] = nn.Linear(4096, num_classes)
    criterion = nn.CrossEntropyLoss()
#    print('model: {}'.format(model))

    # -- optimizer
    optimizer        = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler        = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=5)
    device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=epochs)

    # -- train the model
    # running_loss             = 0
    # print_every              = 1
    # train_losses, val_losses = [], []
    # for epoch in range(epochs):
    #     for itr (inputs, labels) in enumerate(dataloaders['train']):
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         logps = model.forward(inputs)
    #         loss = criterion(logps, labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
            
    #         if itr % print_every == 0:
    #             val_loss = 0
    #             accuracy = 0
    #             model.eval()
    #             with torch.no_grad():
    #                 for inputs, labels in dataloaders['val']:
    #                     inputs, labels = inputs.to(device), labels.to(device)
    #                     logps          = model.forward(inputs)
    #                     batch_loss     = criterion(logps, labels)
    #                     val_loss      += batch_loss.item()
                        
    #                     ps               = torch.exp(logps)
    #                     top_p, top_class = ps.topk(1, dim=1)
    #                     equals           = top_class == labels.view(*top_class.shape)
    #                     accuracy        += torch.mean(equals.type(torch.FloatTensor)).item()
                
    #             train_losses.append(running_loss/len(dataloaders['train']))
    #             val_losses.append(val_loss/len(dataloaders['val']))                    
                
    #             print(f"Epoch {epoch+1}/{epochs}.. "
    #                   f"Train loss: {running_loss/print_every:.3f}.. "
    #                   f"Test loss: {val_loss/len(dataloaders['val']):.3f}.. "
    #                   f"Test accuracy: {accuracy/len(dataloaders['val']):.3f}")
    #             running_loss = 0
    #             model.train()
    # torch.save(model, 'resnet50_trained.pth')






    # for i in range(epochs):
    #     for phase in ['train', 'val']:
    #         if phase == 'train':
    #             model.train()
    #         else:
    #             model.eval()
            
    #         samples     = 0
    #         loss_sum    = 0
    #         correct_sum = 0

    #         for j, batch in enumerate(dataloaders[phase]):
    #             X      = batch['image']
    #             labels = batch['label']
    #             if device == 'cuda':
    #                 X      = X.cuda()
    #                 labels = labels.cuda()

    #             optimizer.zero_grad()

    #             with torch.set_grad_enabled(phase == 'train'):
    #                 y = model(X)
    #                 loss = criterion(y, labels.view(-1, 1).float())

    #                 if phase == 'train':
    #                     loss.backward()
    #                     optimizer.step()
                        
    #                 loss_sum     += loss.item() * X.shape[0] # We need to multiple by batch size as loss is the mean loss of the samples in the batch
    #                 samples      += X.shape[0]
    #                 num_corrects  = torch.sum((y >= 0.5).float() == labels.view(-1, 1).float())
    #                 correct_sum  += num_corrects
                    
    #                 # Print batch statistics every 50 batches
    #                 if j % 50 == 49 and phase == 'train':
    #                     print('{}:{} - loss: {}, acc: {}'.format(
    #                         i + 1, 
    #                         j + 1, 
    #                         float(loss_sum) / float(samples), 
    #                         float(correct_sum) / float(samples)
    #                     ))
                    
    #         # Print epoch statistics
    #         epoch_acc = float(correct_sum) / float(samples)
    #         epoch_loss = float(loss_sum) / float(samples)
    #         print('epoch: {} - {} loss: {}, {} acc: {}'.format(i + 1, phase, epoch_loss, phase, epoch_acc))
            
    #         # Deep copy the model
    #         if phase == 'val' and epoch_acc > best_acc:
    #             best_acc = epoch_acc
    #             best_model_wts = copy.deepcopy(model.state_dict())
    #             torch.save(best_model_wts, 'resnet50.pth')