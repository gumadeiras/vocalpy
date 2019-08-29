import os
import copy
import time
import logging
import numpy             as     np
import matplotlib.pyplot as     plt
from   tqdm              import tqdm

import torch
import torch.nn                 as     nn
import torch.optim              as     optim
from   torch.optim              import lr_scheduler as lrs
from   torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from   torchvision       import datasets, models, transforms

def eval_model(model, dataloaders):
    since = time.time()

    model.eval()

    running_corrects = 0

    preds = []
    targets = []
    confusion_matrix = np.zeros((12, 12))

    phase = 'test'
    # Iterate over data.
    with tqdm(total=(len(dataloaders[phase].dataset)/128)+1) as tq:
        for itr, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                preds.append(pred.data.cpu().numpy())
                targets.append(labels.data.cpu().numpy())
                for t, p in zip(labels.view(-1), pred.view(-1)):
                    confusion_matrix[t.long().data.cpu().numpy(), p.long().data.cpu().numpy()] += 1

            # statistics
            running_corrects += torch.sum(pred == labels.data)

            running_avg_correct = running_corrects/((itr+1)*128)
            tq.set_postfix(loss='{:05.3f}'.format(running_avg_correct))
            tq.update()

        epoch_acc  = running_corrects.double() / len(dataloaders[phase].dataset)


        print('Acc: {:.4f}'.format(epoch_acc))

    return preds, targets, confusion_matrix

if __name__ == '__main__':
    # -- training parameters
    dataset_dir      = '../dataset/18-november-dirty-augment'
    testset_dir      = '../dataset/testset'
    batch_size       = 128
    validation_split = .1 # -- split training set into train/val sets
    epochs           = 100
    print('training params:\n dataset dir: {}\n batch size: {}\n val split: {}\n epochs: {}'.format(dataset_dir, batch_size, validation_split, epochs))

    # -- transforms to use
    trans         = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    # -- create dataset
    image_dataset = datasets.ImageFolder(dataset_dir, transform=trans)
    test_dataset  = datasets.ImageFolder(testset_dir, transform=trans)
    class_names   = test_dataset.classes
    num_classes   = len(class_names)
    dataset_size  = len(test_dataset)
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
          'val': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, num_workers=4, sampler=valid_sampler),
         'test': torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=4, shuffle=False),
    }

    # -- load pre-trained resnet50
    model  = models.resnet50(pretrained=True)
    model.fc  = nn.Sequential(nn.Linear(2048, 2048),
                          nn.ReLU(inplace=True),
                          nn.Dropout(0.2),
                          nn.Linear(2048, num_classes))
    model.load_state_dict(torch.load('resnet50_trained.pth'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    preds, targets, confusion_matrix = eval_model(model, dataloaders)
    np.save('/home/user/git/vocalpy/training/resnet50/preds', preds)
    np.save('/home/user/git/vocalpy/training/resnet50/targets', targets)
    np.save('/home/user/git/vocalpy/training/resnet50/confusion', confusion_matrix)