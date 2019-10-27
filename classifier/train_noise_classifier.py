import numpy       as     np
from   PIL         import Image

import torch
import torch.nn    as     nn
from   torchvision import transforms

# load pre-trained ShuffleNetV2 ~8MB model
model = torch.hub.load('pytorch/vision', 'shufflenet_v2_x1_0', pretrained=True)

# can be vocal or noise
num_classes = 2

# replace fully-connected layer to match our dataset
# added 512 and dropout
model.fc    = nn.Sequential(nn.Linear(1024, 512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(512, num_classes))

# use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# transform input images to match network
trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# prepare dataset
dataset_dir = "/Users/gustavo/Documents/git/vocalpy/training/dataset/vocal_noise"
train_dir = os.path.join(dataset_dir, "train")

# -- create training dataset1p
train_dataset = datasets.ImageFolder(train_dir, transform=trans)
class_names   = train_dataset.classes
num_classes   = len(class_names)
train_size    = len(train_dataset)
train_indices = list(range(train_size))
np.random.shuffle(train_indices)

print('train dataset has {} images'.format(train_size))
print('train dataset has {} classes:'.format(num_classes))
print(class_names)

# -- create testing dataset
test_dir      = os.path.join(dataset_dir, "test")
test_dataset  = datasets.ImageFolder(test_dir, transform=trans)
test_size     = len(test_dataset)
test_indices  = list(range(test_size))
np.random.shuffle(test_indices)

class_names   = test_dataset.classes
num_classes   = len(class_names)
train_size    = len(test_dataset)
print('test dataset has {} images'.format(train_size))
print('test dataset has {} classes:'.format(num_classes))
print(class_names)

# -- create dataloaders
train_sampler = SubsetRandomSampler(train_indices)
test_sampler  = SubsetRandomSampler(test_indices)

batch_size  = 32
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler),
     'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, sampler=test_sampler)
}

