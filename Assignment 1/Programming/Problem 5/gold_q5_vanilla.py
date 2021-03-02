import numpy as np
import torch
import torchvision
import pickle

import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)
                                         

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2) # window_size=2, stride=2 
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1) 
        self.conv1_ = nn.Conv2d(32, 64, 3, stride=1, padding=1)         
        self.conv1__ = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_ = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(512*2*2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        
        self.drop_layer = nn.Dropout(0.2)

        
        # input: 3x32x32
        # after conv1: 32,3x3,p=1 -> 32x32x32
        # after conv1_: 64,3x3,p=1 -> 64x32x32
        # after conv1__: 64,3x3,p=1 -> 64x32x32
        # after pooling: 64x16x16
        # after conv2: 128,3x3,p=1 -> 128x16x16
        # after conv2_: 128,3x3,p=1 -> 128x16x16
        # after pooling: 128x8x8
        # after conv3: 256,3x3,p=1 -> 256x8x8
        # after pooling: 256x4x4
        # after conv4: 512,3x3,p=1 -> 512x4x4
        # after pooling: 512x2x2
        # review
        # fully connecteds...
    
    # Without dropout
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_(x))
        x = self.pool(F.relu(self.conv1__(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv2_(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu((self.conv4(x))))
        x = x.view(-1, 512*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
    # With dropout
#    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.relu(self.conv1_(x))
#        x = self.pool(F.relu(self.conv1__(x)))
#        x = self.drop_layer(x)
#        x = F.relu(self.conv2(x))
#        x = self.pool(F.relu(self.conv2_(x)))
#        x = self.drop_layer(x)
#        x = self.pool(F.relu(self.conv3(x)))
#        x = self.drop_layer(x)
#        x = self.pool(F.relu((self.conv4(x))))
#        x = self.drop_layer(x)
#        x = x.view(-1, 512*2*2)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x

# Comment it if it's being loaded
net = Net()

# with open('CNN_model_vanilla.pickle','rb') as f:
#    net = pickle.load(f)

net.to(device)
for name, param in net.named_parameters():
    if param.device.type != 'cuda':
        print('param {}, not on GPU'.format(name))

net.train(True)


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.99)


def eval(data_l):
    correct = 0
    total = 0
    with torch.no_grad():
        running_loss = 0.0
        for i, data in enumerate(data_l, 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loss_ = criterion(outputs, labels)
            running_loss += loss_.item()
    return [running_loss/i, 100 * correct / total]

train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}


for epoch in range(80):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
#         if i % verbose_len == verbose_len-1:    # print every 1000 mini-batches
#             print(f"epoch:{epoch + 1}, batch:{i + 1} loss:{running_loss / verbose_len}")
#             running_loss = 0.0

    net.eval(True)            
    train_loss, train_accuracy = eval(trainloader)
    valid_loss, valid_accuracy = eval(testloader)
    net.train(True)
    
    print(f"epoch:{epoch + 1} training_loss:{train_loss}, training_accuracy:{train_accuracy}")
    print(f"epoch:{epoch + 1} validation_loss:{valid_loss}, validation_accuracy:{valid_accuracy}")
    
    train_logs['train_accuracy'].append(train_accuracy)
    train_logs['validation_accuracy'].append(valid_accuracy)
    train_logs['train_loss'].append(train_loss)
    train_logs['validation_loss'].append(valid_loss)

print('Finished Training')


with open('logs_vanilla_80_epoch.pickle','wb') as f:
    pickle.dump(train_logs, f)
    

# To save the model if required
with open('CNN_model_vanilla_80_epoch.pickle','wb') as f:
    pickle.dump(net, f)


