import numpy as np
import torch
import torchvision
import pickle

from solution import load_cifar10
data = load_cifar10('/tmp/data', flatten=True)  # can use flatten=False to get the image shape.
train_data, valid_data, test_data = data
image, label = train_data[0][0], train_data[1][0]
image.shape

from solution import NN
import time
nn = NN(data=data)
start = time.time()
logs = nn.train_loop(20)
print(time.time()-start)
print(nn.evaluate())

with open('logs_glorot_tanh_4h.pickle','wb') as f:
    pickle.dump(logs,f)


