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
# nn = NN(data=data)
# start = time.time()
# logs = nn.train_loop(20)
# print(time.time()-start)
# print(nn.evaluate())

with open('model_relu_4h.pickle','rb') as f:
    nn = pickle.load(f)


# with open('logs_glorot_relu_4h.pickle','wb') as f:
#    pickle.dump(logs,f)

# with open('model_relu_4h.pickle','wb') as f:
#    pickle.dump(nn,f)


def loss_single_sample(nn,x,y,eps,k):
    dims = nn.weights[f"W{last_layer}"].shape
    row = np.int(np.floor(k/dims[1]))
    col = np.mod(k,dims[1])-1
    # print(f"row:{row}, col:{col}")
    nn.weights[f"W{last_layer}"][row,col] = nn.weights[f"W{last_layer}"][row,col] + eps
    # print(nn.weights['W4'][0,i])
    cache = nn.forward(x)
    nn.weights[f"W{last_layer}"][row,col] = nn.weights[f"W{last_layer}"][row,col] - eps
    pred = cache[f"Z{last_layer}"][0]
    # print(f"prediction={pred[y]}")
    return -np.log(pred[y])

def exact_grad(nn,cache,y,k):

    dims = nn.weights[f"W{last_layer}"].shape
    row = np.int(np.floor(k/dims[1]))
    col = np.mod(k,dims[1])-1

    output = cache[f"Z{last_layer}"]
    # print(f"cache={output[0][y]}")
    grads = {}

    # labels are already one-hotted
    grads[f"dA{last_layer}"] = -(y-output)
    grads[f"dW{last_layer}"] = (cache[f"Z{last_layer-1}"].T)@(grads[f"dA{last_layer}"])
    return grads[f"dW{last_layer}"][row,col]









last_layer = 5

N = np.reshape(np.array([10**j for j in range(1,6)]),(1,5))
k = np.reshape(np.arange(1,6),(5,1))
N = N*k
p = 100
# Ns
num_of_Ns = 15
rows = np.random.choice([0,1,2,3,4],num_of_Ns)
cols = np.random.choice([0,1,2,3,4],num_of_Ns)

Ns = [N[rows[i],cols[i]] for i in range(len(rows))]
Ns = np.sort(Ns)

sample_id = np.random.randint(0,train_data[0].shape[0])
x_ = train_data[0][sample_id]
y_onehot = train_data[1][sample_id]
y_ = np.argmax(y_onehot)

max_diff = []

for n in Ns:
    epsilon = 1/n
    nabla_N = []
    exact_grads = []
    for k in range(1,p+1):
        grad_approx = (loss_single_sample(nn,x_,y_,epsilon,k)-loss_single_sample(nn,x_,y_,-epsilon,k))/(2*epsilon)
        nabla_N.append(grad_approx)

        grad_exact = exact_grad(nn,nn.forward(x_),y_onehot,k)
        exact_grads.append(grad_exact)

    difference = np.max(np.abs(np.array(nabla_N)-np.array(exact_grads)))
    # print(difference)
    max_diff.append(difference)




out_dict = {'Ns':Ns,'diff':max_diff}
with open('max_diffs.pickle','wb') as f:
    pickle.dump(out_dict,f)



