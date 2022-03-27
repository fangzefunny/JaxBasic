# basic 
import os
import time 
import numpy as np 
from sklearn.model_selection import train_test_split

# jax pkg 
import jax.numpy as jnp 
from jax import grad, jit, vmap 
from jax import random 
from jax.scipy.special import logsumexp

# for MNIST download 
import torch
from torchvision import datasets, transforms

# fro visualize
import matplotlib.pyplot as plt 
import seaborn as sns 

#--------------------------------
#        System variables
#--------------------------------

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{path}/figures'):
    os.mkdir(f'{path}/figures')

# define some color 
Blue    = .85 * np.array([   9, 132, 227]) / 255
Green   = .85 * np.array([   0, 184, 148]) / 255
Red     = .85 * np.array([ 255, 118, 117]) / 255
Yellow  = .85 * np.array([ 253, 203, 110]) / 255
Purple  = .85 * np.array([ 108,  92, 231]) / 255
colors  = [ Blue, Red, Green, Yellow, Purple]
sfz, mfz, lfz = 11, 13, 15
dpi     = 250
sns.set_style("whitegrid", {'axes.grid' : False})

#---------------------------------
#        MNIST Dataloader 
#---------------------------------

def get_MNIST():
    mnist_data = datasets.MNIST(f'{path}/data', train=True, download=True,
                                transform=transforms.Compose(
                                    [ transforms.ToTensor(), transforms.Normalize( (.1307,), (.3081,))]
                                ))
    data = (mnist_data.data.type( torch.FloatTensor) / 255).bernoulli()
    label = (mnist_data.targets.type( torch.FloatTensor) / 255).bernoulli()
    
    return train_test_split(data.numpy(), label.numpy(), test_size=.2)

#---------------------------------
#           MLP model 
#---------------------------------

ReLU = lambda x: jnp.maximum( 0, x)

softmax = lambda x: x - logsumexp(x)

def Linear( inDim, outDim, key, scale=1e-2):
    '''
    inDim: integer
        The input dim 
    outDim: integer
        The output dim 
    key: PRNGKey
        A Jax PRNGKey
    scale: the amplitude
    '''
    wkey, bkey = random.split(key, num=2)
    w = scale * random.normal( wkey, [ inDim, outDim])
    b = scale * random.normal( bkey, [ outDim,])
    return w, b

class MLP:

    def __init__( self, dims, key):
        nLayer = len(dims)
        keys = random.split( key, num=nLayer)
        self.params = [Linear(m, n, k) for m, n, k in 
                    zip(dims[:-1], dims[1:], keys)]
    
    def forward( self, params, x):
        # forward the hidden layer 
        for w, b in params[:-1]:
            x = ReLU( jnp.dot( x, w) + b)
        # the output layer 
        y = softmax(jnp.dot( x, params[-1][0]) + params[-1][1])
        return y

    def acc( self, x, y_target):
        log_y_hat = self.forward( self.params, x)
        y_hat_cls = jnp.argmax( log_y_hat, axis=1)
        return jnp.mean( y_hat_cls==y_target)

    def loss( self, params, x, y_target):
        log_y_hat = self.forward( params, x)
        return -jnp.mean( y_target * log_y_hat)

    def step( self, x, y, lr=.01):
        grads = grad( self.loss)( self.params, x, y)
        self.params = [(w - lr * dw, b - lr * db)
          for (w, b), (dw, db) in zip(self.params, grads)]

#---------------------------
#      Simulate on MNIST
#---------------------------

def one_hot(x, k=10, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def sim( dims=[ 784, 256, 128, 10], if_jit=False,
        lr=0.01, maxEpoch=10, batchSize=128, seed=0):

    # record time 
    times_lst = [] 

    # load data 
    x_train, x_test, y_train, y_test = get_MNIST()
    _dataset = torch.utils.data.TensorDataset( 
                        torch.FloatTensor(x_train), 
                        torch.FloatTensor(y_train))
    _dataloader = torch.utils.data.DataLoader( _dataset, 
                    batch_size=batchSize,
                    drop_last=True)
    n_train, n_test = x_train.shape[0], x_test.shape[0]

    # init model 
    model = MLP( dims, key=random.PRNGKey(seed))
    if if_jit: jit_step = jit(model.step)

    # start training 
    x_test  = jnp.reshape( x_test, [n_test, -1])
    y_test  = jnp.array( y_test)
    x_train = jnp.reshape( x_train, [n_train, -1])
    y_train = jnp.array( y_train)
    for epoch in range(maxEpoch):
        start_time = time.time()
        train_acc = model.acc( x_train, y_train)
        test_acc  = model.acc( x_test, y_test)
        print( f'Train acc {train_acc:3f}, Test acc {test_acc:3f}')
        for x_batch, y_batch in _dataloader:
            x_batch = jnp.reshape( x_batch.numpy(), [batchSize, -1])
            y_batch = one_hot(y_batch.numpy())
            if if_jit: 
                jit_step( x_batch, y_batch, lr=lr)
            else:
                model.step( x_batch, y_batch, lr=lr)
        epoch_time = time.time() - start_time
        times_lst.append( epoch_time)
        print(f'Epoch {epoch} in {epoch_time:0.2f}s')
    
    return times_lst

if __name__ == '__main__':

    time1 = sim(if_jit=False)