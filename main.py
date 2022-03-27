# basic 
import os
import time 
import numpy as np 

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
    return data, label 


if __name__ == '__main__':

    x, y = get_MNIST()
    

