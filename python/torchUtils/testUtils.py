import numpy as np
import torch

def initialize_weights(array):
    shape = array.shape
    nelem = np.array(shape).prod()
    weights = (torch.arange(0,nelem)+1)/nelem 
    return weights.reshape(shape)

def initialize_linear(linear):
    weights = initialize_weights(linear.weight)
    bias = initialize_weights(linear.bias)
    
    linear.weight.data = weights
    linear.bias.data = bias