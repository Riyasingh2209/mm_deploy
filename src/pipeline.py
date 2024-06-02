import pandas as pd
import numpy as np
import math

from src.config import config

theta0 = None
theta  = None
 
def initialize_layer_biases(num_units):
    return np.random.uniform(low=-1,high=1,size=(1,num_units))

def initialize_layer_weights(num_units_l_1, num_units_1):
    return np.random.uniform(low=-1,high=1,size=(num_units_l_1,num_units_1))

for l in range(1,config.NUM_LAYERS-1):
    
    theta0.append(initialize_layer_biases(config.P[1]/math.sqrt(config.P[l-1])))
    theta.append(initialize_layer_weights(config.P[l-1],config.P[1]/math.sqrt(config.P[l-1])))
    
theta0.append(initialize_layer_biases(config.P[1]/math.sqrt(config.P[l-1])))
theta.append(initialize_layer_weights(config.P[1],config.P[1]/math.sqrt(config.P[l-1])))
     