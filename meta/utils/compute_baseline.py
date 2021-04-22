# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:07:57 2020

@author: xxx
"""

import numpy as np
from tqdm import tqdm

"""
N=50000
K=50000
total_avg_loss = 0
for i in tqdm(range(N)):
    A = np.random.uniform(0.1, 5)
    phi = np.random.uniform(0, np.pi)
    x_range = np.random.uniform(-5, 5, size=K)
    y_range = A*np.sin(x_range+phi)
    loss = np.mean(np.square(y_range))
    
    total_avg_loss += loss
    

total_avg_loss/=N
print('total avg loss: %.5f' % total_avg_loss)
"""
