# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:51:53 2020

@author: xxx
"""

import numpy as np

def stopping_criterion(lr_rates):
    #input: a list of learning rates used in each epoch
    #output: True if training must stop, False otherwise
            
    """Description: checks if we have more than k lr annealings in 10 consecutive epochs
        to determine whether to stop the training or not. k will be determined 
        experimentally. This criterion is subject to modification
    """
    
    annealing_steps=0
    for i in range(len(lr_rates)-1):
        if lr_rates[i] != lr_rates[i+1]:
            annealing_steps+=1
    
    if annealing_steps>3:
        return True
    else:
        return False