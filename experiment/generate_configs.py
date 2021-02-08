# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:57:34 2020

@author: xxx
"""

"""This file is used to create the list of configurations for Experiment 1"""
import json

def create_Nout(value):
    if value>100:
        value_list = [1, 10, 100, value]
    elif 10<value and value<=100:
        value_list = [1, 10, value]
    elif 1<value and value<=10:
        value_list = [1, value]
    elif value == 1:
        value_list = [1]
    else:
        return Exception('Not valid value')
    
    return value_list

def create_Kin_list(value, train_test_split):
    if train_test_split == 0.1:
        min_value = 10
    elif train_test_split == 0.25:
        min_value = 4
    elif train_test_split == 0.5:
        min_value = 2
    
    
    if min_value<10:
        if value>100:
            value_list = [min_value, 10, 100, value]
        elif 10<value and value<=100:
            value_list = [min_value, 10, value]
        elif min_value<value and value<=10:
            value_list = [min_value, value]
        elif value == min_value:
            value_list = [min_value]
        else:
            return []
    
    elif min_value==10:
        if value>100:
            value_list = [10, 100, value]
        elif 10<value and value<=100:
            value_list = [10, value]
        elif value==10:
            return [10]
        else:
            return []
        
    
    
    return value_list
 
"""       
def write_config(B_list, p_list, train_test_split_list):
    valid_configurations = []
    for B in B_list:
        for p in p_list:
            N = int(1/p) #no tasks
            if N > B:
                continue
            K = int(B/N) #datapoints per task
            for Nout in create_Nout(N):
                for tt_split_ratio in train_test_split_list:
                    Kin_list = create_Kin_list(K, tt_split_ratio)
                    if Kin_list == []:
                        continue
                    else:
                        for Kin in Kin_list:
                            configuration={'Budget':B, 'p':p, 
                                           'Kin':Kin, 'Nout':Nout, 
                                           'train_test_split':tt_split_ratio}
                            valid_configurations.append(configuration)
                        
                        
    #print(len(valid_configurations))
    with open('config_experiment1.json', 'w') as f:
        json.dump(valid_configurations, f)              

B_list = [10000, 20000, 50000, 100000, 1000000]
p_list = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
train_test_split_list = [0.1, 0.25, 0.5]
"""
import numpy as np
def write_config(B_list, p_list, train_test_split_list, runs):
    valid_configurations = []
    for B in B_list:
        for p in p_list:
            N = int(1/p) #no tasks
            if N > B:
                continue
            K = int(np.ceil(B/N)) #datapoints per task
            for tt_split_ratio in train_test_split_list:
                Kin_list = create_Kin_list(K, tt_split_ratio)
                if Kin_list == []:
                    continue
                else:
                    for run in range(1, runs+1):
                        configuration={'Budget':B, 'p':p, 
                                       'Kin':K, 'Nout':N, 
                                       'train_test_split':tt_split_ratio,
                                       'run':run}
                        valid_configurations.append(configuration)
                        print(configuration)
                        
    print(len(valid_configurations))
    with open('config_experiment1.json', 'w') as f:
        json.dump(valid_configurations, f)              



B_list = [10000]
#p_list = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.0142, 0.025, 0.05, 0.1, 0.5, 1]
p_list = [0.0142]
train_test_split_list = [0.1]
runs=10

write_config(B_list, p_list, train_test_split_list, runs)
