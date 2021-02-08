# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:01:11 2020

@author: xxx
"""

from meta.data.MetaSamplers import UniformMetaSampler
from meta.data.SinusoidGenerator import SinusoidGenerator
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch 


""" Preprocessor class"""
#This class will be removed soon.
#Its sole method SplitBatch does the following:
#Receives the total number of points that will be used for one meta-update Nout*Kin from the dataloader
#It splits the points to 4 lists: train_inputs, train_targets, test_inputs, test_targets 
#to make the dataloader interface smoothly with the mean_outer_loss method of MAMLTrainOP
class Preprocessor:
    def __init__(self, no_of_tasks, no_of_points_per_task, train_test_split):
        self.Nout = no_of_tasks
        self.Kin = no_of_points_per_task
        self.train_test_split = train_test_split
    
    def SplitBatch(self, batch):
        task_IDs = batch[0]
        datapoints = batch[1]
        
        IDs, train_inputs, train_targets, test_inputs, test_targets = [],[],[],[],[]
        
        for i in range(self.Nout):
            task_ID = task_IDs[i*self.Kin]
            IDs.append(task_ID)
            
            task_points = datapoints[i*self.Kin:(i+1)*self.Kin]
            task_train_points = task_points[:int((1-self.train_test_split)*self.Kin)]
            task_test_points = task_points[int((1-self.train_test_split)*self.Kin):]
            
            
            task_train_inputs = task_train_points[:,0].unsqueeze(-1)
            train_inputs.append(task_train_inputs)
            
            task_train_targets = task_train_points[:,1].unsqueeze(-1)
            train_targets.append(task_train_targets)
            
            task_test_inputs = task_test_points[:,0].unsqueeze(-1)
            test_inputs.append(task_test_inputs)
            
            task_test_targets = task_test_points[:,1].unsqueeze(-1)
            test_targets.append(task_test_targets)
        
        return IDs, train_inputs, train_targets, test_inputs, test_targets


class TestPreprocessor:
    def __init__(self, no_of_tasks, no_of_points_per_task):
        self.Nout = no_of_tasks
        self.Kin = no_of_points_per_task
    
    def SplitBatch(self, batch, no_adapt_points):
        task_IDs = batch[0]
        datapoints = batch[1]
        
        IDs, train_inputs, train_targets, test_inputs, test_targets = [], [], [], [], []
        
        for i in range(self.Nout):
            task_ID = task_IDs[i*self.Kin]
            IDs.append(task_ID)
            
            task_points = datapoints[i*self.Kin:(i+1)*self.Kin]
            task_train_points = task_points[:no_adapt_points]
            task_test_points = task_points[no_adapt_points:]
            
            
            task_train_inputs = task_train_points[:,0].unsqueeze(-1)
            train_inputs.append(task_train_inputs)
            
            task_train_targets = task_train_points[:,1].unsqueeze(-1)
            train_targets.append(task_train_targets)
            
            task_test_inputs = task_test_points[:,0].unsqueeze(-1)
            test_inputs.append(task_test_inputs)
            
            task_test_targets = task_test_points[:,1].unsqueeze(-1)
            test_targets.append(task_test_targets)
        
        return IDs, train_inputs, train_targets, test_inputs, test_targets
