# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:28:50 2020

@author: xxx
"""

from torch.utils.data.dataset import Dataset
from meta.data.StaticDataset import StaticMetaDataset, FullBatchSampler
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import random
import numpy as np
from PIL import Image
import os
import operator as op
from functools import reduce

def n_choose_r(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom 

class CifarStaticDataset(StaticMetaDataset):
    '''
    Implementation of a static Sinusoid Dataset.
    '''
    def __init__(self, root_dir, mode, no_of_tasks, no_of_data_points_per_task):
        '''
        no_of_tasks: positive integer - initial number of tasks to sample.
        no_of_data_points_per_task - positive int.
        '''
        super(CifarStaticDataset, self).__init__()
        self.data_root = os.path.join(root_dir, 'data')
        self.split_root = os.path.join(root_dir, 'splits', 'bertinetto')
        self.available_tasks = ReadTasks(self.split_root, mode)
        self.no_of_tasks = no_of_tasks
        self.task_parametrization_array = self.generate_task_parametrizations(no_of_tasks)
        self.task_dataset_array = self.generate_task_datasets(self.task_parametrization_array, no_of_data_points_per_task)
    
    #############################################################################
    def generate_task_datasets(self, task_parametrization_array, no_of_points_per_task):
        '''
        Input:
            task_parametrization_array: an array of true task parameters.
            no_of_points_per_task: pos. int
        Output:
            array of SinusoidStaticTask datasets, corresponding to the input parameters.
        '''
        if not task_parametrization_array:
            return []
        else:
            task_dataset_array = np.empty(len(task_parametrization_array), dtype = CifarStaticTask)
    
            for i, task in enumerate(task_parametrization_array):
                task_dataset_array[i] = CifarStaticTask(self.data_root, task, no_of_points_per_task)
                
            return task_dataset_array

    ############################################################################    
    def add_new_tasks(self, no_of_tasks, no_of_points_per_task):
        '''Adds additional tasks to MetaDataset.
        Input: 
            no_of_tasks: pos. int
            no_of_poinmts_per_task: pos. int
        Output:
            None. Adds to the class variables.
        '''
        new_tasks = self.generate_task_parametrizations(no_of_tasks)
        new_task_dataset_array = self.generate_task_datasets(new_tasks, no_of_points_per_task)
        
        if len(new_task_dataset_array)==0:
            print('No tasks have been added. No available tasks')
            return
        else:
            self.task_parametrization_array =  np.concatenate([self.task_parametrization_array, new_tasks], axis = 0)
            self.task_dataset_array = np.concatenate([self.task_dataset_array,new_task_dataset_array], axis = 0)
        
    ###########################################################################        
    def generate_task_parametrizations(self, no_of_tasks : int):
        '''
        Implementation of generate task parametrizations in base class
        '''
        
        try:
            available_tasks = self.available_tasks.copy()
            
            #make sure available tasks are greater than the num of task asked to be created
            num_initial_available_tasks = len(available_tasks)
            assert num_initial_available_tasks >= no_of_tasks, "no_of_tasks greater than available"
        
        except AssertionError:
            no_of_tasks = num_initial_available_tasks
            print('no_of_tasks set to num of available tasks: %d' % no_of_tasks)
            
            if no_of_tasks == 0:
                return []
        
        task_parameterisation_array = []
        for _ in range(no_of_tasks):
            index = random.randint(0, len(available_tasks)-1)
            task = available_tasks.pop(index)
            task_parameterisation_array.append(task)
        
        #make sure tasks are deleted from available task array
        assert len(available_tasks) + no_of_tasks == num_initial_available_tasks
        
        self.available_tasks = available_tasks 
        
        return task_parameterisation_array
    
    ##########################################################################
    def add_data_per_task(self, no_of_data_points_per_task):
        '''
        Adds data points to all tasks in the meta dataset
        Input: pos. int
        '''
        for task in self.task_dataset_array:
            task.add_data(no_of_data_points_per_task)

    ##########################################################################
    def increase_tasks_and_data_per_task(self, task_increase, data_increase):
        '''
        Increases the number of tasks and data points per task by specified number. 
        For newly created tasks will create the same number of data points as in first task 
        in the existing meta-dataset. And then add additional data points to all tasks - new and old.
        '''
        if data_increase > 0:
            self.add_data_per_task(data_increase)
        data_for_new_tasks = self.get_task_length(task_index = 0)
        if task_increase > 0:
            self.add_new_tasks(task_increase, data_for_new_tasks)


class CifarStaticTask(Dataset):
    def __init__(self, data_root: str, task: str, no_of_samples : int):
        self._data_root = data_root
        self._task = task
        
        self.task_root = os.path.join(self.data_root, self.task)
        self.available_images = os.listdir(self.task_root)
        
        if no_of_samples > 0:
            self.data = self.sample_datapoints(no_of_samples)
       
    @property
    def data_root(self):
        return self._data_root
    
    @property
    def task(self):
        return self._task
    
    
    def sample_datapoints(self, no_of_datapoints = 1):
        
        try:
            available_images = self.available_images.copy()
            
            #make sure available datapoints are greater than or equal 
            #to the num of datapoints asked to be sampled
            num_available_images = len(available_images)
            assert num_available_images >= no_of_datapoints, "no_of_datapoints to sample greater than available"
        
        except AssertionError:
            no_of_datapoints = num_available_images
            print('no_of_datapoints set to num of available datapoints, i.e., %d' % num_available_images)
            
            if no_of_datapoints == 0:
                return []
        
        sampled_images = []
        for _ in range(no_of_datapoints):
            index = random.randint(0, len(available_images)-1)
            datapoint = available_images.pop(index)
            sampled_images.append(datapoint)
        
        imgs = []
        for image in sampled_images:
            img = np.array(Image.open(os.path.join(self.task_root, image)))
            img = np.rollaxis(img, 2, 0)  
            imgs.append(img.astype(np.float32))
            
        imgs = np.array(imgs).astype(np.float32)
        
        self.available_images = available_images
        
        return imgs

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]
    
    def add_data(self, no_of_samples):
        new_data = self.sample_datapoints(no_of_samples)
        if len(new_data)==0:
            print('No more available datapoints for task: %s' % self._task)
            return
        else:
            self.data = np.concatenate([self.data, new_data], axis = 0)

class CifarBatchSampler(Sampler):
    '''
    Sampler that samples data from static dataset as follows:
    A sample will contain a given number of tasks (specified at sampler creation), or all tasks if 
    not specified and will sample a given number of data points from a task
    or all data points from a task if data points per task was not specified. 
    This is a batch sampler, and each batch will contain data for a single! task. In other words,
    iterating over this sampler will yield all data corresponding to one task in a single iteration 
    until the whole sample data is exhausted. 
    '''
    ###################################################################
    def __init__(self, data_source, no_of_tasks = None, N_way=5, no_of_points_per_class = None):
        super(CifarBatchSampler, self).__init__(data_source)
        self._data_source = data_source
        self._no_of_tasks = no_of_tasks
        self._N_way = N_way
        self._no_of_points_per_class = no_of_points_per_class

    ###################################################################       
    def __iter__(self):
        if self._no_of_tasks is None:
            num_classes = len(self._data_source)
            no_of_tasks = n_choose_r(num_classes, self._N_way)
            #no_of_tasks = num_classes // self._N_way
        else:
            try:
                no_of_tasks = self._no_of_tasks
                
                num_classes = len(self._data_source)
                num_N_groups = n_choose_r(num_classes, self._N_way)
                
                assert no_of_tasks <= num_N_groups
            
            except AssertionError:
                print('Not enough classes to create %d tasks' % no_of_tasks)
                no_of_tasks = n_choose_r(num_classes, self._N_way)
                print('no_of_tasks has been set to %d' % no_of_tasks)
        
        #class_indices = np.random.choice(a = len(self._data_source), size = no_of_tasks*self._N_way, replace = False)
        for i in range(no_of_tasks):
            #task_class_indices = class_indices[self._N_way*i:self._N_way*(i+1)]
            task_class_indices = np.random.choice(a = len(self._data_source), size = self._N_way)
            
            indices_list = []
            for task_index in task_class_indices:
                task_length = self._data_source.get_task_length(task_index)
            
                if self._no_of_points_per_class is None:
                    no_of_points = task_length
                else:
                    no_of_points = self._no_of_points_per_class
            
                task_data_index_list = np.random.choice(a = task_length, size = no_of_points, replace = False)
                for task_data_point in task_data_index_list:
                    indices_list.append((task_index, task_data_point))
            
            yield indices_list



def ReadTasks(split_root, phase):
    with open(os.path.join(split_root, '%s.txt' % phase)) as f:
        lines = f.readlines()
        
    tasks = []
    for line in lines:
        tasks.append(line[:-1])
    
    return tasks


  

"""

root_dir = 'C:\\Users\\Georgios\\Desktop\\MediaTEK\\natural datasets\\CIFAR-FS\\cifar100\\cifar100'


CifarDataset = CifarStaticDataset(root_dir, 'train', 150, 400)
print(len(CifarDataset.task_dataset_array))

for taskDataset in CifarDataset.task_dataset_array[:10]:
    print(taskDataset.data.shape)

CifarDataset.increase_tasks_and_data_per_task(50, 300)


for taskDataset in CifarDataset.task_dataset_array[:10]:
    print(taskDataset.data.shape)

"""






#CifarSampler = CifarBatchSampler(data_source = CifarDataset, no_of_tasks = 1, N_way = 5, no_of_points_per_class = None)
#dataloader = DataLoader(CifarDataset, batch_sampler=CifarSampler, num_workers=0)


"""
x=[]
for i in range(2):
    for task_idx, task_data in enumerate(dataloader):
        print('-----')
        #print(task_data.size())
"""       





""" 
# test add_data_per_task, increase_tasks_and_data_per_task

print('----------------------------------------------------')
CifarDataset.add_data_per_task(2)
CifarSampler = FullBatchSampler(CifarDataset)
dataloader = DataLoader(CifarDataset, batch_sampler=CifarSampler, num_workers=0)
print(dataloader)

for task_idx, task_data in enumerate(dataloader):
    print(task_data.size())
    
print('------------------------------------------------')

CifarDataset.increase_tasks_and_data_per_task(2, 3)
CifarSampler = FullBatchSampler(CifarDataset, 5, 5)
dataloader = DataLoader(CifarDataset, batch_sampler=CifarSampler, num_workers=0)
print(dataloader)

for task_idx, task_data in enumerate(dataloader):
    print(task_data.size())
"""
