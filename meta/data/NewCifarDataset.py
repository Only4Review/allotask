# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:26:09 2020

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
    def __init__(self, root_dir, mode, no_of_tasks, classes_per_task, no_of_data_points_per_task):
        '''
        no_of_tasks: positive integer - initial number of tasks to sample.
        no_of_data_points_per_task - positive int.
        '''
        super(CifarStaticDataset, self).__init__()
        self.data_root = os.path.join(root_dir, 'data')
        self.split_root = os.path.join(root_dir, 'splits', 'bertinetto')
        self.available_classes = ReadTasks(self.split_root, mode)
        self.class_data_dict = self.load_data()
        
        self.classes_per_task = classes_per_task
        self.task_additions = 0
        self.task_seed = 0
        self.task_parametrization_array = self.generate_task_parametrizations(no_of_tasks)
        self.task_dataset_array = self.generate_task_datasets(self.task_parametrization_array, no_of_data_points_per_task)
    
    ###########################################################################
    def load_data(self, ):
        class_data_dict = {}
        for _class in self.available_classes: 
            class_root_dir = os.path.join(self.data_root, _class)
            class_images_names = os.listdir(class_root_dir)
            class_images = []
            for image in class_images_names:
                img = np.array(Image.open(os.path.join(class_root_dir, image)))
                img = np.rollaxis(img, 2, 0)  
                class_images.append(img.astype(np.float32))
            
            class_data_dict[_class] = class_images
        
        return class_data_dict
            
    ###########################################################################        
    def generate_task_parametrizations(self, no_of_tasks : int):
        '''
        Implementation of generate task parametrizations in base class
        '''
        
        try:
            available_tasks = n_choose_r(len(self.available_classes), no_of_tasks) - self.task_additions
            assert available_tasks >= no_of_tasks, "no_of_tasks greater than available"
        
        except AssertionError:
            no_of_tasks = available_tasks
            print('no_of_tasks set to num of available tasks: %d' % no_of_tasks)
            
            if no_of_tasks == 0:
                return []
        
        task_parameterisation_array = []
        for i in range(no_of_tasks):
            random.seed(self.task_seed)
            self.task_seed += 1
            
            task_classes = random.sample(self.available_classes, self.classes_per_task)
            assert task_classes not in task_parameterisation_array
                
            task_parameterisation_array.append(task_classes)
        
        self.task_additions += len(task_parameterisation_array)

        return task_parameterisation_array
    
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
    def __init__(self, data_root: str, task: list, no_of_samples : int):
        self._data_root = data_root
        self._task = task #task is a list of strings. Each string corresponds to one class.
        self.task_classes = task
        
        self.task_roots = {}
        self.available_images = {}
        for task_class in self.task_classes:
            self.task_roots[task_class] = os.path.join(self.data_root, task_class)
            self.available_images[task_class] = os.listdir(self.task_roots[task_class])
        
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
            
            min_num_available_images_accros_task_classes = min([len(self.available_images[x]) for x in self.task_classes])
            
            #make sure available datapoints are greater than or equal 
            #to the num of datapoints asked to be sampled
            
            assert min_num_available_images_accros_task_classes >= no_of_datapoints, "no_of_datapoints to sample greater than available for some task class"
        
        except AssertionError:
            no_of_datapoints = min_num_available_images_accros_task_classes
            print('no_of_datapoints set to num of min_num_available_images_accros_task_classes, i.e., %d' % no_of_datapoints)
            
            if no_of_datapoints == 0:
                return []
        
        task_images = []
        for task_class in self.task_classes:
            class_sampled_images = []
            for _ in range(no_of_datapoints):
                index = random.randint(0, len(available_images[task_class])-1)
                datapoint = available_images[task_class].pop(index)
                class_sampled_images.append(datapoint)
            
            class_images = []
            for image in class_sampled_images:
                img = np.array(Image.open(os.path.join(self.task_roots[task_class], image)))
                img = np.rollaxis(img, 2, 0)  
                class_images.append(img.astype(np.float32))
            
            class_images = np.array(class_images).astype(np.float32)
            task_images.append(class_images)
        
        if len(task_images)>0:
            task_images = np.array(task_images).astype(np.float32)
            task_images = np.swapaxes(task_images, 0, 1)
        
        self.available_images = available_images
        
        return task_images

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
    def __init__(self, data_source, no_of_tasks = None, no_of_data_points_per_task = None):
        super(CifarBatchSampler, self).__init__(data_source)
        self._data_source = data_source
        self._no_of_tasks = no_of_tasks
        self._no_of_data_points_per_task = no_of_data_points_per_task

    ###################################################################       
    def __iter__(self):
        if self._no_of_tasks is None:
            no_of_tasks = len(self._data_source)
        else:
            try:
                no_of_tasks = self._no_of_tasks
                assert no_of_tasks <= len(self._data_source), 'no of tasks greater than available'
            except AssertionError:
                no_of_tasks = len(self._data_source)
                print('no of tasks set to available tasks: %d' % len(self._data_source))
        
        task_indices = np.random.choice(a = len(self._data_source), size = no_of_tasks, replace = False)
        for task in task_indices:
            indices_list = []
            task_length = self._data_source.get_task_length(task)

            if self._no_of_data_points_per_task is None:
                no_of_points = task_length
            else:
                try:
                    no_of_points = self._no_of_data_points_per_task
                    assert no_of_points <= task_length, 'no_of_datapoints more than available'
                
                except AssertionError:
                    no_of_points = task_length
                    print('_no_of_data_points_per_task set to num available points: %d' % task_length)

            task_data_index_list = np.random.choice(a = task_length, size = no_of_points, replace = False)

            for task_data_point in task_data_index_list:
                indices_list.append((task, task_data_point))

            yield indices_list



def ReadTasks(split_root, phase):
    with open(os.path.join(split_root, '%s.txt' % phase)) as f:
        lines = f.readlines()
        
    tasks = []
    for line in lines:
        tasks.append(line[:-1])
    
    return tasks


  



root_dir = 'C:\\Users\\Georgios\\Desktop\\MediaTEK\\natural datasets\\CIFAR-FS\\cifar100\\cifar100'


CifarDataset = CifarStaticDataset(root_dir, 'train', no_of_tasks=10, classes_per_task=5, no_of_data_points_per_task=20)
CifarSampler = CifarBatchSampler(data_source = CifarDataset, no_of_tasks = 5, no_of_data_points_per_task = 10)
dataloader = DataLoader(CifarDataset, batch_sampler=CifarSampler, num_workers = 0)
for data in dataloader:
    print(data.shape)



"""
print(len(CifarDataset.task_dataset_array))

for taskDataset in CifarDataset.task_dataset_array[:10]:
    print(taskDataset.data.shape)

CifarDataset.increase_tasks_and_data_per_task(2, 30)


for taskDataset in CifarDataset.task_dataset_array[:10]:
    print(taskDataset.data.shape)

print(len(CifarDataset))
print(len(CifarDataset.task_dataset_array))

for task in CifarDataset.task_parametrization_array:
    print(task)
"""





#CifarSampler = CifarBatchSampler(data_source = CifarDataset, no_of_tasks = 1, N_way = 5, no_of_points_per_class = None)
#dataloader = DataLoader(CifarDataset, batch_sampler=CifarSampler, num_workers=0)
