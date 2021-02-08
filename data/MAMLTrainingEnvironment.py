# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:49:57 2020

@author: xxx
"""

import numpy as np
from meta.train_ops.MAMLTrainOp import MAMLTrainOp
from meta.train_ops.ClassificationMAMLTrainOP import ClassificationMAMLTrainOP
from meta.meta_learners.RegressionMetaModel import MetaModel
from meta.meta_learners.ClassificationMetaModel import ConvMetaModel
from meta.data.StaticDataset import SinusoidStaticDataset, FullBatchSampler
from meta.data.CorrectCifarDataset import CifarStaticDataset, CifarBatchSampler
from torch.utils.data import DataLoader
import copy
import torch


class MAMLTrainingEnvironment:
    def __init__(self, args, budget: int, initial_state = None):
        self.budget = budget
        self.args = args  

        if initial_state is None:
            initial_state =  np.random.randint(1, 2 * np.floor(np.sqrt(budget)), 2)
            self.initial_state = initial_state
        else:
            self.initial_state = initial_state
            
        self.current_state = initial_state
        self.create_MetaModel()
        self.create_maml_trainer()
        #self.create_lr_scheduler() # this is not used, commented out by wangq 2021Jan7
        
        self.create_dataset(dataset_type='train')
        print(self.dataset_size(self.dataset))
        self.create_dataset(dataset_type='val')
        self.create_dataset(dataset_type='test') # added 16/12/2020

    def dataset_size(self, dataset):
        return [len(dataset), dataset.get_task_length(0)]
    
    def create_MetaModel(self, ):
        if self.args.dataset == 'Cifar':
            self.model = ConvMetaModel(3, self.args.no_of_classes)
        else:
            self.model = MetaModel()
            
    def create_maml_trainer(self, ):
        if self.args.dataset == 'Cifar':
            self.maml_trainer = ClassificationMAMLTrainOP(self.model, self.args)
        else:
            self.maml_trainer = MAMLTrainOp(self.model, self.args)
    
    def create_lr_scheduler(self, factor=0.5, patience=250, verbose=True):
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.maml_trainer.meta_optimizer, 'min', factor=0.5, patience=250, verbose=True)
    
    def create_dataset(self, dataset_type = 'train'):
        if dataset_type == 'train':
            
            no_of_tasks = self.initial_state[0]
            no_of_points_per_task = self.initial_state[1]
            
            if self.args.dataset == 'Cifar':
                self.dataset = CifarStaticDataset(self.args.root_dir, dataset_type, no_of_tasks=no_of_tasks, 
                                                  classes_per_task=self.args.no_of_classes, 
                                                  no_of_data_points_per_task = no_of_points_per_task//self.args.no_of_classes)
            else:
                self.dataset = SinusoidStaticDataset(no_of_tasks, no_of_points_per_task, **self.args.dataset_params)
        
        elif dataset_type == 'val':
            
            no_of_tasks = 500
            no_of_points_per_task = 50
            
            if self.args.dataset == 'Cifar':
                self.val_dataset = CifarStaticDataset(self.args.root_dir, dataset_type, no_of_tasks=no_of_tasks, 
                                                      classes_per_task=self.args.no_of_classes, 
                                                      no_of_data_points_per_task = no_of_points_per_task//self.args.no_of_classes)
            else:
                self.val_dataset = SinusoidStaticDataset(no_of_tasks, no_of_points_per_task, **self.args.dataset_params)
        
        elif dataset_type == 'test':
            no_of_tasks = 1000
            no_of_points_per_task = 50
            
            if self.args.dataset == 'Cifar':
                self.test_dataset = CifarStaticDataset(self.args.root_dir, dataset_type, no_of_tasks=no_of_tasks, 
                                                      classes_per_task=self.args.no_of_classes, 
                                                      no_of_data_points_per_task = no_of_points_per_task//self.args.no_of_classes)
            else:
                self.test_dataset = SinusoidStaticDataset(no_of_tasks, no_of_points_per_task, **self.args.dataset_params)
        
    def create_dataloader(self, dataset = 'train', no_of_tasks=None, no_of_points_per_task=None):
        if self.args.dataset == 'Cifar':
            sampler = CifarBatchSampler(data_source = dataset, no_of_tasks = no_of_tasks, no_of_data_points_per_task = no_of_points_per_task)
        else:
            sampler = FullBatchSampler(dataset, no_of_tasks = no_of_tasks, no_of_points_per_task = no_of_points_per_task)
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=self.args.num_workers)
        return dataloader
    
    
    def save_checkpoint(self,):
        checkpoint = {'model_state_dict': self.maml_trainer.model.state_dict(),
                      'optimizer_state_dict': self.maml_trainer.meta_optimizer.state_dict()}
        
        #torch.save(checkpoint, 'checkpoint.pth')
        self.checkpoint = copy.deepcopy(checkpoint)
        
    def load_checkpoint(self,):
        #checkpoint = torch.load('checkpoint.pth')
        checkpoint = copy.deepcopy(self.checkpoint)
        self.maml_trainer.model.load_state_dict(checkpoint['model_state_dict'])
        self.maml_trainer.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        #self.maml_trainer.model.to(device = self.args.device)
        
        del checkpoint

    def save_initial_checkpoint(self):
        checkpoint = {'model_state_dict': self.maml_trainer.model.state_dict(),
                      'optimizer_state_dict': self.maml_trainer.meta_optimizer.state_dict()}
        
        #torch.save(checkpoint, 'checkpoint.pth')
        self.initial_checkpoint = copy.deepcopy(checkpoint)

    def load_initial_checkpoint(self):
        # to reset the model by loading the random initialization of the model
        checkpoint = copy.deepcopy(self.initial_checkpoint)
        self.maml_trainer.model.load_state_dict(checkpoint['model_state_dict'])
        self.maml_trainer.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint

    def reset(self):
        self.__init__(self.args, self.budget, self.initial_state)

    def reset_model_params(self):
        for layer in self.maml_trainer.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                
    def step(self, action, in_place=True):
        if in_place:
            self.dataset.increase_tasks_and_data_per_task(action[0], action[1])
            self.current_state = self.dataset_size(self.dataset)
    
            if self.current_state[0] * self.current_state[1] >= self.budget:
                done = 1
            else:
                done = 0
            
            return self.current_state, done
        
        else:
            new_dataset = copy.deepcopy(self.dataset)
            new_dataset.increase_tasks_and_data_per_task(action[0], action[1])
            return new_dataset
            
    def step_with_unused_data(self, action, in_place=True):
        if in_place:
            self.dataset.increase_tasks_and_data_per_task(action[0], action[1])
            self.current_state = self.dataset_size(self.dataset)
    
            if self.current_state[0] * self.current_state[1] >= self.budget:
                done = 1
            else:
                done = 0
            
            return self.current_state, done
        
        else:
            new_dataset = copy.deepcopy(self.dataset)
            new_dataset.increase_tasks_and_data_per_task(action[0], action[1])
            return new_dataset
            
        
