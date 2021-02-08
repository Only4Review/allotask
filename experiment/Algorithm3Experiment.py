# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:32:26 2020

@author: xxx
"""

from meta.experiment.Experiment import Experiment
from meta.meta_learners.RegressionMetaModel import MetaModel
from meta.train_ops.MAMLTrainOp import MAMLTrainOp
from meta.data.MAMLTrainingEnvironment import MAMLTrainingEnvironment
from meta.data_allocator.Alg3Policy import Alg3Policy
from meta.data_allocator.Alg3PolicyAllocator import Alg3PolicyAllocator
from meta.utils.VarianceExperimentLogger import VarianceExperimentLogger

#dataloader related modules
from meta.data.StaticDataset import SinusoidStaticDataset, FullBatchSampler
from torch.utils.data.dataloader import DataLoader
#logging
from meta.utils.setup_logging import setup_logging
import meta.CONSTANTS as see
#generic modules
import numpy as np

import argparse
import torch
import matplotlib.pyplot as plt
import os 
import pickle
import tracemalloc

class Algorithm3Experiment:
    def __init__(self, experimentID, args, config):
        self.experimentID = experimentID
        self.args = args
        self.config = config
    
    def setup_logs(self, init=True):
        #re-write the global logger
        see.logs = VarianceExperimentLogger(dataset='sinusoid', ID=self.experimentID, config = self.config)
        if init:
            see.logs.make_logdir()
    
            # save the training and configuration information
            see.logs.write(self.args, 'args.pickle')
            
            see.logs.cache = {}
            see.logs.cache['action_rewards']={}
            see.logs.cache['train_tasks_hist_data'] = {}
            see.logs.cache['test_tasks_hist_data'] = {}
            see.logs.cache['train_avg_loss'] = []
            see.logs.cache['test_avg_loss'] = {}
            see.logs.cache['val_avg_loss'] = []
            see.logs.cache['lr_rates'] = []
            see.logs.cache['expected_meta_update_time'] = []
            see.logs.cache['total_meta_updates'] = []
            
            see.logs.cache['allocation_history'] = None
        else:
            with open(os.path.join(see.logs.log_folder, 'log.pickle'), 'rb') as handle:
                see.logs.cache = pickle.load(handle)
            
        
    def run(self):

        budget = self.config['budget']
        start_point =  self.config['start_point']
        search_range = self.config['search_range']
        train_iters_btw_decisions = self.config['train_iters_btw_decisions']
        use_moments = self.config['use_moments']
        
        env = MAMLTrainingEnvironment(self.args, budget, initial_state = start_point)
        
        policy = Alg3Policy(policy_grid=search_range)
        allocator = Alg3PolicyAllocator(policy, env)
        
        allocation_history = []
        allocation_history.append(start_point)
        
        reached_budget=False
        while not reached_budget:
            #print('-------------new decision-----------')
            
            #--------training with current dataset--------')
            train_dataset_size = env.dataset_size(env.dataset)
            train_no_of_tasks = train_dataset_size[0]
            train_no_of_points_per_task = train_dataset_size[1]
            
            train_dataloader = env.create_dataloader(dataset = env.dataset, no_of_tasks=None, no_of_points_per_task=None)
            _ = env.maml_trainer.training_phase(train_dataloader, num_of_training_iterations=train_iters_btw_decisions)
            
            #-----------------allocation---------------
            allocator.env = env
            action = allocator.select_action(use_moments = use_moments)
            print('Action Taken: ', action)
            if action[0]==0 and action[1]==0:
                break
            
            action, new_state, reached_budget = allocator.update_state(action)
            allocation_history.append(new_state)
            env = allocator.env
            
            #--------calculate the size of the augmented dataset-----
            train_dataset_size = env.dataset_size(env.dataset)
            print('Dataset Size After Allocation: ', train_dataset_size)
            
            #----------plot the action rewards--------
            
            plt.figure()
            plt.title('Action Rewards')
            actions = list(allocator.policy.actions.keys())
            for action in actions:
                see.logs.cache['action_rewards'][action] = allocator.policy.action_rewards[action]
                plt.plot(allocator.policy.action_rewards[action], label = action)
            plt.legend()
            plt.savefig('policy_rewards.png')
                
            
        #save the allocation history
        allocation_history = np.array(allocation_history)
        see.logs.cache['allocation_history'] = allocation_history
        
        #-----------------(to be remove from the official code)---------
        #plot the allocation history (quick plotting for debugging)
        plt.figure()
        plt.plot(allocation_history[:,0], allocation_history[:,1])
        plt.xlabel('tasks')
        plt.ylabel('datapoints per task')
        plt.savefig('allocation.png')
        #--------------------
        
        #train till convergence
        train_dataset_size = env.dataset_size(env.dataset)
        train_no_of_tasks = train_dataset_size[0]
        train_no_of_points_per_task = train_dataset_size[1]
        
        train_dataloader = env.create_dataloader(dataset = env.dataset, no_of_tasks=None, no_of_points_per_task = None)
        val_dataloader = env.create_dataloader(dataset = env.val_dataset, no_of_tasks=None, no_of_points_per_task = None)
        env.maml_trainer.train(train_dataloader, val_dataloader)
        
        #do evaluation of performance on the test dataset
        self.evaluate(env)
    
    def evaluate(self, env):
        env.create_dataset('test')
        test_dataloader = env.create_dataloader(env.test_dataset, no_of_tasks = None, no_of_points_per_task=None)

        # Update the model in the train_op
        env.maml_trainer.model = see.logs.load_model(checkpoint_index='best')
        env.maml_trainer.model.eval()
        
        see.logs.cache['test_loss'] = env.maml_trainer.mean_outer_loss(test_dataloader)
        see.logs.cache['test_accuracy'] = env.maml_trainer.get_accuracy(test_dataloader)
        
        print('test_loss: ', see.logs.cache['test_loss'])
        print('test_accuracy: ', see.logs.cache['test_accuracy'])
        
        #update the log file
        see.logs.write(see.logs.cache, name='log.pickle')

def data_and_labels_sinusoid_splitter(data):
    assert(data.shape[1] == 2)
    return data[:, 0].unsqueeze(-1), data[:, 1].unsqueeze(-1)

if __name__ == '__main__':
    # 1.) set args
    
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    #parser.add_argument('folder', type=str,
    #        help='Path to the folder the data is downloaded to.')
    
    parser.add_argument('--exp_config_dir', type=str, default='meta/experiment/config_experiment1.json',
            help='Directory of the configuration of the experiment.')
    
    parser.add_argument('--dataset', type=str, default='sinusoid',
                        help='dataset')
    
    parser.add_argument('--data-root', type=str, default='meta/data/',
                        help='data folder')
    
    parser.add_argument('--K', type=int, default=15,
            help='Number of datapoints per task')
    
    parser.add_argument('--Kin', type=int, default=15,
            help='Inner loop batch size')
    
    parser.add_argument('--N', type=int, default=150,
            help='Number of tasks.')
    
    parser.add_argument('--Nout', type=int, default=25,
            help='Outer loop batch size')
    
    parser.add_argument('--train_test_split_inner', type=int, default=0.1,
            help='Train test split for the inner loop. Default: 0.1')
    
    parser.add_argument('--first-order', action='store_true',
            help='Use the first-order approximation of MAML.')
    
    parser.add_argument('--inner-step-size', type=float, default=0.01,
            help='Step-size for the inner loop. Default: 0.01')
    
    parser.add_argument('--num_adaptation_steps', type=int, default=1,
            help='Number of inner gradient updates during training. Default: 1')
    
    parser.add_argument('--meta_lr', type=float, default=0.001,
            help='The learning rate of the meta optimiser (outer loop lr). Default: 0.001')
    
    parser.add_argument('--hidden-size', type=int, default=64,
            help='Number of channels for each convolutional layer (default: 64).')
    
    parser.add_argument('--output-folder', type=str, default=None,
            help='Path to the output folder for saving the model (optional).')
    
    parser.add_argument('--max-num-epochs', type=int, default=2500,
            help='Max Number of epochs')
    
    parser.add_argument('--num-workers', type=int, default=0,
            help='Number of workers for data loading (default: 0).')
    
    parser.add_argument('--use-cuda', action='store_true',
            help='Use CUDA if available.')
    
    args = parser.parse_args()
    
    args.data_and_labels_splitter = data_and_labels_sinusoid_splitter
    args.device = torch.device('cuda')

    # 2.) create the dataset 
    amplitude_range = (0.1,5)
    phase_range = (0, np.pi)
    noise_range = (0,0.1)
    x_range = (-5, 5)
    
    dataset_params = {'amplitude_range': amplitude_range,
                      'phase_range': phase_range,
                      'noise_std_range': noise_range,
                      'x_range': x_range}
    
    args.dataset_params = dataset_params
    
    
    #Create the configurations of the policy experiment
    experimentID = 9
    configurations = []
    for budget in [2000, 5000]:
        search_range_len = 2
        train_iters_btw_decisions =  100
        for use_moments in ['1']:
            start_run = 0
            end_run = 10
    
            for run in range(start_run, end_run):
                config={}
                config['budget'] = budget
                
                
                config['search_range'] = (search_range_len,search_range_len) #be careful  
                #do not get confused by the key name 'search_range' - result of previous verion naming and new modifications in the policy code
                # this will give only two actions [search_range_len,0] and [0,search_range_len]
                # so if search_range_len=1, then you either add one task or one datapoint per each task in each step of the allocation procedure
                
                config['train_iters_btw_decisions'] = train_iters_btw_decisions
                config['use_moments'] = use_moments
                config['run'] = run
                config['start_point'] = np.array([5,5])
                configurations.append(config)
    
    
    #training       
    for config in configurations:
        print(config)
        exp = Algorithm3Experiment(experimentID, args, config)
        exp.setup_logs()
        exp.run()
        
        #evaluation is done in run method.
