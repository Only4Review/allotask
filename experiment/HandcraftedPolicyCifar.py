import sys
sys.path.insert(0,'/proj/gpu_xxx/allotMeta2/salsa')
from meta.data.MAMLTrainingEnvironment import MAMLTrainingEnvironment
import meta.CONSTANTS as see
from meta.utils.ClassificationExperimentLogger import ClassificationExperimentLogger
from meta.data_allocator.Alg3Policy import Alg3Policy
from meta.data_allocator.Alg3PolicyAllocator import Alg3PolicyAllocator

import torch
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt
import argparse
import os
import pickle
import numpy as np

class HandcraftedPolicyClassificationExperiment:
    def __init__(self, args, config):
        self.args = args #argument parser object
        self.config = config #dict: contains configuration info
        
    def read_config(self, config):
        return config
    
    @staticmethod
    def update_args_from_config(args, config):
        args.no_of_classes = config['no_of_classes']
        args.dataset = config['dataset']

        return args
    
    def setup_logs(self, init=True):
        #re-write the global logger
        see.logs = ClassificationExperimentLogger(self.config)
        
        if init:
            see.logs.make_logdir()
    
            # save the training and configuration information
            see.logs.write(self.config, 'config.pickle')
            see.logs.write(self.args, 'args.pickle')
            
            see.logs.cache = {}
            see.logs.cache['action_rewards']={}
            
            see.logs.cache['train_avg_loss'] = []
            see.logs.cache['test_avg_loss'] = {}
            see.logs.cache['val_avg_loss'] = []
            see.logs.cache['lr_rates'] = []
            see.logs.cache['expected_meta_update_time'] = []
            see.logs.cache['total_meta_updates'] = []
        
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

            env.reset_model_params()
            train_dataloader = env.create_dataloader(dataset = env.dataset, no_of_tasks = 25, no_of_points_per_task = 50)
            # training the model here will lead to overfitting and hence lower final test accuracy --- wangq
            train_loss = 0 #env.maml_trainer.training_phase(train_dataloader, num_of_training_iterations=train_iters_btw_decisions)
                
            print('Training between decisions - train loss(avg): %.3f' % train_loss)
            
            #-----------------allocation---------------
            allocator.env = env
            action = allocator.select_action(use_moments = use_moments)
            print(action)
            if action[0]==0 and action[1]==0:
                break
            
            action, new_state, reached_budget = allocator.update_state(action)
            allocation_history.append(new_state)
            env = allocator.env
            
            #--------calculate the size of the augmented dataset-----
            train_dataset_size = env.dataset_size(env.dataset)
            print('Dataset Size After Allocation: ', train_dataset_size)
            
            '''
            #----------plot the action rewards--------
            plt.figure()
            plt.title('Action Rewards')
            actions = list(allocator.policy.actions.keys())
            for action in actions:
                see.logs.cache['action_rewards'][action] = allocator.policy.action_rewards[action]
                plt.plot(allocator.policy.action_rewards[action], label = action)
            plt.legend()
            plt.savefig('policy_rewards.png')
            plt.close('all')
            '''    
            
        #save the allocation history
        allocation_history = np.array(allocation_history)
        see.logs.cache['allocation_history'] = allocation_history
        
        #-----------------(to be remove from the official code)---------
        '''
        #plot the allocation history (quick plotting for debugging)
        plt.figure()
        plt.plot(allocation_history[:,0], allocation_history[:,1])
        plt.xlabel('tasks')
        plt.ylabel('datapoints per task')
        plt.savefig('allocation.png')
        #--------------------
        '''
        #train till convergence
        train_dataset_size = env.dataset_size(env.dataset)
        train_no_of_tasks = train_dataset_size[0]
        train_no_of_points_per_task = train_dataset_size[1]
        
        train_dataloader = env.create_dataloader(dataset = env.dataset, no_of_tasks = 25, no_of_points_per_task = 50)
        val_dataloader = env.create_dataloader(dataset = env.val_dataset, no_of_tasks = None, no_of_points_per_task = None)
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
        
        
if __name__ == '__main__':
    # 1.) set args
    
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    
    parser.add_argument('--dataset', type=str, default='cifar-fs')
    
    parser.add_argument('--exp_config_dir', type=str, default='meta/experiment/config_experiment1.json',
            help='Directory of the configuration of the experiment.')

    parser.add_argument('--root-dir', type=str, default='meta/dataset/cifar100',
                        help='root directory folder')
    
    parser.add_argument('--train_test_split_inner', type=int, default=0.5,
            help='Train test split for the inner loop. Default: 0.3')
    
    parser.add_argument('--first-order', action='store_true',
            help='Use the first-order approximation of MAML.')
    
    parser.add_argument('--inner-step-size', type=float, default=0.01,
            help='Step-size for the inner loop. Default: 0.01')
    
    parser.add_argument('--train-adapt-steps', type=int, default=1,
            help='Number of inner gradient updates during training. Default: 1')
    
    parser.add_argument('--eval-adapt-steps', type=int, default=5,
            help='Number of inner gradient updates during evaluation. Default: 5')
    
    parser.add_argument('--meta_lr', type=float, default=0.001,
            help='The learning rate of the meta optimiser (outer loop lr). Default: 0.001')
    
    parser.add_argument('--hidden-size', type=int, default=64,
            help='Number of channels for each convolutional layer (default: 64).')
    
    parser.add_argument('--max-num-epochs', type=int, default=60000,
            help='Max Number of epochs')
    
    parser.add_argument('--num-workers', type=int, default=4,
            help='Number of workers for data loading (default: 4).')

    parser.add_argument('--train-iters-for-decisions', type=int, default=500,
            help='train iters for decisions (default: 500).')
    
    parser.add_argument('--addProportion', type=float, default=0.1,
            help='The proportion of added data points to the those in the current dataset when making decision. Default: 0.1')
    
    parser.add_argument('--budget', type=int, default=60000,
            help='Budget')

    parser.add_argument('--use-cuda', type=int, default=1, help='For cuda set to 1. For cpu set to 0. Default: 1.')
    args = parser.parse_args()
    
    #args.DataloaderProcessing = DataloaderProcessing #data processing function for preparing the data for the meta_outer_loss method of MAMLTrainOP
    args.loss_func = CrossEntropyLoss() #multi-class classification
    
    if torch.cuda.is_available() and args.use_cuda == 1:
        args.device = torch.device('cuda')
        print('Device: ', args.device)
    elif not(torch.cuda.is_available()) and args.use_cuda == 1:
        args.device = torch.device('cpu')
        print('Cuda is not available. Using cpu instead.')
    else:
        args.device = torch.device('cpu')
        print('Device: ', args.device)
        
    no_of_classes = 5
    
    configurations = []
    
    budget = args.budget
    start_run = 0
    end_run = 5            
    for run in range(start_run, end_run):
        config = {'dataset': 'Cifar', 'ExperimentType': 'HandcraftedPolicy', 'no_of_classes': no_of_classes,
                  'budget': budget, 'search_range':(100*args.addProportion, -1), 'train_iters_btw_decisions':250, 'train_iters_for_decisions':args.train_iters_for_decisions,
                  'use_moments':'1', 'run':run, 'start_point': [20,20], 'eval_adapt_steps':args.eval_adapt_steps}
        configurations.append(config)
    
    print(configurations)
    
    for config in configurations:
        print(config)
        updated_args = HandcraftedPolicyClassificationExperiment.update_args_from_config(args, config)
        policy_exp = HandcraftedPolicyClassificationExperiment(updated_args, config)
        policy_exp.setup_logs()
        policy_exp.run()
