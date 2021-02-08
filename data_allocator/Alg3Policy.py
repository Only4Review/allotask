# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:32:50 2020

@author: xxx
"""

import pdb

class Alg3Policy:
    def __init__(self, policy_grid=(5,5)):
        self.policy_grid = policy_grid
        self.actions = {1:[self.policy_grid[0],0], 2:[0,self.policy_grid[1]]}
        self.action_rewards = {1:[], 2:[]}

    def check_budget_violation(self, dataset, budget):
        tasks = len(dataset)
        datapoints_per_task = dataset.get_task_length(0)
        if tasks*datapoints_per_task>budget:
            return True
        else:
            return False
        
    def evaluate(self, env, use_moments):
        env.save_checkpoint()
        
        if use_moments=='1':
            # modified by wangq 8/12/2020: add the same amount of data for two actions
            if self.policy_grid[0] > 0 and self.policy_grid[1] >0:
                actions = {1:[self.policy_grid[0],0], 2:[0,self.policy_grid[1]]}
            else:
                num_current_tasks = len(env.dataset)
                datapoints_per_current_tasks = env.dataset.get_task_length(0)
                datapoints_added = num_current_tasks * datapoints_per_current_tasks * env.args.addProportion
                num_added_points_per_class = int(datapoints_added // (num_current_tasks * env.args.no_of_classes))
                num_added_tasks = int(datapoints_added // datapoints_per_current_tasks)
                if num_added_points_per_class < 2:
                    num_added_points_per_class = 2
                    num_added_tasks = int(num_added_points_per_class * env.args.no_of_classes * num_current_tasks // datapoints_per_current_tasks)
                actions = {1: [num_added_tasks, 0],
                           2: [0, num_added_points_per_class]}
                print(actions)
            actions2moments = {}
            for action in actions.keys():
                 
                new_dataset = env.step(actions[action], in_place=False)

                #tasks = len(new_dataset)
                #datapoints_per_task = new_dataset.get_task_length(0)
                
                #check for budget violation
		# not need to check voilation here, checked in main script, over budget once is allowed ---wangq
                violation = self.check_budget_violation(new_dataset, env.budget)
                
                if violation:
                    continue
                else:
                    """Load model from current checkpoint"""
                    #model_snapshot = env.create_MetaModel_snapshot()
                    env.load_checkpoint()
                    env.reset_model_params() # this might not reset all parameters, allowing learning somehow?
                    """create the new train dataloader"""
                    ##(max value for datapoints per task set to 150 to avoid OOM)
                    new_train_dataloader = env.create_dataloader(dataset = new_dataset, no_of_tasks=50, no_of_points_per_task=50)
                    
                    """train for a number of meta-updates using the new temporary dataset"""
                    train_loss_reward_phase = env.maml_trainer.training_phase(new_train_dataloader, num_of_training_iterations=env.args.train_iters_for_decisions)
                    
                    print('Training in rewards calculations phase - train loss(avg): %.3f' % train_loss_reward_phase)
                    
                    """evaluate expected performance on the validation dataset"""
                    env.maml_trainer.model.eval()
                    val_dataloader = env.create_dataloader(dataset = env.val_dataset, no_of_tasks=None, no_of_points_per_task=None)
                    empirical_mean = env.maml_trainer.mean_outer_loss(val_dataloader).item()
                    empirical_acc = env.maml_trainer.get_accuracy(val_dataloader)
                    #eval_dataloader = env.create_dataloader(dataset = new_dataset, no_of_tasks=None, no_of_points_per_task=None)
                    #empirical_mean = env.maml_trainer.compute_moments(eval_dataloader, permutations=10)
                    
                    self.action_rewards[action].append(empirical_mean)
                    self.action_rewards[action].append(empirical_acc)
                    #store results
                    actions2moments[action]={}
                    actions2moments[action]['mean'] = empirical_mean
                    actions2moments[action]['acc'] = empirical_acc
                    #update model from the saved snapshot
                    #env.update_MetaModel_from_snapshot(model_snapshot)#update model from snapshot
                    
                    #delete snapshot to avoid potential memory leakage
                    #del model_snapshot
                
                del new_dataset
                
            """load the checkpoint after the calculation of the rewards of all actions"""
            env.load_checkpoint()
            
            print(actions2moments)#print the action rewards for debugging
            
            if len(list(actions2moments.keys()))==0:
                return [0,0]
            elif len(list(actions2moments.keys()))==1:
                return actions[list(actions2moments.keys())[0]]
            else:
                if actions2moments[1]['mean']<actions2moments[2]['mean']:
                    action = 1
                else:
                    action = 2
                
                return actions[action]
    
                
            
