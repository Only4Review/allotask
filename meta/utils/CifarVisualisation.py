# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:43:36 2020

@author: xxx
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle
import numpy as np
from random import sample
import pdb


class ClassificationExperimentVisualiser:
    def __init__(self, log_dir = '/allotMeta2/salsa/meta/results/classification/', save_name='visualisations_wrt_datapoints'):
        self.log_dir = log_dir
        
        #create save directory
        self.save_name = save_name
        self.save_dir = os.path.join(log_dir, save_name)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        #create the dictionary that will store all the data for visualisation
        self.data={}
    
    def load_data(self, load_args):
        
        #method to load the data given the appropriate load_args parameters
        results_folder = os.path.join(self.log_dir, '%s/%s/%d-way/budget_%d' % (load_args['dataset'], 
                                                                               load_args['ExperimentType'], 
                                                                               load_args['no_of_classes'],
                                                                               load_args['budget']
                                                                               )
                                                                               )
        
        if load_args['dataset'] not in self.data.keys():
            self.data[load_args['dataset']] = {}
        
        if load_args['ExperimentType'] not in self.data[load_args['dataset']].keys():
            self.data[load_args['dataset']][load_args['ExperimentType']] = {}
        
        if load_args['no_of_classes'] not in self.data[load_args['dataset']][load_args['ExperimentType']].keys():
            self.data[load_args['dataset']][load_args['ExperimentType']][load_args['no_of_classes']] = {}
        
        
        config_data={}
        configuration_folders = os.listdir(results_folder)
        
        for config_folder in configuration_folders:
            if not os.path.exists(os.path.join(results_folder, config_folder, 'log', 'log.pickle')):
                print('No log.pickle file for %s configuration' % (config_folder))
                continue
            
            if load_args['ExperimentType'] == 'BaselineExperiment':
                no_of_tasks = int(config_folder.split('_')[1]) # x-axis variable: 0--tasks, 1--datapoints
                datapoints_per_task_per_taskclass = int(config_folder.split('_')[1])
                run = int(config_folder.split('_')[2])
                
                if no_of_tasks not in config_data.keys():
                    config_data[no_of_tasks]={}
                
                with open(os.path.join(results_folder, config_folder, 'log', 'log.pickle'), 'rb') as handle:
                        config_data[no_of_tasks][run] = pickle.load(handle)
            
            elif load_args['ExperimentType'] == 'HandcraftedPolicy':
                train_iters_btw_decisions = 50 #int(config_folder.split('_')[1])
                #if train_iters_btw_decisions != 250:
                #    pass                
                run = int(config_folder.split('_')[-1])
                with open(os.path.join(results_folder, config_folder, 'log', 'log.pickle'), 'rb') as handle:
                        config_data[run] = pickle.load(handle)
                        
        
        self.data[load_args['dataset']][load_args['ExperimentType']][load_args['no_of_classes']][load_args['budget']] = config_data
    
    def GeneratePerformanceCurve(self, load_args, policy_budget_list, y_axis = 'test_loss'):
        plt.figure()
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        
        #plt.title('%s - %s - %d-way - budget:%d' % (load_args['dataset'], load_args['ExperimentType'], load_args['no_of_classes'], load_args['budget']), fontsize=16)
        
        ExperimentData = self.data[load_args['dataset']][load_args['ExperimentType']][load_args['no_of_classes']][load_args['budget']]
        no_of_tasks_list = ExperimentData.keys()
        #values for the scatter plot
        all_task_values = []
        all_y_axis_values = []
        
        #values for the avg curve
        task_values = []
        avg_y_axis_values = []
        
        for no_of_tasks in sorted(no_of_tasks_list):
            y_axis_values = []
            for run in ExperimentData[no_of_tasks].keys():
                y_axis_value = ExperimentData[no_of_tasks][run][y_axis]
                
                if y_axis == 'test_loss':
                    y_axis_value = y_axis_value.item()
                if y_axis == 'train_avg_loss':
                    y_axis_value = np.mean(y_axis_value[-10:])
                     
                y_axis_values.append(y_axis_value)
                
                all_task_values.append(no_of_tasks)
                all_y_axis_values.append(y_axis_value)
            
            task_values.append(no_of_tasks)
            avg_y_axis_value = np.mean(np.array(y_axis_values))
            avg_y_axis_values.append(avg_y_axis_value)
       
        plt.scatter(np.array(all_task_values), np.array(all_y_axis_values), s=10, marker='o', label='GS runs')
        plt.plot(task_values, avg_y_axis_values, label='GS mean')
        
        if self.data[load_args['dataset']].get('HandcraftedPolicy'):
            PolicyData = self.data[load_args['dataset']]['HandcraftedPolicy'][load_args['no_of_classes']]
            print(PolicyData.keys())
            for budget in policy_budget_list:
                number_of_runs = len(PolicyData[budget].keys())
                policy_tasks = np.zeros(number_of_runs)
                test_accuracies = np.zeros(number_of_runs)
                for i, run in enumerate(PolicyData[budget].keys()):
                    allocated_tasks = PolicyData[budget][run]['allocation_history'][-1,0]
                    policy_tasks[i] = allocated_tasks
                    test_accuracies[i] = PolicyData[budget][run]['test_accuracy']
                
                if budget==62500:
                    budget=60000
                    
                plt.scatter(policy_tasks, test_accuracies, s=12, marker='o', label='SDM %d' % budget)      
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('number of datapoints per class', fontsize=15)
        
        if y_axis == 'test_accuracy':
            y_label = 'test accuracy'
        elif y_axis == 'test_loss':
            y_label = 'test loss'
        elif y_axis == 'train_avg_loss':
            y_label = 'train avg loss'
        elif y_axis == 'train_avg_accuracy':
            y_label = 'train avg accuracy'
            
        plt.ylabel(y_label, fontsize=15)
        plt.grid(which='both', axis='both')
        if 'loss' in y_axis:
            lg = plt.legend(loc='upper right')
        else:
            lg = plt.legend(loc='lower right')
        
        outer_file_name = '%s %s %s' % (load_args['dataset'],load_args['ExperimentType'],load_args['no_of_classes'])
        savefile = os.path.join(self.save_dir, outer_file_name)
        Path(savefile).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(savefile, '%d_%s.png' % (load_args['budget'], y_axis)), bbox_extra_artists=(lg,), bbox_inches='tight')
    
    def GenerateMultipleCurves(self, load_args, budget_list, y_axis = 'test_loss',legendLabels=None):
        plt.figure()
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        
        #plt.title('%s - %s - %d-way - budget:%d' % (load_args['dataset'], load_args['ExperimentType'], load_args['no_of_classes'], load_args['budget']), fontsize=16)
        multipleBudgetCurves_path = os.path.join(self.save_dir, 'multipleBudgetCurves')
        Path(multipleBudgetCurves_path).mkdir(parents=True, exist_ok=True)
        
        optimal_datapoints_list = []
        optimal_test_accuracy_list = []
        iter = 0
        for budget in budget_list:
            
            load_args['budget'] = budget
            self.load_data(baseline_load_args)
            ExperimentData = self.data[load_args['dataset']][load_args['ExperimentType']][load_args['no_of_classes']][budget]
            no_of_tasks_list = ExperimentData.keys()
            #values for the scatter plot
            all_task_values = []
            all_y_axis_values = []
            
            #values for the avg curve
            task_values = []
            avg_y_axis_values = []
            
            for no_of_tasks in sorted(no_of_tasks_list):
                if no_of_tasks > 600:
                    continue
                
                y_axis_values = []
                for run in ExperimentData[no_of_tasks].keys():
                    try:
                        y_axis_value = ExperimentData[no_of_tasks][run][y_axis] * 100
                    except:
                        break
                    
                    if y_axis == 'test_loss':
                        y_axis_value = y_axis_value.item()
                    if y_axis == 'train_avg_loss':
                        y_axis_value = np.mean(y_axis_value[-10:])
                    y_axis_values.append(y_axis_value)
                    
                    all_task_values.append(no_of_tasks)
                    all_y_axis_values.append(y_axis_value)
                
                task_values.append(no_of_tasks)
                avg_y_axis_value = np.mean(np.array(y_axis_values))
                avg_y_axis_values.append(avg_y_axis_value)

            plt.scatter(np.array(all_task_values), np.array(all_y_axis_values), s=5, marker='o')
            if legendLabels is not None:
                plt.plot(task_values, avg_y_axis_values,label='b='+legendLabels[iter])
            else:
                plt.plot(task_values, avg_y_axis_values,label=str(budget))
            iter += 1
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlabel('number of datapoints per class', fontsize=15)
        plt.xlabel('Data points per class', fontsize=12)
        
        if y_axis == 'test_accuracy':
            y_label = 'test accuracy (%)'
        elif y_axis == 'test_loss':
            y_label = 'test loss'
        elif y_axis == 'train_avg_loss':
            y_label = 'train avg loss'
        elif y_axis == 'train_avg_accuracy':
            y_label = 'train avg accuracy'
            
        plt.ylabel(y_label, fontsize=12)
        plt.yticks([30,40,50,60,70],[30,40,50,60,70])
        plt.grid(which='both', axis='both')
        if 'loss' in y_axis:
            lg = plt.legend(loc='upper right',bbox_to_anchor=(1., 0),fontsize=12)
        else:
            lg = plt.legend(loc='lower right',bbox_to_anchor=(1., 0),fontsize=7)
        
        outer_file_name = '%s %s %s' % (load_args['dataset'],load_args['ExperimentType'],load_args['no_of_classes'])
        savefile = os.path.join(self.save_dir, outer_file_name)
        Path(savefile).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(multipleBudgetCurves_path, '%d_%s.pdf' % (load_args['budget'], y_axis)), bbox_extra_artists=(lg,), bbox_inches='tight', dpi=300)

    def GenerateTrainAccCurves(self, load_args, budget_list, y_axis = 'train_avg_accuracy',legendLabels=None):
        plt.figure()
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        
        #plt.title('%s - %s - %d-way - budget:%d' % (load_args['dataset'], load_args['ExperimentType'], load_args['no_of_classes'], load_args['budget']), fontsize=16)
        multipleBudgetCurves_path = os.path.join(self.save_dir, 'multipleBudgetCurves')
        Path(multipleBudgetCurves_path).mkdir(parents=True, exist_ok=True)
        
        optimal_datapoints_list = []
        optimal_test_accuracy_list = []
        avg_y_axis_values_all_budgets = []
        avg_y_axis_stdValues_all_budgets = []
        iter = 0
        for budget in budget_list:
            
            load_args['budget'] = budget
            self.load_data(baseline_load_args)
            ExperimentData = self.data[load_args['dataset']][load_args['ExperimentType']][load_args['no_of_classes']][budget]
            no_of_tasks_list = ExperimentData.keys()
            #values for the scatter plot
            all_task_values = []
            all_y_axis_values = []
            
            #values for the avg curve
            task_values = []
            avg_y_axis_values = []
            for no_of_tasks in sorted(no_of_tasks_list):
                y_axis_values = []
                if budget == 4 or budget<1000:
                    runs_with_trainAcc_list = [3,4]
                else:
                    runs_with_trainAcc_list = [5,6]
                for run in runs_with_trainAcc_list:#ExperimentData[no_of_tasks].keys():
                    try:
                        y_axis_value = ExperimentData[no_of_tasks][run][y_axis]
                    except:
                        continue
                    
                    if y_axis == 'test_loss':
                        y_axis_value = y_axis_value.item()
                    if y_axis == 'train_avg_loss':
                        y_axis_value = np.mean(y_axis_value[-10:])
                    y_axis_values.append(y_axis_value)
                    
                    all_task_values.append(no_of_tasks)
                    all_y_axis_values.append(y_axis_value)
                
                task_values.append(no_of_tasks)
                avg_y_axis_value = np.mean(np.array(y_axis_values))
                avg_y_axis_values.append(avg_y_axis_value)                
            iter += 1
            avg_y_axis_values_all_budgets.append(np.nanmean(avg_y_axis_values))
            avg_y_axis_stdValues_all_budgets.append(np.nanstd(avg_y_axis_values))
        plt.errorbar(budget_list, avg_y_axis_values_all_budgets, avg_y_axis_stdValues_all_budgets, label=str(budget)+' budget GS mean')
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlabel('number of datapoints per class', fontsize=15)
        plt.xlabel('datapoints per class', fontsize=12)
        
        if y_axis == 'test_accuracy':
            y_label = 'test accuracy'
        elif y_axis == 'test_loss':
            y_label = 'test loss'
        elif y_axis == 'train_avg_loss':
            y_label = 'train avg loss'
        elif y_axis == 'train_avg_accuracy':
            y_label = 'train avg accuracy'
            
        plt.ylabel(y_label, fontsize=15)
        plt.grid(which='both', axis='both')
        if 'loss' in y_axis:
            lg = plt.legend(loc='upper right',bbox_to_anchor=(1., 0),fontsize='xx-small')
        else:
            lg = plt.legend(loc='lower right',bbox_to_anchor=(1., 0),fontsize='xx-small')
        
        outer_file_name = '%s %s %s' % (load_args['dataset'],load_args['ExperimentType'],load_args['no_of_classes'])
        savefile = os.path.join(self.save_dir, outer_file_name)
        Path(savefile).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(multipleBudgetCurves_path, '%d_%s.png' % (load_args['budget'], y_axis)), bbox_extra_artists=(lg,), bbox_inches='tight')
        return avg_y_axis_values_all_budgets, avg_y_axis_stdValues_all_budgets



    def optimal_datapoints_with_budget_curve(self, load_args,budget_list, y_axis='test_accuracy'):
        optimal_allocation_path = os.path.join(self.save_dir, 'optimal no of datapoints')
        Path(optimal_allocation_path).mkdir(parents=True, exist_ok=True)      

        optimal_datapoints_list = []
        optimal_test_accuracy_list = []
        optimal_datapoints_list_multipleSampling = []
        for budget in budget_list:
            load_args['budget'] = budget
            self.load_data(baseline_load_args)
            ExperimentData = self.data[load_args['dataset']][load_args['ExperimentType']][load_args['no_of_classes']][budget]
            number_of_runs = len(ExperimentData.keys())
            no_of_tasks_list = ExperimentData.keys()

            numSampling = 200
            
            optimal_datapoints_list = []
            for iSample in range(numSampling):
                #values for the scatter plot
                all_task_values = []
                all_y_axis_values = []
                
                #values for the avg curve
                task_values = []
                avg_y_axis_values = []
                no_of_tasks_list = sorted(no_of_tasks_list)
                
                for no_of_tasks in no_of_tasks_list:
                    y_axis_values = []                
                    for run in sample(ExperimentData[no_of_tasks].keys(), 1):
                        y_axis_value = ExperimentData[no_of_tasks][run][y_axis]
                        
                        if y_axis == 'test_loss':
                            y_axis_value = y_axis_value.item()
                        if y_axis == 'train_avg_loss':
                            y_axis_value = np.mean(y_axis_value[-10:])
                            
                        y_axis_values.append(y_axis_value)
                        
                        all_task_values.append(no_of_tasks)
                        all_y_axis_values.append(y_axis_value)
                    
                    task_values.append(no_of_tasks)
                    avg_y_axis_value = np.mean(np.array(y_axis_values))
                    avg_y_axis_values.append(avg_y_axis_value)

                #get optimal datapoints for this budget
                optimal_datapoints = no_of_tasks_list[np.argmax(avg_y_axis_values)]
                optimal_datapoints_list.append(optimal_datapoints)

            optimal_datapoints_list_multipleSampling.append(optimal_datapoints_list)
        optimal_datapoints_list_multipleSampling = np.array(optimal_datapoints_list_multipleSampling)
        plt.figure()
        plt.rc('xtick', labelsize=6)
        plt.rc('ytick', labelsize=6)
        plt.errorbar(budget_list, np.mean(optimal_datapoints_list_multipleSampling,1), np.std(optimal_datapoints_list_multipleSampling,1), marker='o', label='GS mean')
        #plt.plot(range(7), optimal_datapoints_list, marker='o')
        plt.savefig(os.path.join(optimal_allocation_path, 'optimal_datapoints_budget_errorbar.png'),dpi=300)
        return optimal_datapoints_list_multipleSampling



#### optimal datapoints / training accuracy as a function of budget
budget_list =  [800, 2000, 4000, 8000, 10000, 20000, 30000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 4]
#[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
#budget_list =  [800, 2000, 4000, 8000, 20000, 40000, 60000, 100000, 140000, 180000, 4]
#budget_list = [100,200,300,400,500,600,700,800,900]
load_dir = '/allotMeta2/salsa/meta/results/classification'
vis = ClassificationExperimentVisualiser(load_dir)
baseline_load_args = {'dataset':'Cifar', 'ExperimentType': 'BaselineExperiment', 'no_of_classes': 5}
optimal_datapoints_list_multipleSampling = vis.optimal_datapoints_with_budget_curve(baseline_load_args, budget_list, y_axis='test_accuracy')
optimal_train_acc_list, optimal_train_acc_std_list = vis.GenerateTrainAccCurves(baseline_load_args, budget_list, y_axis='train_avg_accuracy')


plt.figure()
plt.errorbar(budget_list[:-1], np.mean(optimal_datapoints_list_multipleSampling[:-1,],1), np.std(optimal_datapoints_list_multipleSampling[:-1,],1), marker='o')
plt.errorbar([200000], np.mean(optimal_datapoints_list_multipleSampling[-1,]), np.std(optimal_datapoints_list_multipleSampling[-1,]), marker='o', color='tab:red')
plt.xlabel('Budget', fontweight='bold')
plt.ylabel('Optimal data points per class', fontweight='bold')
xtickLabels = ['', '', '', '', '10000', '20000', '30000', '40000', '60000', '80000', '100000', '120000', '140000', '160000', '180000', 'infinite']
plt.xticks(budget_list[:-1]+[200000], xtickLabels, rotation=45)
plt.savefig(os.path.join(load_dir,'visualisations_wrt_datapoints/Cifar_optimalNumDatapoints_budget.png'), bbox_inches='tight', dpi=300)    

plt.figure()
plt.plot(budget_list[:-1], 100*np.array(optimal_train_acc_list[:-1]), marker='o')
plt.plot([200000], 100*np.array(optimal_train_acc_list[-1]), marker='o', color='tab:red')
plt.xlabel('Budget', fontweight='bold')
plt.ylabel('Training Accuracy (%)', fontweight='bold')
plt.xticks(budget_list[:-1]+[200000], xtickLabels, rotation=45)
plt.savefig(os.path.join(load_dir,'visualisations_wrt_datapoints/Cifar_trainingAcc_budget.png'), bbox_inches='tight', dpi=300)    




'''
##### plot curves wrt different learning rates alpha
budget_list = [60003,60000,60004,60005,60006,60007,60008]
load_dir = '/allotMeta2/salsa/meta/results/classification'
vis = ClassificationExperimentVisualiser(load_dir)
baseline_load_args = {'dataset':'Cifar', 'ExperimentType': 'BaselineExperiment', 'no_of_classes': 5}
legendLabels = ['0.005','0.01','0.02','0.05','0.1','0.2','0.5']
vis.GenerateMultipleCurves(baseline_load_args, budget_list, y_axis='test_accuracy', legendLabels=legendLabels)
optimal_datapoints_list_multipleSampling = vis.optimal_datapoints_with_budget_curve(baseline_load_args, budget_list, y_axis='test_accuracy')
import pdb
pdb.set_trace()
plt.figure(figsize=(5,2))
lr_list = [0.005,0.01,0.02,0.05,0.1,0.2,0.5]
plt.errorbar(lr_list, np.mean(optimal_datapoints_list_multipleSampling,1), np.std(optimal_datapoints_list_multipleSampling,1), marker='o',color = 'k')
plt.xlabel('inner loop learning rate')
plt.ylabel('optimal data points \n per class')
plt.xscale('log')
plt.grid(which='both', axis='both')
#plt.xticks(range(len(budget_list)), legendLabels)
plt.savefig(os.path.join(load_dir,'visualisations_wrt_datapoints/OptimalNumDatapoints_wrt_lr.pdf'), bbox_inches='tight', dpi=300)    
'''

'''
#### multiple curves in one plot
budget_list =  [800, 2000, 4000, 8000, 20000, 40000, 80000, 100000, 140000, 4]
#budget_list = [100,200,300,400,500,600,700,800,900]
load_dir = '/allotMeta2/salsa/meta/results/classification'
vis = ClassificationExperimentVisualiser(load_dir)
baseline_load_args = {'dataset':'Cifar', 'ExperimentType': 'BaselineExperiment', 'no_of_classes': 5}
legendLabels = ['800', '2000', '4000', '8000', '20000', '40000', '80000', '100000', '140000', 'infinite']
vis.GenerateMultipleCurves(baseline_load_args, budget_list, y_axis='test_accuracy', legendLabels=legendLabels)
'''
