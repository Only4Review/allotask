# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:03:57 2020

@author: xxx
"""

#This file received the log directory and generates the visual inspection graphs.

import pickle
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from random import sample
        
#Class GraphGenerator is used to plot the data saved in the log file of an experiment.
class GraphGenerator:
    def __init__(self, log_dir, save_dir_name='visualisations'):
        self.log_dir = log_dir
        self.save_dir = os.path.join(log_dir, save_dir_name)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
    
    def TrainTimes(self, budget_list):
        info = {}
        
        result_folders = os.listdir(self.log_dir)
        for folder in result_folders:
            
            if folder == 'visualisations':
                continue
            elif float(folder.split('_')[-1]).is_integer()==False:
                continue
            elif os.path.exists(os.path.join(self.log_dir, folder, 'log', 'log.pickle'))==False:
                continue
            
            with open(os.path.join(self.log_dir, folder, 'log', 'config.pickle'), 'rb') as handle:
                config = pickle.load(handle)
            
            budget = config['Budget']
            
            if budget not in budget_list:
                continue
            
            p = config['p']
                        
            run = config['run']
            
            if budget not in info.keys():
                info[budget]={}
            
            if p not in info[budget].keys():
                info[budget][p]={}
            
            if run not in info[budget][p].keys():
                info[budget][p][run]={}
            
            with open(os.path.join(self.log_dir, folder, 'log', 'log.pickle'), 'rb') as handle:
                log_dict = pickle.load(handle)
            
            info[budget][p][run]['total_meta_updates']=log_dict['total_meta_updates'][0]
            info[budget][p][run]['expected_meta_update_time']=log_dict['expected_meta_update_time'][0]
        
        Path(os.path.join(self.save_dir,'TrainTimes')).mkdir(parents=True, exist_ok=True)
        
        for budget in info.keys():
            p_vals = list(info[budget].keys())
            total_times = []
            for p in p_vals:
                total_time = 0
                num_runs = len(info[budget][p].keys())
                for run in info[budget][p].keys():
                    total_time += info[budget][p][run]['total_meta_updates']*info[budget][p][run]['expected_meta_update_time']
                total_times.append(total_time)
            
            zipped = zip(p_vals, total_times)
            sorted_zipped = sorted(zipped, key = lambda t: t[0])
            p_vals = [x for x, _ in sorted_zipped]
            total_times = [x for _, x in sorted_zipped]
                
            plt.figure()
            plt.title('Training Times - Total Time: %.1f s  -  Budget:%d ' % (sum(total_times), budget))
            plt.plot(p_vals, total_times, marker="o")
            plt.ylabel('Total Time')
            plt.xlabel('p')
            plt.savefig(os.path.join(self.save_dir,'TrainTimes','TrainTimes_%d.png' % budget), bbox_inches='tight')
            plt.close('all')
            
               
                
    def TestLossGraph(self, custom_adapt_pts, budget_list):
        def sort_based_on_p(p_vals, vals1, vals2):
            zipped = zip(p_vals, vals1, vals2)
            sorted_zipped = sorted(zipped, key = lambda t: t[0])
            p_vals = [x for x, _, _ in sorted_zipped]
            vals1 = [x for _, x, _ in sorted_zipped]
            vals2 = [x for _, _, x in sorted_zipped]
            return p_vals, vals1, vals2
            
        
        #create the structure of the performance folder
        Path(os.path.join(self.save_dir,'Performance')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.save_dir,'Performance/overlaid')).mkdir(parents=True, exist_ok=True)
            
        info = {}

        result_folders = os.listdir(self.log_dir)

        for folder in result_folders:
            
            if folder == 'visualisations':
                continue
            elif float(folder.split('_')[-1]).is_integer()==False:
                continue
            elif os.path.exists(os.path.join(self.log_dir, folder, 'log', 'log.pickle'))==False:
                continue
            
            #with open(os.path.join(self.log_dir, folder, 'log', 'config.pickle'), 'rb') as handle:
            #    config = pickle.load(handle)
            
            budget = int(folder.split('_')[0])
            
            if budget not in budget_list:
                continue
            
            p = float(folder.split('_')[1])
            
            run = int(folder.split('_')[-1])
            
            if budget not in info.keys():
                info[budget]={}
            
            if p not in info[budget].keys():
                info[budget][p]={}
            
            if run not in info[budget][p].keys():
                info[budget][p][run]={}
            
            with open(os.path.join(self.log_dir, folder, 'log', 'log.pickle'), 'rb') as handle:
                log_dict = pickle.load(handle)
            for adapt_point in log_dict['test_avg_loss'].keys():
                info[budget][p][run][adapt_point]=log_dict['test_avg_loss'][adapt_point]
            
        #save the data for the overlaid budget performance graph
        overlaid_info_full_adaptation = {}
        
        #initialise the dict for the overlaid budget performance for fixed adaptation
        overlaid_info_fixed_adaptation = {}
        for adapt_pt in custom_adapt_pts:
            overlaid_info_fixed_adaptation[adapt_pt]={}
        optimal_datapoints_list = []
        plt.figure()

        for budget in sorted(info.keys()):
            p_values = list(info[budget].keys())
            loss_values = {'mean':{}, 'min':{}}
            all_y_axis_values = []
            all_no_datapoints = []
            for p in p_values:
                adapt_pt_vals = {'mean':{}, 'min':{}}
                for adapt_point in [500]:#info[budget][p][1].keys():
                    mean_loss_val = 0
                    min_loss_val = 10**5
                    num_runs = 0#len(info[budget][p].keys())
                    y_axis_values = []
                    for run in info[budget][p].keys():
                        try:
                            mean_loss_val += info[budget][p][run][adapt_point]
                            num_runs += 1
                        except:
                            continue
                        if info[budget][p][run][adapt_point]<min_loss_val:
                            min_loss_val = info[budget][p][run][adapt_point]
                        all_y_axis_values.append(info[budget][p][run][adapt_point])
                        all_no_datapoints.append(round(budget*p))
                    mean_loss_val /= num_runs
                    
                    adapt_pt_vals['mean'][adapt_point] = mean_loss_val
                    adapt_pt_vals['min'][adapt_point] = min_loss_val
                    
                loss_values['mean'][p] = adapt_pt_vals['mean']
                loss_values['min'][p] = adapt_pt_vals['min']
            
            #'Fixed budget - Varying adapt point at test time performance curve'
            sorted_p_values = sorted(p_values)
            
            plt.scatter(np.array(all_no_datapoints), np.array(all_y_axis_values), s=5, marker='o')
            adapt_point = 500
            plt.plot([round(budget*p) for p in sorted_p_values], [loss_values['mean'][p][adapt_point] for p in sorted_p_values], markersize=4, marker="o", label='b='+str(budget))

        plt.xlabel('data points per task', fontsize=15)
        plt.ylabel('test loss (%s)' % 'mean', fontsize=15)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([7,100000])
        plt.grid(which='both', axis='both')
        plt.legend(bbox_to_anchor=(0.75, 0), loc='lower left')
        plt.savefig(os.path.join(self.save_dir,'Performance/overlaid/test_loss_allbudget_%d.pdf' % budget), bbox_inches='tight', dpi=300)              


    def optimalNumDatapointsGraph(self, custom_adapt_pts, budget_list):
        def sort_based_on_p(p_vals, vals1, vals2):
            zipped = zip(p_vals, vals1, vals2)
            sorted_zipped = sorted(zipped, key = lambda t: t[0])
            p_vals = [x for x, _, _ in sorted_zipped]
            vals1 = [x for _, x, _ in sorted_zipped]
            vals2 = [x for _, _, x in sorted_zipped]
            return p_vals, vals1, vals2
            
        
        #create the structure of the performance folder
        Path(os.path.join(self.save_dir,'Performance')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.save_dir,'Performance/overlaid')).mkdir(parents=True, exist_ok=True)
            
        info = {}

        result_folders = os.listdir(self.log_dir)

        for folder in result_folders:
            
            if folder == 'visualisations':
                continue
            elif float(folder.split('_')[-1]).is_integer()==False:
                continue
            elif os.path.exists(os.path.join(self.log_dir, folder, 'log', 'log.pickle'))==False:
                continue
            
            #with open(os.path.join(self.log_dir, folder, 'log', 'config.pickle'), 'rb') as handle:
            #    config = pickle.load(handle)
            
            budget = int(folder.split('_')[0])
            
            if budget not in budget_list:
                continue
            
            p = float(folder.split('_')[1])
            
            run = int(folder.split('_')[-1])
            
            if budget not in info.keys():
                info[budget]={}
            
            if p not in info[budget].keys():
                info[budget][p]={}
            
            if run not in info[budget][p].keys():
                info[budget][p][run]={}
            
            with open(os.path.join(self.log_dir, folder, 'log', 'log.pickle'), 'rb') as handle:
                log_dict = pickle.load(handle)
            for adapt_point in log_dict['test_avg_loss'].keys():
                info[budget][p][run][adapt_point]=log_dict['test_avg_loss'][adapt_point]
            
        #save the data for the overlaid budget performance graph
        overlaid_info_full_adaptation = {}
        
        #initialise the dict for the overlaid budget performance for fixed adaptation
        overlaid_info_fixed_adaptation = {}
        for adapt_pt in custom_adapt_pts:
            overlaid_info_fixed_adaptation[adapt_pt]={}

        optimal_datapoints_list_multipleSampling = []
        for iSample in range(1000):
            optimal_datapoints_list = []
            for budget in sorted(info.keys()):
                p_values = list(info[budget].keys())
                loss_values = {'mean':{}, 'min':{}}
                for p in p_values:
                    adapt_pt_vals = {'mean':{}, 'min':{}}

                    for adapt_point in [500]:#info[budget][p][1].keys():
                        mean_loss_val = 0
                        num_runs = 0#len(info[budget][p].keys())
                        for run in sample(info[budget][p].keys(),1):
                            try:
                                mean_loss_val += info[budget][p][run][adapt_point]
                                num_runs += 1
                            except:
                                continue
                        mean_loss_val /= num_runs
                        
                        adapt_pt_vals['mean'][adapt_point] = mean_loss_val
                    loss_values['mean'][p] = adapt_pt_vals['mean']
                sorted_p_values = sorted(p_values)
                adapt_point = 500
                try:
                    tmp = np.array([loss_values['mean'][p][500].cpu() for p in sorted_p_values])
                    optimal_index = np.argmin(tmp)
                except:
                    import pdb
                    pdb.set_trace()
                    continue

                optimal_datapoints_list.append(int(sorted_p_values[optimal_index]*budget))              
            optimal_datapoints_list_multipleSampling.append(optimal_datapoints_list)
        optimal_datapoints_list_multipleSampling = np.array(optimal_datapoints_list_multipleSampling)
        '''
        plt.figure()
        plt.xlabel('Budget')
        plt.ylabel('Optimal number of data points per task')
        plt.errorbar(budget_list, np.mean(optimal_datapoints_list_multipleSampling,0),np.std(optimal_datapoints_list_multipleSampling,0), marker='o')
        plt.savefig(os.path.join(self.save_dir,'Performance/overlaid/optimal_datapoints_budget.png'), bbox_inches='tight')    
        plt.close('all')
        '''
        return optimal_datapoints_list_multipleSampling

    def TrainLossGraph(self, custom_adapt_pts, budget_list):
        def sort_based_on_p(p_vals, vals1, vals2):
            zipped = zip(p_vals, vals1, vals2)
            sorted_zipped = sorted(zipped, key = lambda t: t[0])
            p_vals = [x for x, _, _ in sorted_zipped]
            vals1 = [x for _, x, _ in sorted_zipped]
            vals2 = [x for _, _, x in sorted_zipped]
            return p_vals, vals1, vals2
            
        
        #create the structure of the performance folder
        Path(os.path.join(self.save_dir,'Performance')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.save_dir,'Performance/overlaid')).mkdir(parents=True, exist_ok=True)
            
        train_info = {}
        test_info = {}

        result_folders = os.listdir(self.log_dir)

        for folder in result_folders:
            
            if folder == 'visualisations':
                continue
            elif float(folder.split('_')[-1]).is_integer()==False:
                continue
            elif os.path.exists(os.path.join(self.log_dir, folder, 'log', 'log.pickle'))==False:
                continue
            
            #with open(os.path.join(self.log_dir, folder, 'log', 'config.pickle'), 'rb') as handle:
            #    config = pickle.load(handle)
            
            budget = int(folder.split('_')[0])
            
            if budget not in budget_list:
                continue
            
            p = float(folder.split('_')[1])
            no_datapoints = float(folder.split('_')[2])           
            run = int(folder.split('_')[-1])
            
            if budget not in test_info.keys():
                test_info[budget]={}
                train_info[budget]={}
            if p not in test_info[budget].keys():
                test_info[budget][p]={}
                train_info[budget][p]={}
            if run not in test_info[budget][p].keys():
                test_info[budget][p][run]={}
                train_info[budget][p][run]={}
            with open(os.path.join(self.log_dir, folder, 'log', 'log.pickle'), 'rb') as handle:
                log_dict = pickle.load(handle)

            train_info[budget][p][run]=np.mean(log_dict['train_avg_loss'][-10:])
            test_info[budget][p][run]=log_dict['test_avg_loss']
        #save the data for the overlaid budget performance graph
        overlaid_info_full_adaptation = {}

        #initialise the dict for the overlaid budget performance for fixed adaptation
        overlaid_info_fixed_adaptation = {}
        for adapt_pt in custom_adapt_pts:
            overlaid_info_fixed_adaptation[adapt_pt]={}
        optimal_datapoints_list = []
        optimal_train_loss_list = []
        plt.figure()

        for budget in sorted(test_info.keys()):
            p_values = list(test_info[budget].keys())
            train_loss_values = {'mean':{}}
            test_loss_values = {'mean':{}}
            for p in p_values:
                adapt_pt_vals = {'mean':{}}                
                mean_train_loss = 0
                mean_test_loss = 0
                num_runs = len(test_info[budget][p].keys())
                for run in test_info[budget][p].keys():
                   
                    mean_test_loss += test_info[budget][p][run][500]
                    mean_train_loss += train_info[budget][p][run]

                mean_test_loss /= num_runs
                mean_train_loss /= num_runs
                    
                train_loss_values['mean'][p] = mean_train_loss
                test_loss_values['mean'][p] = mean_test_loss
            
            
            #'Fixed budget - Varying adapt point at test time performance curve'
            sorted_p_values = sorted(p_values)
            #plt.figure()
            #plt.title('Budget:%d \n Performance Profile for varying adapt points at meta-testing' % budget)
            plt.plot([round(budget*p) for p in sorted_p_values], [train_loss_values['mean'][p] for p in sorted_p_values], marker="o", label=budget)
            #optimal_index = np.argmin(np.array([train_loss_values['mean'][p] for p in sorted_p_values]))
            optimal_index = np.argmin(np.array([test_loss_values['mean'][p] for p in sorted_p_values]))
            #optimal_index = np.argmin(sorted_p_values)
            print(sorted_p_values[optimal_index]*budget)
            optimal_train_loss = train_loss_values['mean'][sorted_p_values[optimal_index]]
            optimal_datapoints_list.append(int(sorted_p_values[optimal_index]*budget))      
            optimal_train_loss_list.append(optimal_train_loss)          
        plt.xlabel('number of data points')
        plt.ylabel('Train Loss (%s)' % 'mean')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.savefig(os.path.join(self.save_dir,'Performance/overlaid/training_loss_allbudget_%d.png' % budget), bbox_inches='tight')    
        
        plt.figure()
        plt.xlabel('Budget')
        plt.ylabel('Train Loss (%s)' % 'mean')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.grid(which='both', axis='both')
        plt.plot(budget_list,optimal_train_loss_list,marker='o')
        plt.savefig(os.path.join(self.save_dir,'Performance/overlaid/training_loss_wrt_budget.png'), bbox_inches='tight')    
            

        plt.close('all')
        return optimal_train_loss_list



'''
### Test loss vs datapoints for different budgets
remote_dir = '/allotMeta2/salsa/meta/results/experiment_5_sinusoid'
grapher = GraphGenerator(remote_dir)

#budget_list = [201, 601, 1001, 4001, 8000,10000, 20000 ,40000, 60000, 80000]
budget_list = [8,16,32,64, 201, 601, 1001, 4001]
#grapher.TrainLoss(budget_list)
custom_adapt_pts = [500]
grapher.TrainLossGraph(custom_adapt_pts, budget_list)
#grapher.TrainTimes(budget_list)
'''
    
'''
### optimal number of datapoints vs learning rate
remote_dir = '/allotMeta2/salsa/meta/results/experiment_5_sinusoid'
grapher = GraphGenerator(remote_dir)

#budget_list = [100, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000, 20000, 30000, 40000, 60000, 80000]
budget_list = [10002, 10003, 10004, 10000, 10005, 10006, 10007, 10008]
custom_adapt_pts = [500]
#grapher.TestLossGraph(custom_adapt_pts, budget_list)
optimal_train_loss_list = grapher.TestLossGraph(custom_adapt_pts, budget_list)
optimal_datapoints_list_multipleSampling = grapher.optimalNumDatapointsGraph(custom_adapt_pts, budget_list)
#grapher.TrainTimes(budget_list)

import pdb
pdb.set_trace()
plt.figure(figsize=(5,2))
plt.xlabel('inner loop learning rate')
plt.ylabel('optimal data points \n per task')
#plt.xticks(range(len(budget_list)),['0.001', '0.002', '0.005', '0.01', '0.02', '0.05', '0.1', '0.2'])
plt.grid(which='both', axis='both')
plt.xscale('log')
lr_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
plt.errorbar(lr_list, np.mean(optimal_datapoints_list_multipleSampling,0),np.std(optimal_datapoints_list_multipleSampling,0), marker='o',color = 'k')
plt.savefig(os.path.join(remote_dir,'visualisations/Performance/overlaid/Sinusoid_optimalNumDatapoints_lr.pdf'), bbox_inches='tight', dpi=300)    

plt.figure()
plt.xlabel('budget')
plt.ylabel('training loss (mean)')
plt.grid(which='both', axis='both')
plt.plot(range(len(budget_list)),optimal_train_loss_list,marker='o',color = 'tab:blue')
plt.savefig(os.path.join(remote_dir,'visualisations/Performance/overlaid/Sinusoid_trainLoss_lr.png'), bbox_inches='tight', dpi=300)    
'''


### optimal number of datapoints vs budget
remote_dir = '/allotMeta2/salsa/meta/results/experiment_5_sinusoid'
grapher = GraphGenerator(remote_dir)

#budget_list = [201, 601, 1001, 2000, 4000, 8000, 10000, 20000, 40000, 60000, 80000]
#budget_list = [10002, 10003, 10004, 10000, 10005, 10006, 10007, 10008]
#budget_list = [4,8,16,32,64,201,601,1001,4000,10000,40000, 80000] #for test loss curves
budget_list = [4,8,16,32,64,201,601,1001,4000,6000,8000, 10000, 20000, 30000, 40000, 60000, 80000] # for optimal data points

custom_adapt_pts = [500]
#grapher.TestLossGraph(custom_adapt_pts, budget_list)
optimal_train_loss_list = grapher.TrainLossGraph(custom_adapt_pts, budget_list)
optimal_datapoints_list_multipleSampling = grapher.optimalNumDatapointsGraph(custom_adapt_pts, budget_list)
#grapher.TrainTimes(budget_list)

fig, ax = plt.subplots(figsize=(5,5), nrows=2, ncols=1, sharex=True)
#ax[0].xlabel('budget')
ax[0].set_ylabel('optimal data points \n per task',fontsize=12)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].grid(which='both', axis='both')
ax[0].errorbar(budget_list, np.mean(optimal_datapoints_list_multipleSampling,0),np.std(optimal_datapoints_list_multipleSampling,0), marker='o',color = 'k')
ax[0].axvline(x=8,linestyle='dashed')
#plt.savefig(os.path.join(remote_dir,'visualisations/Performance/overlaid/Sinusoid_optimalNumDatapoints_budget_log.png'), bbox_inches='tight', dpi=300)    
x1,x2,y1,y2 = ax[0].axis()
print(x1,x2,y1,y2)

#plt.figure()
ax[1].set_xlabel('budget',fontsize=12)
ax[1].set_ylabel('training loss (mean)',fontsize=12)
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].grid(which='both', axis='both')
ax[1].plot(budget_list,optimal_train_loss_list,marker='o',color = 'k')
ax[1].axvline(x=8,linestyle='dashed')
x1,x2,y1,y2 = ax[1].axis()
print(x1,x2,y1,y2)

plt.savefig(os.path.join(remote_dir,'visualisations/Performance/overlaid/Sinusoid_optimal_datapoint_trainLoss_budget_log.png'), bbox_inches='tight', dpi=300)    

