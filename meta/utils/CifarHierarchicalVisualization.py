import pickle
import os
import numpy as np
from matplotlib import pyplot as plt

def collate_logs_and_configs(root_folder):
    configs = []
    logs = []
    for budget_folder in os.listdir(root_folder):
        for i, exp in enumerate(os.listdir(os.path.join(root_folder, budget_folder))):
            configs.append(pickle.load(open(os.path.join(root_folder, budget_folder, exp, 'log', 'config.pickle'), 'rb')))
            logs.append(pickle.load(open(os.path.join(root_folder, budget_folder, exp, 'log', 'log.pickle'), 'rb')))


def plot_test_losses(configs, logs, arr_len: int):
    assert len(logs) == len(configs), "Missmatched logs and configs"
    out_arr = np.zeros((arr_len, arr_len))
    counter_arr = np.zeros((arr_len, arr_len))
    for i in range(len(configs)):
        log = logs[i]
        conf = configs[i]
        n_easy = conf['no_of_datapoints_per_easy_tasks']
        n_hard = conf['no_of_datapoints_per_hard_tasks']
        x = int(n_easy / 2 - 1)
        y = int(n_hard / 2 - 1)
        out_arr[x,y] += log['test_accuracy']
        counter_arr[x,y] += 1
    
    assert len(set(counter_arr[np.where(counter_arr != 0)])) == 1, "Different experiments have different numbers of runs."



