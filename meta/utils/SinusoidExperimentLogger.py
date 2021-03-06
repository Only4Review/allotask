from pathlib import Path
import os
from glob import glob
import torch
import pickle
from meta.utils.BaseExperimentLogger import Logger


"""
config_info is a python dict which contains: 
    1.) Budget B
    2.) p = 1/N = K/KN = K/B, where K: no datapoints per task and N: no tasks
    3.) Kin: inner loop batch_size (datapoints batch size)
    4.) Kout: outer loop batch_size (tasks batch size)
    5.) train_test_split
"""
class SinusoidExperimentLogger(Logger):
    def __init__(self, dataset, ID, config_info):
        self.results_folder = "meta/results/experiment_%s_%s/%s_%s_%s_%s_%s_%s" % (ID, 
                                                                           dataset,
                                                                           str(config_info['Budget']),
                                                                           str(round(config_info['p'], 4)),
                                                                           str(config_info['Kin']),
                                                                           str(config_info['Nout']),
                                                                           str(config_info['train_test_split']),
                                                                           str(config_info['run'])
                                                                           )
        self.log_folder = os.path.join(self.results_folder, 'log')
        self.model_folder = os.path.join(self.results_folder, 'metamodel')
    
        self.cache = {}
        
    def update_cache(self, point, name, taskID=None):
        if taskID==None:
            if name in self.cache.keys():
                self.cache[name].append(point)
            else:
                self.cache[name]=[]
                self.cache[name].append(point)
        else:
            if taskID in self.cache[name]:
                self.cache[name][taskID].append(point)
            else:
                self.cache[name][taskID]=[]
                self.cache[name][taskID].append(point)
        
        
    def make_logdir(self):
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        Path(self.log_folder).mkdir(parents=True, exist_ok=True)
        Path(self.model_folder).mkdir(parents=True, exist_ok=True)

    def write(self, data, name='log.pickle'):
        with open(os.path.join(self.log_folder, name), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_model(self, model, checkpoint_index):
        torch.save(model, os.path.join(self.model_folder,"MetaModel_{}.pt".format(checkpoint_index)))
    
    def load_model(self, checkpoint_index='best'):
        if checkpoint_index == 'best':
            model_dirs = glob(os.path.join(self.model_folder,'*.pt'))
            if len(model_dirs)==1:
                model = torch.load(model_dirs[0])
            else:
                greatest_epoch = 0
                for model_dir in model_dirs:
                    filename = os.path.basename(model_dir)
                    checkpoint_epoch = int(filename.split('_')[1].split('.')[0])
                    if checkpoint_epoch > greatest_epoch:
                        greatest_epoch = checkpoint_epoch
                model = torch.load(os.path.join(self.model_folder,"MetaModel_%d.pt" % greatest_epoch))
        
        else:
            model = torch.load(os.path.join(self.model_folder,"MetaModel_%d.pt" % checkpoint_index))
        
        return model
    
    def clean_model_checkpoints(self, checkpoint_to_keep):
        model_dirs = glob(os.path.join(self.model_folder,'*.pt'))
        
        for model_dir in model_dirs:
            if os.path.basename(model_dir) == "MetaModel_%d.pt" % checkpoint_to_keep:
                continue
            else:
                os.remove(model_dir)