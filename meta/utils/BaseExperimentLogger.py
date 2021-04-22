from abc import ABC, abstractmethod
from pathlib import Path
import os
import torch
import pickle


class Logger(ABC):
    @abstractmethod
    def make_logdir(self):
        pass

    @abstractmethod
    def write(self, logs):
        pass

    @abstractmethod
    def save_model(self, model, checkpoint_index):
        pass



class BaseExperimentLogger(Logger):
    def __init__(self, model_name='model', dataset='dataset', data_file_path='data', id = ''):
        self.results_folder = "results/result_{}_{}_{}".format(model_name, dataset, id)
        self.data_file_name = data_file_path.split("/")[-1]
        self.log_file_name = "log_{}.txt".format(self.data_file_name.split(".")[0])
        self.result_log_path = os.path.join(self.results_folder, self.log_file_name)

    def make_logdir(self):
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)

    def write(self, logs):
        with open(self.result_log_path, "a") as f:
            f.write(logs)

    def save_model(self, model, checkpoint_index):
        torch.save(model, os.path.join(self.results_folder,"model_{}.pt".format(checkpoint_index)))

        with open(self.result_log_path, "a") as f:
            f.write('Model saved\n')

    def dump_data_pickle(self, data):
        pickle.dump(data, open(os.path.join(
            self.results_folder,
            "torch_data_{}.pickle".format(self.data_file_name.split(".")[0])
        ), "wb"))