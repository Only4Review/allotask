from torch.utils.data import Sampler
from meta.data.ParametrizationSamplers import UniformIntSampler, UniformSampler
import numpy as np

        
class UniformMetaSampler(Sampler):
    '''
    Sampler intended as parameter batch_sampler for torch.utils.data.dataloader.DataLoader class.
    Generates a uniform iterator for a meta dataset, in the sense that all sampled tasks have the 
    same number of data points inside. 
    '''

    def __init__(self, data_source, no_of_tasks = 1, no_of_points_per_task = 1, no_of_batches = 1, task_sampler = None, data_from_task_sampler = None):
        '''
        Input:
            data_source - a meta dataset (torch.utils.data.DataSet)
            no_of_tasks - number of tasks sampled in a single batch (positive int)
            no_of_points_per_task - number of points sampled in each task (same for all tasks) (positive int)
            no_of_batches - positive int
            task_sampler - a function that receives a positive int and returns the said number of task indices
            data_from_task_sampler - a function that receives a positive int and returns the said number of task indices
            Here, the assumption is that sampling data points from tasks is homogenous among tasks.
        '''
        
        super(UniformMetaSampler, self).__init__(data_source)

        if not (isinstance(no_of_tasks, int) and isinstance(no_of_points_per_task, int) and no_of_tasks > 0 and no_of_points_per_task > 0):
            raise(ValueError('Wrong parameters for UniformMetaSampler constructor: no_of_tasks: {}, no_of_points_per_task: {}'.format(no_of_tasks, no_of_points_per_task)))
        self._data_source = data_source
        self._no_of_tasks = no_of_tasks
        self._no_of_points_per_task = no_of_points_per_task

        if isinstance(no_of_batches, int) and no_of_batches >=1:
            self._no_of_batches = no_of_batches 
        else:
            raise(ValueError('Wrong parameters for UniformMetaSampler constructor: no_of_batches: {}'.format(no_of_tasks)))
        
        if task_sampler is None:
            self._task_sampler = data_source.default_task_sampler
        else: 
            self._task_sampler = task_sampler
            
        if data_from_task_sampler is None:
            self._data_from_task_sampler = data_source.default_data_sampler
        else:
            self._data_from_task_sampler = data_from_task_sampler

    def __iter__(self):
            for i in range(self._no_of_batches):
                task_index_list = self._task_sampler(self._no_of_tasks)
                indices_list = []
                for task in task_index_list:
                    task_data_index_list = self._data_from_task_sampler(self._no_of_points_per_task)
                    for task_data_point in task_data_index_list:
                        indices_list.append((task, task_data_point))
                yield indices_list

class StaticSampler(Sampler):
    '''
    Sampler intended as parameter batch_sampler for torch.utils.data.dataloader.DataLoader class.
    Generates a uniform iterator for a meta dataset, in the sense that all sampled tasks have the 
    same number of data points inside. 
    '''

    def __init__(self, data_source, no_of_tasks = 1, no_of_points_per_task = 1, task_sampler = None, data_from_task_sampler = None):
        '''
        Input:
            data_source - a meta dataset (torch.utils.data.DataSet)
            no_of_tasks - number of tasks sampled in a single batch (positive int)
            no_of_points_per_task - number of points sampled in each task (same for all tasks) (positive int)
            no_of_batches - positive int
            task_sampler - a function that receives a positive int and returns the said number of task indices
            data_from_task_sampler - a function that receives a positive int and returns the said number of task indices
            Here, the assumption is that sampling data points from tasks is homogenous among tasks.
        '''
        
        super(StaticSampler, self).__init__(data_source)

        if not (isinstance(no_of_tasks, int) and isinstance(no_of_points_per_task, int) and no_of_tasks > 0 and no_of_points_per_task > 0):
            raise(ValueError('Wrong parameters for UniformMetaSampler constructor: no_of_tasks: {}, no_of_points_per_task: {}'.format(no_of_tasks, no_of_points_per_task)))
        self._data_source = data_source
        self._no_of_tasks = no_of_tasks
        self._no_of_points_per_task = no_of_points_per_task


        if task_sampler is None:
            self._task_sampler = data_source.default_task_sampler
        else: 
            self._task_sampler = task_sampler
            
        if data_from_task_sampler is None:
            self._data_from_task_sampler = data_source.default_data_sampler
        else:
            self._data_from_task_sampler = data_from_task_sampler
        
        task_index_list = self._task_sampler(self._no_of_tasks)
        self.indices_list = []
        for task in task_index_list:
            task_data_index_list = self._data_from_task_sampler(self._no_of_points_per_task)
            for task_data_point in task_data_index_list:
                self.indices_list.append((task, task_data_point))

            
    def __iter__(self):
        return iter(self.indices_list)

