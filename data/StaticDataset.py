from torch.utils.data.dataset import Dataset
import numpy as np
from torch.utils.data import Sampler


class StaticMetaDataset(Dataset):
    ''' 
    Intended as the base class for Static Meta Datasets. Assumes a two-tuple indexing form 
    where the first coordinate is a task index - in this case, an integer, and the second one - the index of 
    data point within task - this might not be an integer and is implementation dependant.
    Input:
        task_parametrization_array: an array of the true task parameters (effectively a mapping between an 
                                    an integer - the array index and the true task parametrization).
        task_dataset_array: an array where each cell is a data set corresponding to the task indexed by the 
            same integer as in task_parametrization array.  
        no_of_tasks: integer, number of task in dataset.
    '''
    ###################################
    def __init__(self):
        self.no_of_tasks = 0 
        self.task_parametrization_array = None
        self.task_dataset_array = None
    
    ####################################
    def generate_task_parametrizations(self, no_of_tasks, **params):
        '''
        Procedure that generates task parameters for a particular implementation. 
        Input: 
            no_of_tasks: positive int.
            **params: any additional params needed for task generation in particular implementation.
        '''
        raise(NotImplementedError)

    #####################################   
    def get_task_length(self, task_index):
        '''
        Returns the length of the dataset associated with a particular task.
        Input:
            task_index: positive integer.
        Output:
            non-negative integer.
        ''' 
        return len(self.task_dataset_array[task_index])

    #####################################
    def __getitem__(self, index):
        assert(len(index)==2)
        task_param = index[0]
        sample_param = index[1]
        return self.task_dataset_array[task_param][sample_param]
    
    #####################################
    def __len__(self):
        '''
        Returns the number of tasks in meta dataset
        '''
        return len(self.task_parametrization_array)
    
    #####################################
    def get_task_param(self, task_index):
        '''
        Returns parameters of the task associated with a given integer index.
        '''
        return self.task_array[task_index]


#########################################################################
class SinusoidStaticDataset(StaticMetaDataset):
    '''
    Implementation of a static Sinusoid Dataset.
    '''
    def __init__(self, no_of_tasks, no_of_data_points_per_task, **params):
        '''
        no_of_tasks: positive integer - initial number of tasks to sample.
        no_of_data_points_per_task - positive int.
        '''
        super(SinusoidStaticDataset, self).__init__()
        self.amplitude_range = params['amplitude_range']
        self.phase_range = params['phase_range']
        self.noise_std_range = params['noise_std_range']
        self.x_range = params['x_range']
        self.task_parametrization_array = self.generate_task_parametrizations(no_of_tasks)
        self.no_of_tasks = no_of_tasks
        self.task_dataset_array = self.generate_task_datasets(self.task_parametrization_array, no_of_data_points_per_task)
    
    #############################################################################
    def generate_task_datasets(self, task_parametrization_array, no_of_points_per_task):
        '''
        Input:
            task_parametrization_array: an array of true task parameters.
            no_of_points_per_task: pos. int
        Output:
            array of SinusoidStaticTask datasets, corresponding to the input parameters.
        '''
        task_dataset_array = np.empty(len(task_parametrization_array), dtype = SinusoidStaticTask)

        for i, task in enumerate(task_parametrization_array):
            task_dataset_array[i] = SinusoidStaticTask(task[0], task[1], task[2], self.x_range, no_of_points_per_task)
        return task_dataset_array

    ############################################################################    
    def add_new_tasks(self, no_of_tasks, no_of_points_per_task):
        '''Adds additional tasks to MetaDataset.
        Input: 
            no_of_tasks: pos. int
            no_of_poinmts_per_task: pos. int
        Output:
            None. Adds to the class variables.
        '''
        new_tasks = self.generate_task_parametrizations(no_of_tasks)
        new_task_dataset_array = self.generate_task_datasets(new_tasks, no_of_points_per_task)

        self.task_parametrization_array =  np.concatenate([self.task_parametrization_array, new_tasks], axis = 0)
        self.task_dataset_array = np.concatenate([self.task_dataset_array,new_task_dataset_array], axis = 0)
        
    ###########################################################################        
    def generate_task_parametrizations(self, no_of_tasks):
        '''
        Implementation of generate task parametrizations in base class
        '''
        arr = np.stack ([np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], no_of_tasks),
                    np.random.uniform(self.phase_range[0], self.phase_range[1], no_of_tasks),
                    np.random.uniform(self.noise_std_range[0], self.noise_std_range[1], no_of_tasks)], axis = 1)
        return arr
    
    ##########################################################################
    def add_data_per_task(self, no_of_data_points_per_task):
        '''
        Adds data points to all tasks in the meta dataset
        Input: pos. int
        '''
        new_task_data = self.get_task_length(task_index = 0)
        for task in self.task_dataset_array:
            task.add_data(no_of_data_points_per_task)

    ##########################################################################
    def increase_tasks_and_data_per_task(self, task_increase, data_increase):
        '''
        Increases the number of tasks and data points per task by specified number. 
        For newly created tasks will create the same number of data points as in first task 
        in the existing meta-dataset. And then add additional data points to all tasks - new and old.
        '''
        if data_increase > 0:
            self.add_data_per_task(data_increase)
        data_for_new_tasks = self.get_task_length(task_index = 0)
        if task_increase > 0:
            self.add_new_tasks(task_increase, data_for_new_tasks)

###############################################################################
class SinusoidStaticTask(Dataset):
    def __init__(
            self,
            amplitude,
            phase=0,
            noise_std=1,
            x_range=(-2*np.pi, 2*np.pi),
            no_of_samples = 0
    ):
        self._amplitude = amplitude
        self._phase = phase
        self._noise_std = noise_std
        self._x_range = x_range

        if no_of_samples > 0:
            self.data = self.sample_param(no_of_samples)
       
    @property
    def amplitude(self):
        return self._amplitude

    @property
    def phase(self):
        return self._phase

    @property
    def noise_std(self):
        return self._noise_std

    @property
    def x_range(self):
        return self._x_range
    
    def sample_param(self, n = 1):
        x_val = np.random.uniform(self.x_range[0], self.x_range[1], n)
        y_val = np.empty_like(x_val)
        for i,x in enumerate(x_val):
            y_val[i] = self.amplitude * np.sin(x + self.phase) + self.noise_std * np.random.normal(0,1,1) 
        return np.stack([x_val, y_val], axis = 1).astype(np.float32)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]
    
    def add_data(self, no_of_samples):
        new_data = self.sample_param(no_of_samples)
        self.data = np.concatenate([self.data, new_data], axis = 0)

######################################################################
class FullBatchSampler(Sampler):
    '''
    Sampler that samples data from static dataset as follows:
    A sample will contain a given number of tasks (specified at sampler creation), or all tasks if 
    not specified and will sample a given number of data points from a task
    or all data points from a task if data points per task was not specified. 
    This is a batch sampler, and each batch will contain data for a single! task. In other words,
    iterating over this sampler will yield all data corresponding to one task in a single iteration 
    until the whole sample data is exhausted. 
    '''
    ###################################################################
    def __init__(self, data_source, no_of_tasks = None, no_of_points_per_task = None):
        super(FullBatchSampler, self).__init__(data_source)
        self._data_source = data_source
        self._no_of_tasks = no_of_tasks
        self._no_of_data_points_per_task = no_of_points_per_task

    ###################################################################       
    def __iter__(self):
        if self._no_of_tasks is None:
            no_of_tasks = len(self._data_source)
        else:
            no_of_tasks = self._no_of_tasks
            
        task_indices = np.random.choice(a = len(self._data_source), size = no_of_tasks, replace = False)
        for task in task_indices:
            indices_list = []
            task_length = self._data_source.get_task_length(task)
            
            if self._no_of_data_points_per_task is None:
                no_of_points = task_length
            else:
                no_of_points = self._no_of_data_points_per_task
            
            task_data_index_list = np.random.choice(a = task_length, size = no_of_points, replace = False)
            for task_data_point in task_data_index_list:
                indices_list.append((task, task_data_point))
            
            yield indices_list





            


