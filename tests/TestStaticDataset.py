import unittest
from meta.data.StaticDataset import *
from torch.utils.data.dataloader import DataLoader

class TestStaticDataSet(unittest.TestCase):
    def setUp(self):
        #Params for sinusoid dataset
        self.amplitude_range = (0,5)
        self.phase_range = (-2,0)
        self.noise_std_range = (0,0)

        params = {'amplitude_range': self.amplitude_range, 'phase_range': self.phase_range, 'noise_std_range' :self.noise_std_range}

        self.no_of_tasks = 4
        self.no_of_points_per_task = 4

        self.ds = SinusoidStaticDataset(self.no_of_tasks, self.no_of_points_per_task, **params)
                
    def test_no_of_tasks_in_sampler_default(self):
        sampler = FullBatchSampler(data_source = self.ds)
        no_of_tasks = 0
        no_of_points_per_task = 0
        
        for task in sampler:
            no_of_tasks += 1

        self.assertEqual(self.no_of_tasks, no_of_tasks)

    
    def test_no_of_tasks_in_sampler_specified(self):
        specified_no_of_tasks = 2
        sampler = FullBatchSampler(data_source = self.ds, no_of_tasks = specified_no_of_tasks)
        
        no_of_tasks = 0
        for task in sampler:
            no_of_tasks += 1

        self.assertEqual(specified_no_of_tasks, no_of_tasks)        

    def test_no_data_points_per_task_in_sampler_defaut(self):  
        sampler = FullBatchSampler(data_source = self.ds)
        no_of_points_per_task = 0
        
        sample = next(iter(sampler))
        for data_point in sample:
            no_of_points_per_task += 1
        
        self.assertEqual(self.no_of_points_per_task, no_of_points_per_task)  

    def test_no_data_points_per_task_in_sampler_specified(self):  
        specified_no_of_points_per_task = 2
        sampler = FullBatchSampler(data_source = self.ds, no_of_points_per_task = specified_no_of_points_per_task)
        
        no_of_points_per_task = 0
        sample = next(iter(sampler))
        for data_point in sample:
            no_of_points_per_task += 1
        
        self.assertEqual(specified_no_of_points_per_task, no_of_points_per_task)

    def test_data_pipeline(self):
        sampler = FullBatchSampler(data_source = self.ds)

        for sample in sampler:
            data = next(iter(sample))
            self.assertIsInstance(data, tuple)
            self.assertEqual(len(data), 2)

            task = self.ds.task_parametrization_array[data[0]]
            self.assertEqual(len(task), 3)

            self.assertGreaterEqual(task[0], self.amplitude_range[0])
            self.assertLess(task[0], self.amplitude_range[1])

            self.assertGreaterEqual(task[1], self.phase_range[0])
            self.assertLess(task[1], self.phase_range[1])

            self.assertGreaterEqual(task[2], self.noise_std_range[0])
            self.assertLessEqual(task[2], self.noise_std_range[1])

        data_sampler = DataLoader(self.ds, batch_sampler = sampler)

        for idx, sample in enumerate(data_sampler):
            sample = sample.numpy()
            self.assertEqual(sample.shape, (self.no_of_points_per_task, 2))
        self.assertEqual(self.no_of_tasks, idx + 1)

        tasks_to_add = 3
        no_of_points_per_task = self.no_of_points_per_task

        size_before = len(self.ds)
        self.ds.add_new_tasks(tasks_to_add, no_of_points_per_task)
        for idx, sample in enumerate(data_sampler):
            sample = sample.numpy()
            self.assertEqual(sample.shape, (self.no_of_points_per_task, 2))
        self.assertEqual(self.no_of_tasks + tasks_to_add, idx + 1)

    def test_adding_data_points_to_tasks(self):
        data_per_tasks_to_add = 3
        self.ds.add_data_per_task(data_per_tasks_to_add)

        for task in self.ds.task_dataset_array:
            self.assertEqual(len(task), data_per_tasks_to_add + self.no_of_points_per_task)
    
    def test_adding_new_tasks(self):
        tasks_to_add = 1
        no_of_points_per_task = 2
        size_before = len(self.ds)

        self.ds.add_new_tasks(tasks_to_add, no_of_points_per_task)

        self.assertEqual(len(self.ds), size_before + tasks_to_add)

        for i in range(size_before):
            self.assertEqual(len(self.ds.task_dataset_array[i]), self.no_of_points_per_task)
        
        i+=1
        for j in range(tasks_to_add):
            self.assertEqual(len(self.ds.task_dataset_array[i+j]), no_of_points_per_task)
        
    
    def test_increasing_tasks_and_data_points(self):
        tasks_to_add = 2
        data_to_add = 2

        size_before = len(self.ds)

        self.ds.increase_tasks_and_data_per_task(tasks_to_add, 0)
        self.assertEqual(len(self.ds), size_before + tasks_to_add)

        size_before += tasks_to_add
        self.ds.increase_tasks_and_data_per_task(0, 2)
        self.assertEqual(len(self.ds), size_before)

        for task in self.ds.task_dataset_array:
            self.assertEqual(len(task), self.no_of_points_per_task + data_to_add)
        
        self.ds.increase_tasks_and_data_per_task(tasks_to_add, data_to_add)
        self.assertEqual(len(self.ds), size_before + tasks_to_add)
        for task in self.ds.task_dataset_array:
            self.assertEqual(len(task), self.no_of_points_per_task + 2*data_to_add)
        





if __name__ == '__main__':
    unittest.main()