import unittest
from meta.data.MetaSamplers import *
from meta.data.SinusoidGenerator import SinusoidGenerator
from torch.utils.data.dataloader import DataLoader

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        #Params for sinusoid dataset
        self.amplitude_range = (0,0.001)
        self.phase_range = (1,2)
        self.noise_range = (3,4)
        self.sinus_data_source = SinusoidGenerator(self.amplitude_range, self.phase_range, self.noise_range)
        
        #params for sampler
        self.no_of_tasks = 3
        self.no_of_points_per_task = 2
        self.no_of_batches = 2

    def test_Sampler_With_Sinusoid_Dataset(self):
        ''' 
        An end to end test of data pipeline invloving sinusoid dataset, dataloader and uniform sampler. 
        Checks that the parameters are sampled in the right range, a dataloader can be created and iterated
        from sinusoid ds and sampler
        '''
        sampler = UniformMetaSampler(data_source = self.sinus_data_source, 
                                     no_of_tasks = self.no_of_tasks, 
                                     no_of_points_per_task = self.no_of_points_per_task, 
                                     no_of_batches = self.no_of_batches, 
                                     task_sampler = None, data_from_task_sampler = None)

        for sample in sampler:
            data = next(iter(sample))
            self.assertIsInstance(data, tuple)
            self.assertEqual(len(data), 2)

            task = data[0]
            self.assertIsInstance(task, tuple)
            self.assertEqual(len(task), 3)

            self.assertGreaterEqual(task[0], self.amplitude_range[0])
            self.assertLess(task[0], self.amplitude_range[1])

            self.assertGreaterEqual(task[1], self.phase_range[0])
            self.assertLess(task[1], self.phase_range[1])

            self.assertGreaterEqual(task[2], self.noise_range[0])
            self.assertLess(task[2], self.noise_range[1])
        
        data_sampler = DataLoader(self.sinus_data_source, batch_sampler = sampler)
        for sample in data_sampler:
            sample = sample.numpy()
            self.assertEqual(sample.shape, (self.no_of_tasks * self.no_of_points_per_task, 2))


if __name__ == '__main__':
    unittest.main()