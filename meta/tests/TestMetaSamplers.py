import unittest
from meta.data.MetaSamplers import UniformMetaSampler
from meta.data.ParametrizationSamplers import SinusoidNaturalParametrizationSampler, UniformSampler
from meta.data.SinusoidGenerator import SinusoidGenerator


class TestUniformMetaSampler(unittest.TestCase):
    def setUp(self):
        # Params for sinusoid dataset
        self.amplitude_range = (0, 0.001)
        self.phase_range = (1, 2)
        self.noise_range = (3, 4)
        self.sinus_data_source = SinusoidGenerator(self.amplitude_range, self.phase_range, self.noise_range)

        # artificial dataset
        self.artificial_data_source = [1, 2, 3]

        # params for sampler
        self.no_of_tasks = 3
        self.no_of_points_per_task = 2
        self.no_of_batches = 2

        self.sampler = UniformMetaSampler(data_source=self.artificial_data_source,
                                          no_of_tasks=self.no_of_tasks,
                                          no_of_points_per_task=self.no_of_points_per_task,
                                          no_of_batches=self.no_of_batches,
                                          task_sampler=SinusoidNaturalParametrizationSampler().sample,
                                          data_from_task_sampler=UniformSampler(0, 1).sample)

    def test_UniformMetaSamplerBatchLength(self):
        '''
        Tests the functionality of meta sampler on artificial data. The test is that it returns iterators
        of expected length for batches.
        '''
        sampler = self.sampler

        iter(sampler)

        count_batches = 0
        for batch in sampler:
            count_batches += 1
        self.assertEqual(count_batches, self.no_of_batches)

    def test_UniformMetaSamplerDataLength(self):
        '''
        Tests the functionality of meta sampler on artificial data. The test is that it returns iterators
        of expected length for tasks and data within tasks.
        '''
        sampler = self.sampler

        iterator = iter(sampler)

        batch = next(iterator)
        count_data = 0
        for data in batch:
            count_data += 1
        self.assertEqual(count_data, self.no_of_points_per_task * self.no_of_tasks)


if __name__ == '__main__':
    unittest.main()