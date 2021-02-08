import unittest
import numpy as np
from meta.data.SinusoidGenerator import SinusoidGenerator


class TestSinusoidTask(unittest.TestCase):
    def setUp(self):
        self.noiseless_phaseless_metadataset = SinusoidGenerator((0,1), 0, 0)
        self.noisy_phaseless_metadataset = SinusoidGenerator((0, 1), 0, (0, 0.5))

    def test_default_samplers_and___getitem___compatible(self):
        task_data_params = zip(
            self.noiseless_phaseless_metadataset.default_task_sampler(10),
            self.noiseless_phaseless_metadataset.default_data_sampler(10)
        )
        for item in task_data_params:
            self.noiseless_phaseless_metadataset[item]

        self.assertTrue(True)

    def test_default_task_sampler_output_type(self):
        params = self.noiseless_phaseless_metadataset.default_task_sampler(10)
        self.assertTrue(isinstance(params, list))
        for item in params:
            self.assertTrue(isinstance(item, tuple))
            self.assertTrue(len(item) == 3)

    def test___getitem___out_type_and_size(self):
        y1 = self.noiseless_phaseless_metadataset[((0,0,0),0)]
        y2 = self.noisy_phaseless_metadataset[((0,0,0),0)]
        self.assertTrue(isinstance(y1, np.ndarray))
        self.assertEqual(y1.size, 2)
        self.assertTrue(isinstance(y2, np.ndarray))
        self.assertEqual(y2.size, 2)

    def test__validate_parameters(self):
        with self.assertRaises(AssertionError):
            y1 = self.noiseless_phaseless_metadataset[((0, 0, 0.1), 0)]