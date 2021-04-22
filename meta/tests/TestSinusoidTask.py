import unittest
import numpy as np
from meta.data.SinusoidGenerator import SinusoidGeneratorTask


class TestSinusoidTask(unittest.TestCase):
    def setUp(self):
        self.noiseless_sinusoid_task = SinusoidGeneratorTask(1, 0, 0)
        self.noisy_sinusoid_task = SinusoidGeneratorTask(1, 0, 0.1)
        self.caching_task = SinusoidGeneratorTask(1, 0, 0.1, with_caching=True)

    def test_set_properties_fails(self):
        with self.assertRaises(AttributeError):
            self.noiseless_sinusoid_task.amplitude = 2
        with self.assertRaises(AttributeError):
            self.noiseless_sinusoid_task.phase = 0.5
        with self.assertRaises(AttributeError):
            self.noiseless_sinusoid_task.noise_std = 0.3
        with self.assertRaises(AttributeError):
            self.noiseless_sinusoid_task.with_caching = True

    def test_state_change(self):
        state1 = np.random.RandomState(1)
        state2 = np.random.RandomState(2)
        state3 = np.random.RandomState(1)

        self.noisy_sinusoid_task.rng = state1
        x1 = self.noisy_sinusoid_task.sample_param(1)[0]
        y1 = self.noisy_sinusoid_task[x1]

        self.noisy_sinusoid_task.rng = state2
        x2 = self.noisy_sinusoid_task.sample_param(1)[0]
        y2 = self.noisy_sinusoid_task[x2]

        self.noisy_sinusoid_task.rng = state3
        x3 = self.noisy_sinusoid_task.sample_param(1)[0]
        y3 = self.noisy_sinusoid_task[x3]

        self.assertTrue(np.allclose(y1, y3))
        self.assertFalse(np.allclose(y1, y2))

    def test_noiseless_is_noiseless(self):
        y1 = self.noiseless_sinusoid_task[0.5]
        y2 = self.noiseless_sinusoid_task[0.5]
        self.assertTrue(np.allclose(y1, y2))

    def test_caching_is_extendable(self):
        _ = self.caching_task[0.1]
        self.assertEqual(len(self.caching_task.static_dataset), 1)
        __ = self.caching_task[0.1]
        self.assertEqual(len(self.caching_task.static_dataset), 2)

    def test_caching_is_lazy(self):
        # test sampling does not cache
        _ = self.caching_task[0.1]
        _ = self.caching_task.sample_param(10)
        self.assertEqual(len(self.caching_task.static_dataset), 1)

    def test___getitem___expected_out_type_and_shape(self):
        y1 = self.noiseless_sinusoid_task[0.5]
        y2 = self.noisy_sinusoid_task[0.5]
        self.assertTrue(isinstance(y1, np.ndarray))
        self.assertEqual(y1.size, 2)
        self.assertTrue(isinstance(y2, np.ndarray))
        self.assertEqual(y2.size, 2)

    def test___getitem___expected_out_value(self):
        y1 = self.noiseless_sinusoid_task[np.pi/2]
        self.assertAlmostEqual(y1[1], 1)
        y1 = self.noiseless_sinusoid_task[0]
        self.assertAlmostEqual(y1[1], 0)

    def test___len___is_dynamic(self):
        for i in range(1, 10):
            _ = self.noiseless_sinusoid_task[np.random.uniform(-1,1,1)]
            self.assertEqual(len(self.noiseless_sinusoid_task), i)


