import numpy as np
from torch.utils.data import Dataset


class SinusoidGenerator(Dataset):
    def __init__(
            self,
            amplitude_range,
            phase_range,
            noise_std_range=(0,1),
            x_range=(-np.pi, np.pi),
            with_caching=False,
            cache_len=100,
            task_cache_len=1000,
            set_state=False,
            seed=1
    ):
        """
        :param amplitude_range: tuple, high to low, both inclusive, if high = low, this will be a constant for each task
        :param phase_range: tuple, high to low, both inclusive, if high = low, this will be a constant for each task
        :param noise_std_range: tuple, high to low, both inclusive, if high = low, this will be a constant for each task
        :param x_range: tuple, high to low, both inclusive
        :param with_caching: bool, whether to cache calls to __getitem__
        :param cache_len: int, how many calls to __getitem__ to cache
        :param task_cache_len: int, how many calls to __getitem__ to cache per task
        :param set_state: bool, if True, creates a random state from the seed
        :param seed: seeds the random state, default 1, unused if set_state is False
        """
        self.amplitude_range = self._tuple_or_float_to_tuple(amplitude_range)
        self.phase_range = self._tuple_or_float_to_tuple(phase_range)
        self.noise_std_range = self._tuple_or_float_to_tuple(noise_std_range)
        self.num_tasks = 0
        self.x_range = x_range
        self.n_samples = 0
        self._with_caching = with_caching
        self.cache_len = cache_len
        self.task_cache_len = task_cache_len

        if with_caching:
            self.static_dataset = ExtendableDataset([])

        if set_state:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

        # Create a prototypical task to access task-related methods. Unused for direct sampling.
        self.default_task = SinusoidGeneratorTask(1, 0, 0, x_range=x_range, with_caching=False, state=self.rng)

    @staticmethod
    def _tuple_or_float_to_tuple(tof):
        if isinstance(tof, tuple):
            assert len(tof) == 2, "Range tuple should have length 2, not {}.".format(len(tof))
        else:
            try:
                tof = float(tof)
                tof = (tof, tof)
            except TypeError:
                raise NotImplementedError('Please pass a number or tuple for the range.')
        return tof

    # TODO: refactor this into a single method, called _sample_task_params
    def default_task_sampler(self, n):
        arr = np.stack([self.rng.uniform(self.amplitude_range[0], self.amplitude_range[1], n),
                        self.rng.uniform(self.phase_range[0], self.phase_range[1], n),
                        self.rng.uniform(self.noise_std_range[0], self.noise_std_range[1], n)], axis=1)
        arr = list(map(tuple, arr))
        return arr

    # TODO: refactor this into a method _sample_task_data_params
    def default_data_sampler(self, n):
        x_val = self.default_task.sample_param(n)
        return x_val

    @property
    def with_caching(self):
        return self._with_caching

    def __len__(self):
        return self.num_tasks

    #TODO: Do we still need this?
    def _sample_task_params(self):
        parametrization = self.default_task_sampler(1)
        return parametrization

    def _validate_parametrization(self, task_sample_parametrization):
        if isinstance(task_sample_parametrization, tuple):
            assert len(task_sample_parametrization) == 2, 'Must give two parametrizations: one for task and one for sample.'
            task_param = task_sample_parametrization[0]
            sample_param = task_sample_parametrization[1]
        else:
            raise NotImplementedError("Indexing method for type {} not supported".format(type(task_sample_parametrization)))

        if isinstance(task_param, tuple):
            assert len(task_param) == 3, 'Must give exactly 3 parameters for task parametrization.'
        else:
            raise NotImplementedError("Task indexing method for type {} not supported".format(type(task_param)))

        self._check_within_range(task_param[0], self.amplitude_range)
        self._check_within_range(task_param[1], self.phase_range)
        self._check_within_range(task_param[2], self.noise_std_range)
        self._check_within_range(sample_param, self.x_range)
        return task_param, sample_param

    def _check_within_range(self, value, interval):
        assert interval[0] <= value <= interval[1]

    def __getitem__(self, task_sample_parametrization):
        task_param, sample_param = self._validate_parametrization(task_sample_parametrization)

        task = SinusoidGeneratorTask(
            amplitude=task_param[0],
            phase=task_param[1],
            noise_std=task_param[2],
            x_range=self.x_range,
            with_caching=self.with_caching,
            cache_len=self.task_cache_len,
            state=self.rng
        )

        if self.with_caching and len(self) < self.cache_len:
            self.static_dataset.extend([task])

        sample = task[sample_param]
        return task_param[0], sample


class SinusoidGeneratorTask(Dataset):
    def __init__(
            self,
            amplitude,
            phase=0,
            noise_std=1,
            x_range=(-np.pi, np.pi),
            with_caching = False,
            cache_len = 1000,
            state = np.random
    ):
        self._amplitude = amplitude
        self._phase = phase
        self._noise_std = noise_std
        self._x_range = x_range
        self.n_samples = 0
        self._with_caching = with_caching
        self.cache_len = cache_len
        if self._with_caching:
            self.static_dataset = ExtendableDataset([])
        self.rng = state


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

    @property
    def with_caching(self):
        return self._with_caching
    
    def sample_param(self, n = 1):
        x_val = self.rng.uniform(self.x_range[0], self.x_range[1], n)
        return x_val.tolist()

    def __getitem__(self, x_val):
        if isinstance(x_val, float):
            pass
        else:
            try:
                x_val = float(x_val)
            except TypeError:
                raise NotImplementedError("Indexing method for type {} not supported".format(type(x_val)))

        y_val = self.amplitude * np.sin(x_val + self.phase) + self.noise_std * self.rng.normal(0,1,1)
        self.n_samples += 1
        out = np.array([x_val, y_val]).astype(np.float32)

        if self.with_caching and len(self) < self.cache_len:
            self.static_dataset.extend([out])
        return out

    def __len__(self):
        return self.n_samples


class ExtendableDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def extend(self, more_data:list):
        self.data += more_data



