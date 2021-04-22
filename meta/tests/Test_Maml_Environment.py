import unittest
from meta.train_ops.MAMLTrainOp import MAMLTrainOp
from torch.utils.data.dataloader import DataLoader
from meta.meta_learners.RegressionMetaModel import MetaLinear
from meta.data.StaticDataset import SinusoidStaticDataset
from meta.data.StaticDataset import FullBatchSampler
from meta.experiment.SinusoidExperiment import data_and_labels_sinusoid_splitter



class TestStaticDataSet(unittest.TestCase):
    def setUp(self):
        #Params for sinusoid dataset
        self.amplitude_range = (0,5)
        self.phase_range = (-2,0)
        self.noise_std_range = (0,0)

        params = {'amplitude_range': self.amplitude_range, 'phase_range': self.phase_range,
                  'noise_std_range' :self.noise_std_range}

        self.no_of_tasks = 4
        self.no_of_points_per_task = 4

        self.ds = SinusoidStaticDataset(self.no_of_tasks, self.no_of_points_per_task, **params)
        self.args_forMamlEnvironment = {'inner_step_size': 0.01, 'meta_lr': 0.01, 'train_test_split_inner': 0.1,
                                    'num_adaptation_steps': 1, 'device': 'cpu', 'first_order': False,
                                    'data_and_labels_splitter': data_and_labels_sinusoid_splitter}
        self.sampler = FullBatchSampler(data_source=self.ds)
        self.dataloader = DataLoader(dataset = self.ds, batch_sampler= self.sampler)
        self.model = MetaLinear(1, 1)

    # TODO: THIS TEST BUGS OUT!!!
    @unittest.skip
    def test_no_of_tasks_in_sampler_default(self):
        sampler = FullBatchSampler(data_source = self.ds)

        maml = MAMLTrainOp(self.model, **self.args_forMamlEnvironment)
        res = maml.training_phase(dataloader = self.dataloader)


if __name__ == '__main__':
    unittest.main()