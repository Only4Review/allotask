import unittest
from argparse import Namespace
import pickle
from meta.train_ops.MAMLTrainOp import MAMLTrainOp
from meta.data.StaticDataset import SinusoidStaticDataset
from meta.meta_learners.RegressionMetaModel import MetaModel
import torch
import numpy as np
import json
from meta.experiment.SinusoidExperiment import SinusoidExperiment


class MAMLRegressionTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        args = pickle.load(open('meta/tests/MAML_default_args.pickle', 'rb'))
        args = Namespace(**args)
        args.data_and_labels_splitter = SinusoidStaticDataset.data_and_labels_sinusoid_splitter
        args.device = 'cpu'

        dataset_params = {
            'amplitude_range': (0.1, 5),
            'phase_range': (0, np.pi),
            'noise_std_range': (0, 0),
            'x_range': (-5, 5)
        }
        model = MetaModel()

        with open('meta/tests/default_config.json') as f:
            configurations = json.load(f)
        config = configurations[0]
        updated_args = SinusoidExperiment.update_args_from_config(args, config)
        train_op = MAMLTrainOp(model, updated_args)
        self.exp = SinusoidExperiment(updated_args, config, dataset_params, train_op)
                                      
        # TODO: make logging optional
        self.exp.setup_logs()

    def test_functionality_as_expected(self):
        expected = 0.9940322041511536
        self.exp.run()
        actual = self.exp.evaluate(adapt_pts=100)
        self.assertTrue(np.abs(actual-expected) <= 0.001)
