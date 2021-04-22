# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:26:50 2020

@author: xxx
"""

from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear)
import torch.nn as nn


#MetaModel class inherits from MetaModule class which inherits from nn.Module
#We construct the metamodel using the torchmeta module to make it compatible with the torchmeta library
#Different metamodel architectures should be used using the MetaModules found in torchmeta.modules
#For example, dont use Conv2D! Use MetaConv2d! For more information: https://tristandeleu.github.io/pytorch-meta/

class MetaModel(MetaModule):
    def __init__(self):
        super(MetaModel, self).__init__()
        self.input_size = 1
        
        self.features = MetaSequential(
            MetaLinear(1, 40),
            nn.ReLU(),
            MetaLinear(40, 40),
            nn.ReLU(),
            MetaLinear(40,1)
            )


    def forward(self, x, params=None):
        x = self.features(x, params = self.get_subdict(params, 'features'))
        return x
