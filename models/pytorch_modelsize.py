import torch
import numpy as np

from logger.terminal_utils import ExperimentArgParse, logout
import models.standard_models as std_models

import pdb

class SizeEstimator(object):

    def __init__(self, args, input_size=(1, 1, 32, 32), bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.args = args
        self.input_size = input_size
        self.bits = bits
        return

    def get_parameter_sizes(self, model):
        '''Get sizes of all parameters in `model`'''

        if self.args.cl_method == "CWR":
            tw_model, cw_model = model
            model_ = cw_model
        else:
            model_ = model

        mods = list(model_.modules())
        buffs = list(model_.named_buffers())
        sizes = []

        for i in range(1, len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        for i in range(len(buffs)):
            n, b = buffs[i]
            sizes.append(np.array(b.size()))

        self.param_sizes = sizes
        return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def estimate_size(self, model):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes(model)
        self.calc_param_bits()
        total = self.param_bits

        total_megabytes = (total / 8) / (1024 ** 2)
        return total_megabytes, total


if __name__ == "__main__":
    exp_parser = ExperimentArgParse("Model memory size test")
    exp_args = exp_parser.parse()
    exp_args.device = torch.device('cpu')

    model = std_models.TransE(40943, 11, exp_args.hidden_size, exp_args.margin,
                              exp_args.neg_ratio, exp_args.batch_size, exp_args.device)

    se = SizeEstimator(model)
    print(se.estimate_size())
