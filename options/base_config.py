import torch
import argparse
from omegaconf import OmegaConf

from utils.utils import str2bool


class BaseConfig:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--log_dir', default='logs', type=str)
        parser.add_argument('--resume_from_path', default='', type=str)
        parser.add_argument('--base_config', type=str, default='configs/models/gsn_base_config.yaml')
        parser.add_argument('--eval_freq', type=int, default=10)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--n_epochs', type=int, default=500)
        parser.add_argument('--precision', type=int, default=32, choices=[32, 16])
        parser.add_argument('--evaluate', type=str2bool, nargs='?', const=True, default=False)

        self.initialized = True
        return parser

    def get_config_from_checkpoint(self, opt):
        if opt.resume_from_path:
            print('Loading config from checkpoint at {}'.format(opt.resume_from_path))
            checkpoint = torch.load(opt.resume_from_path)
            checkpoint_config = checkpoint['opt']
        else:
            checkpoint_config = OmegaConf.create()  # empty config
        return checkpoint_config, opt

    def parse(self, override_config=None, verbose=True):
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        opt, unknown = parser.parse_known_args()

        if override_config is not None:
            # prioritize override_config if it exists
            base_config = OmegaConf.load(override_config.base_config)

            # also override the argparse so that checkpoint paths get loaded properly
            if 'resume_from_path' in override_config.keys():
                opt.resume_from_path = override_config['resume_from_path']
        else:
            # otherwise grab the base config file from cli args, or argparse default
            base_config = OmegaConf.load(opt.base_config)
            override_config = OmegaConf.create()  # make empty config

        # config from previosuly trained model checkpoint
        checkpoint_config, opt = self.get_config_from_checkpoint(opt)

        # config from argparse
        argparse_config = OmegaConf.create(vars(opt))

        # unknown args from command line (will override base config file)
        # should be in dot-list format: https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#from-a-dot-list
        cli_config = OmegaConf.from_dotlist(unknown)

        # configs to the right take priority over configs to the left during merge
        config = OmegaConf.merge(base_config, checkpoint_config, argparse_config, cli_config, override_config)

        if verbose:
            print('')
            print('----------------- Config ---------------\n')
            print(OmegaConf.to_yaml(config))
            print('------------------- End ----------------\n')
            print('')

        return config
