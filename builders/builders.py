from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.vizdoom import VizdoomDataset
from datasets.replica import ReplicaDataset
import pytorch_lightning as pl


def build_dataloader(data_config, verbose=True):
    datasets = {
        'vizdoom': VizdoomDataset,
        'replica_all': ReplicaDataset,
    }

    if 'train_set_config' not in data_config.keys():
        data_config.train_set_config = {}

    if 'val_set_config' not in data_config.keys():
        data_config.val_set_config = {}

    data_loader_args = ['batch_size', 'shuffle', 'num_workers', 'drop_last', 'pin_memory']
    train_loader_defaults = {'shuffle': True, 'num_workers': 1, 'drop_last': True, 'pin_memory': True}
    val_loader_defaults = {'shuffle': False, 'num_workers': 1, 'drop_last': False, 'pin_memory': True}

    # combine all configs (configs on right have priority if configs share args)
    train_set_config = {**train_loader_defaults, **data_config, **data_config.train_set_config}
    train_set_config = {
        k: train_set_config[k] for k in train_set_config if k not in {'train_set_config', 'val_set_config'}
    }
    train_set = datasets[data_config.dataset](**train_set_config)
    # get only the args for the DataLoader class, since it can't deal with extra args
    train_loader_config = {k: train_set_config[k] for k in data_loader_args}
    train_loader = DataLoader(dataset=train_set, **train_loader_config)

    val_set_config = {**val_loader_defaults, **data_config, **data_config.val_set_config}
    val_set_config = {k: val_set_config[k] for k in val_set_config if k not in {'train_set_config', 'val_set_config'}}
    val_set = datasets[data_config.dataset](**val_set_config)
    val_loader_config = {k: val_set_config[k] for k in data_loader_args}
    val_loader = DataLoader(dataset=val_set, **val_loader_config)

    if verbose:
        print('')
        print('----------- Train Set Config -----------\n')
        print(OmegaConf.to_yaml(train_set_config))
        print('------------ Val Set Config ------------\n')
        print(OmegaConf.to_yaml(val_set_config))
        print('----------------- End ------------------\n')
        print('')

    data_module = DataModule(train_loader=train_loader, val_loader=val_loader)
    return data_module


class DataModule(pl.LightningDataModule):
    def __init__(self, train_loader, val_loader=None, test_loader=None):
        super().__init__()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
