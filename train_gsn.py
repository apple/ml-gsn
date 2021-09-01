import os
import random

import torch
import torch.backends.cudnn as cudnn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from options.base_config import BaseConfig
from utils.callbacks import GSNVizCallback
from builders.builders import build_dataloader
from utils.utils import instantiate_from_config
from models.model_utils import TrajectorySampler


def main(opt):
    # configure dataset so that each epoch has 1k iterations
    opt.data_config.samples_per_epoch = opt.data_config.batch_size * torch.cuda.device_count() * 1000
    data_module = build_dataloader(opt.data_config)

    # build model
    opt.model_config.params.img_res = opt.data_config.img_res
    gsn = instantiate_from_config(opt.model_config)
    # add config to the model so it can be saved during checkpointing
    gsn.opt = opt

    # get real camera trajectories from dataset to sample during training
    real_Rts = data_module.train_loader.dataset.get_trajectory_Rt()
    trajectory_sampler = TrajectorySampler(real_Rts=real_Rts, mode=opt.model_config.params.trajectory_mode)
    gsn.set_trajectory_sampler(trajectory_sampler=trajectory_sampler)

    if opt.resume_from_path:
        checkpoint = torch.load(opt.resume_from_path)['state_dict']

        # get rid of all the inception params which are leftover from FID metric
        keys_for_deletion = []
        for key in checkpoint.keys():
            if 'fid' in key:
                keys_for_deletion.append(key)

        for key in keys_for_deletion:
            del checkpoint[key]

        gsn.load_state_dict(checkpoint, strict=True)
        print('Resuming from checkpoint at {}'.format(opt.resume_from_path))

    checkpoint_callback = ModelCheckpoint(
        monitor='metrics/fid',
        save_last=True,
        dirpath=os.path.join(opt.log_dir, 'checkpoints'),
        filename='gsn-model-best-fid',
        save_top_k=1,
        mode='min',
    )

    voxel_res = opt.model_config.params.voxel_res
    voxel_size = opt.model_config.params.voxel_size
    viz_callback = GSNVizCallback(opt.log_dir, voxel_res=voxel_res, voxel_size=voxel_size)

    callback_list = [viz_callback, checkpoint_callback]

    logger = TensorBoardLogger(os.path.join(opt.log_dir, 'logs'), name="gsn")

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        callbacks=callback_list,
        accelerator='ddp',
        num_sanity_val_steps=0,
        check_val_every_n_epoch=opt.eval_freq,
        logger=logger,
        precision=opt.precision,
        max_epochs=opt.n_epochs,
        progress_bar_refresh_rate=1,
    )

    if opt.evaluate:
        trainer.validate(
            gsn,
            val_dataloaders=data_module.val_dataloader(),
        )
    else:
        trainer.fit(
            gsn,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )


if __name__ == '__main__':
    opt = BaseConfig().parse()

    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
    cudnn.benchmark = True

    main(opt)
