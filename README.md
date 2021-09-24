## Generative Scene Networks (GSN) - Official PyTorch Implementation
**Unconstrained Scene Generation with Locally Conditioned Radiance Fields, ICCV 2021**<br>
Terrance DeVries, Miguel Angel Bautista, Nitish Srivastava, Graham W. Taylor, Joshua M. Susskind<br>

### [Project Page](https://apple.github.io/ml-gsn/) | [Paper](https://arxiv.org/abs/2104.00670) | [Data](#datasets)

## Requirements
This code was tested with Python 3.6 and CUDA 11.1.1, and uses Pytorch Lightning. A suitable conda environment named `gsn` can be created and activated with:
```
conda env create -f environment.yaml python=3.6
conda activate gsn
```
If you do not already have CUDA installed, you can do so with:
```
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sh cuda_11.1.1_455.32.00_linux.run --toolkit --silent --override
rm cuda_11.1.1_455.32.00_linux.run
```
Custom CUDA kernels may not work with older versions of CUDA. This code will revert to a native PyTorch implementation if the CUDA version is incompatible, although runtime may be ~25% slower.

## Datasets
We provide camera trajectories for two datasets that we used to trained our model: Vizdoom and Replica. These datasets are composed of different sequences with corresponding rgb+depth frames and camera parameters (extrinsiscs and intrinsics).

Dataset | Size | Download Link
--- | :---: | :---:
Vizdoom | 2.4 GB | [download](<https://docs-assets.developer.apple.com/ml-research/datasets/gsn/vizdoom.zip>)
Replica | 11.0 GB | [download](<https://docs-assets.developer.apple.com/ml-research/datasets/gsn/replica.zip>)

Datasets can be downloaded by running the following scripts:  
**VizDoom**<br>
```
python scripts/download_vizdoom.py
```
**Replica**<br>
```
python scripts/download_replica.py
```

## Interactive exploration demo
We provide a [Jupyter notebook](notebooks/walkthrough_demo.ipynb) that allows for interactive exploration of scenes generated from a pre-trained model. Use the WASD keys to freely navigate through the scene! Once you are done, the notebook interpolates the camera path to render a continuous trajectory. Note: You need to download the Replica dataset before via this [script](scripts/download_replica.py) before running the notebook.

Explore scene with WASD to set keypoints | Rendered trajectory
:---: | :---:
<img src="./assets/keyframes.gif" width=256px> | <img src="./assets/camera_trajectory.gif" width=256px>

## Training models
Download the training dataset (if you have not done so already) and begin training with the following commands:  
**VizDoom**<br>
```
bash scripts/launch_gsn_vizdoom_64x64.sh
```

**Replica**<br>
```
bash scripts/launch_gsn_replica_64x64.sh
```

Training takes about 3 days to reach 500k iterations with a batch size of 32 on two A100 GPUs.

## Pre-trained models
We provide pre-trained models for GSN to replicate our experimental results. In particular, we provide models for the Vizdoom dataset trained at 64x64 resolution, and for Replica dataset trained at 64x64 and 128x128. Note that either model can be rendered at higher resolutions than native resolution used durinig training by changing the intrinsic camera parameters during inference.

Dataset | Train Resolution | FID (5k) | Download Link
--- | :---: | :---: | :---: 
Vizdoom | 64x64 | 35.9 | [download](<https://docs-assets.developer.apple.com/ml-research/models/gsn/vizdoom_64x64.ckpt>)
Replica | 64x64 | 41.5 | [download](<https://docs-assets.developer.apple.com/ml-research/models/gsn/replica_64x64.ckpt>)
Replica | 128x128 | 43.4 | [download](<https://docs-assets.developer.apple.com/ml-research/models/gsn/replica_128x128.ckpt>)

### Evaluating pre-trained models
The evaluation script requires the [training set](#datasets) to run. Download it first if you have not yet done so.
Download and run evaluation for pre-trained models with the following commands:  
**VizDoom**<br>
```
bash scripts/eval_vizdoom_64x_64_pretrained.sh
```
**Replica**<br>
```
bash scripts/eval_replica_64x_64_pretrained.sh
```
Running evaluation will compute the FID score and save sample sheets in the log directory.

## Citation
```
@article{devries2021unconstrained,
    title={Unconstrained Scene Generation with Locally Conditioned Radiance Fields},
    author={Terrance DeVries and Miguel Angel Bautista and 
            Nitish Srivastava and Graham W. Taylor and 
            Joshua M. Susskind},
    journal={arXiv},
    year={2021}
}
```
## License
This sample code is released under the [LICENSE](LICENSE) terms.
