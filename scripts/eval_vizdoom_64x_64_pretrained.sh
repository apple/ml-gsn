# download pretrained model
wget https://docs-assets.developer.apple.com/ml-research/models/gsn/vizdoom_64x64.ckpt

# run evaluation
python train_gsn.py \
--evaluate True \
--resume_from_path 'vizdoom_64x64.ckpt' \
data_config.data='vizdoom' \
data_config.data_dir='data/vizdoom_data_iss'