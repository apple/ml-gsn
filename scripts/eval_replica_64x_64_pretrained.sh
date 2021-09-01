# download pretrained model
wget https://docs-assets.developer.apple.com/ml-research/models/gsn/replica_64x64.ckpt

# run evaluation
python train_gsn.py \
--evaluate True \
--resume_from_path 'replica_64x64.ckpt' \
data_config.data='replica_all' \
data_config.data_dir='data/replica_all'