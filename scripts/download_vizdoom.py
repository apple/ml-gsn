from torchvision.datasets.utils import download_and_extract_archive

vizdoom_url = 'https://docs-assets.developer.apple.com/ml-research/datasets/gsn/vizdoom.zip'
data_dir = 'data/'
download_and_extract_archive(url=vizdoom_url, download_root=data_dir)