"""Utility module for downloading pre-trained checkpoints."""

import omegaconf
import os
import urllib.request

from .data import DatasetConfiguration, ZincDatasetConfiguration, OGBDatasetConfiguration


PRETRAINED_CHECKPOINTS_ROOT = 'https://users.flatironinstitute.org/~wzhou/autobahn/checkpoints/'


def download_pretrained_checkpoint(data_config: DatasetConfiguration, cache_folder: os.PathLike) -> str:
    """Utility function to download a pre-trained checkpoint from the original repository.

    Parameters
    ----------
    data_config : DatasetConfiguration
        The configuration identifying the dataset for which to download the pre-trained checkpoint.
    cache_folder : os.PathLike
        Path to folder where to cache the downloaded checkpoint.

    Returns
    -------
    str
        Path to downloaded or existing checkpoint.
    """
    if omegaconf.OmegaConf.is_config(data_config):
        config_type = omegaconf.OmegaConf.get_type(data_config)
    else:
        config_type = type(data_config)

    if issubclass(config_type, ZincDatasetConfiguration):
        if data_config.use_subset:
            name = 'zinc_subset'
        else:
            name = 'zinc_full'
    elif issubclass(config_type, OGBDatasetConfiguration):
        name = data_config.data_name.split('-')[1]
    else:
        raise ValueError('Unknown dataset configuration, cannot find pre-trained weights.')

    checkpoint_name = name + '.ckpt'
    checkpoint_path = os.path.join(cache_folder, checkpoint_name)

    if os.path.exists(checkpoint_path):
        print('Found cached checkpoint at location {}'.format(checkpoint_path))
        return checkpoint_path

    print('Local checkpoint not found! Downloading pre-trained weights and saving them at {}'.format(checkpoint_path))
    checkpoint_url = PRETRAINED_CHECKPOINTS_ROOT + checkpoint_name

    os.makedirs(cache_folder, exist_ok=True)
    urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
    return checkpoint_path
