import bisect
import datetime
import dataclasses
import os
import random
import warnings

from typing import Dict, List, Optional, Sequence

import hydra.utils
import pytorch_lightning
import pytorch_lightning.callbacks
import torch
from torch.utils.tensorboard import SummaryWriter


def make_folder_in_sequence(root: str, name: str, min_sequence_digits: int=2, max_attempts=1000) -> str:  # noqa E252
    """Creates a folder with the given name in the given root directory, adding
    a sequence number if the name already exists.

    Parameters
    ----------
    root
        Root directory in which to create the folder
    name
        Name of the folder to create
    min_sequence_digits
        Optional integer representing the number of digits to include in the
        sequence number.

    Returns
    -------
    str
        The full path to the created folder.
    """
    base_dirname = os.path.join(root, name)

    if not os.path.exists(base_dirname):
        os.makedirs(base_dirname, exist_ok=True)
        return base_dirname

    digit_format = '_{{:0{}d}}'.format(min_sequence_digits)

    for i in range(1, max_attempts):
        dirname = base_dirname + digit_format.format(i)

        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            return dirname

    raise ValueError('Could not create folder after {} attempts.'.format(max_attempts))


@dataclasses.dataclass
class TrainerConfiguration:
    batch_size: int = 64
    seed: Optional[int] = None
    output_folder: Optional[str] = None
    num_gpus: int = 1
    max_epochs: int = 100
    mixed_precision: bool = False



def ensure_config_defaults(config: TrainerConfiguration):
    """Ensure default values of the configuration are set, and fixes path to data_folder if necessary.
    """
    if config.seed is None:
        config.seed = random.randint(0, 2 ** 32 - 1)

    if config.output_folder is None:
        config.output_folder = os.path.abspath(make_folder_in_sequence(os.getcwd(), 'run'))

    if hasattr(config, 'data'):
        config.data.data_folder = hydra.utils.to_absolute_path(config.data.data_folder)


def make_trainer(config: TrainerConfiguration, monitor_loss='val_loss', monitor_mode='min'):
    """Creates a trainer according to the given configuration.

    Parameters
    ----------
    config : TrainerConfiguration
        Configuration for the trainer.
    monitor_loss : str
        The loss to monitor for checkpoint saving.

    Returns
    -------
    pytorch_lightning.Trainer
        Trainer class with the given configuration.
    """
    ensure_config_defaults(config)

    callbacks = [
        pytorch_lightning.callbacks.GPUStatsMonitor(intra_step_time=True),
        pytorch_lightning.callbacks.LearningRateMonitor(),
        pytorch_lightning.callbacks.ModelCheckpoint(monitor=monitor_loss, mode=monitor_mode),
        #AutogradProfilerCallback(profile_idx=10),
    ]

    kwargs = {}

    if config.optim.gradient_clip_norm is not None:
        kwargs['gradient_clip_val'] = config.optim.gradient_clip_norm

    if config.mixed_precision:
        kwargs['precision'] = 16

    if config.num_gpus > 1:
        kwargs['accelerator'] = 'ddp'

    trainer = pytorch_lightning.Trainer(
        gpus=config.num_gpus,
        default_root_dir=config.output_folder,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        progress_bar_refresh_rate=5,
        **kwargs)

    return trainer


def list_checkpoints(path: str) -> Dict[str, List[str]]:
    """Lists best checkpoint by version in the given folder.

    Parameters
    ----------
    path : str
        Path to output folder of pytorch-lightning training run.
    """
    result = {}

    base_path = os.path.join(path, 'lightning_logs')
    versions = os.listdir(base_path)

    for version in versions:
        ckpt_folder_path = os.path.join(base_path, version, 'checkpoints')

        last_ckpt_path = os.path.join(ckpt_folder_path, 'last.ckpt')

        if os.path.exists(last_ckpt_path):
            ckpt = torch.load(last_ckpt_path, map_location='cpu')
            model_callback = pytorch_lightning.callbacks.ModelCheckpoint

            if model_callback in ckpt['callbacks']:
                ckpt_path = ckpt['callbacks'][model_callback]['best_model_path']
            else:
                ckpt_path = last_ckpt_path
        else:
            ckpts = os.listdir(ckpt_folder_path)
            ckpts.sort(reverse=True)
            ckpt_path = os.path.join(ckpt_folder_path, ckpts[0])
        result[version] = ckpt_path

    return result



def warmup_decay_lr(epoch: float, milestones: Sequence[float]=None, warmup_epochs: float=5, gamma: float=0.1):
    """Return a learning rate multiplier for a learning rate schedule with warmup and step decay.

    Parameters
    ----------
    epoch : Number
        The current epoch.
    milestones : Sequence[Number]
        A sequence of milestone epochs at which the learning rate is multiplied by a factor `gamma`.
    warmup_epochs : Number
        The length of the warmup in epochs.
    gamma : float
        The amount by which to multiply the learning rate by at each milestone.

    Returns
    -------
    float
        The learning rate factor to apply.
    """
    if milestones is None:
        milestones = []

    num_steps = bisect.bisect(milestones, epoch)

    return (gamma ** num_steps) * min(1.0, (epoch + 1) / (warmup_epochs + 1))


class WarmupStepScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Scheduler which implements a warmup and milestone decay scheduling policy."""
    def __init__(self, optimizer, milestones: Sequence[float]=None, warmup_epochs: float=5, gamma: float=0.1, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.milestones = milestones
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        lr_factor = warmup_decay_lr(self.last_epoch, self.milestones, self.warmup_epochs, self.gamma)
        return [lr_factor * base_lr for base_lr in self.base_lrs]



def initialize_TB_logging(root_dir='./', logging_path=None):
    if logging_path is None:
        basename = "logdir"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        logging_path = "_".join([basename, suffix])
    full_path = root_dir + logging_path
    writer = SummaryWriter(full_path)
    return writer


class AutogradProfilerCallback(pytorch_lightning.Callback):
    """Pytorch-lightning callback used for profiling at a given step."""
    def __init__(self, enabled: bool=True, use_cuda: bool=True, profile_idx: int=20):
        """Creates a new profiler callback.

        Parameters
        ----------
        enabled : bool
            Indicates whether the profiler is enabled. If False, this callback will be a no-op.
        use_cuda : bool
            Whether to profile cuda ops.
        profile_idx : int
            The step at which to profile.
        """
        self._enabled = enabled
        self._use_cuda = use_cuda
        self._profiler = None
        self._profile_idx = profile_idx
        self._profile_done = False


    def on_train_batch_start(self, trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule, batch, batch_idx, dataloader_idx):
        if not self._profile_done and self._profile_idx in (batch_idx, batch_idx + 1):
            pl_module.print('Profiling training batch at step {}'.format(batch_idx))
            self._profiler = torch.autograd.profiler.profile(
                enabled=self._enabled, use_cuda=self._use_cuda)
            self._profiler.__enter__()

    def on_train_batch_end(self, trainer: pytorch_lightning.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self._profile_done and self._profile_idx in (batch_idx, batch_idx + 1):
            self._profiler.__exit__(None, None, None)
            if self._profile_idx == batch_idx:
                self._profiler.export_chrome_trace(os.path.join(trainer.default_root_dir, 'profile.trace'))
                self._profile_done = True
