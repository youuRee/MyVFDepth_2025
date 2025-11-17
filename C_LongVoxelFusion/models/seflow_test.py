import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
import hydra, wandb, os, sys
from hydra.core.hydra_config import HydraConfig
from opensf.src.dataset import HDF5Dataset
from opensf.src.trainer import ModelWrapper

checkpoint = '/home/work/user/OpenSceneFlow/pretrained/seflow_best.ckpt'

seflow = ModelWrapper.load_from_checkpoint(checkpoint, eval=True).cuda()
print(seflow)