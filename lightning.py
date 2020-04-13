import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from apex import amp

from adamw import AdamW
from losses import dice_round, ComboLoss

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from zoo.models import SeResNext50_Unet_Loc

from imgaug import augmenters as iaa

from utils import *

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import gc

from pathlib import Path
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

import boto3


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def dice(im1, im2, empty_score=1.0):

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = im1 & im2

    return 2. * intersection.sum() / im_sum


class XviewDataset(Dataset):
    def __init__(self, path, train=True):
        super().__init__()
        self.train = train
        self.input_shape = (512, 512)
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        all_files = list(Path(path).rglob(pattern=f'*pre_*.json'))
        train_files, val_files = train_test_split(all_files, test_size=0.1)
        self.data_files = train_files if train else val_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        return self._train_item(idx) if self.train else self._val_item(idx)

    def _train_item(self, idx):

        fn = str(self.data_files[idx]).replace('/labels/', '/images/').replace('json', 'png')

        img = cv2.imread(fn, cv2.IMREAD_COLOR)

        if random.random() > 0.985:
            img = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)

        if random.random() > 0.5:
            img = img[::-1, ...]
            msk0 = msk0[::-1, ...]

        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                msk0 = np.rot90(msk0, k=rot)

        if random.random() > 0.9:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)
            
        if random.random() > 0.9:
            rot_pnt =  (img.shape[0] // 2 + random.randint(-320, 320), img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)

        crop_size = self.input_shape[0]
        if random.random() > 0.3:
            crop_size = random.randint(int(self.input_shape[0] / 1.1), int(self.input_shape[0] / 0.9))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 5)
        for i in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk0[y0:y0+crop_size, x0:x0+crop_size].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]

        if crop_size != self.input_shape[0]:
            img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, self.input_shape, interpolation=cv2.INTER_LINEAR)

        if random.random() > 0.99:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.99:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.99:
            if random.random() > 0.99:
                img = clahe(img)
            elif random.random() > 0.99:
                img = gauss_noise(img)
            elif random.random() > 0.99:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.99:
            if random.random() > 0.99:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.99:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.99:
                img = contrast(img, 0.9 + random.random() * 0.2)
                
        if random.random() > 0.999:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        msk = msk0[..., np.newaxis]

        msk = (msk > 127) * 1

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'fn': fn}
        return sample

    def _val_item(self, idx):
        fn = str(self.data_files[idx]).replace('/labels/', '/images/').replace('json', 'png')

        img = cv2.imread(fn, cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)
        
        msk = msk0[..., np.newaxis]

        msk = (msk > 127) * 1

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'fn': fn}
        return sample


class SeResNext50Lightning(SeResNext50_Unet_Loc):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super(SeResNext50Lightning, self).__init__()

    def prepare_data(self):
        self.train_dataset = XviewDataset('/newvolume/xview/data', train=True)
        self.val_dataset = XviewDataset('/newvolume/xview/data', train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=5, shuffle=True, pin_memory=False, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=5, shuffle=False, pin_memory=False)

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = AdamW(params, lr=0.00015, weight_decay=1e-6)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 29, 43, 53, 65, 80, 90, 100, 110, 130, 150, 170, 180, 190], gamma=0.5)
        return [optimizer], [scheduler]

    def seg_loss(self, y_hat, y):
        return ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=False).cuda()(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        msks = batch["msk"]
        out = self.forward(imgs)
        loss = self.seg_loss(out, msks)

        with torch.no_grad():
            _probs = torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, msks[:, 0, ...])

        metrics = {
            'loss': loss.mean().data.cpu().numpy(),
            'dice_sc': dice_sc.mean().data.cpu().numpy()
        }
        self.logger.log_metrics(metrics, step=batch_idx)
        return {'loss': loss, 'dice_sc': dice_sc}

    def on_train_end(self):
        status = 'Interrupted' if self.interrupted else 'Completed'
        self.logger.finalize(status)
        # self.logger.experiment.send_notification(self, "xView Completed", status=status)
        super().on_train_end()
        if not self.interrupted:
            ec2 = boto3.client('ec2')
            instance_id = os.environ['INSTANCE_ID']
            ec2.stop_instances(InstanceIds=[instance_id], DryRun=True)
        return None

    def validation_step(self, batch, batch_idx):
        dices0 = []
        _thr = 0.5
        imgs = batch["img"]
        msks = batch["msk"]
        out = self.forward(imgs)
        
        _probs = torch.sigmoid(out[:, 0, ...])
        loss = dice_round(_probs, msks[:, 0, ...])
        seg_loss = self.seg_loss(out, msks)

        for j in range(msks.shape[0]):
            dices0.append(dice(msks[j, 0].bool(), _probs[j] > _thr))

        d0 = torch.Tensor(dices0).type_as(seg_loss)

        metrics = {
            'val_loss': loss.mean().data.cpu().numpy(),
            'seg_loss': seg_loss.mean().data.cpu().numpy(),
            'accuracy': d0.mean().data.cpu().numpy()
        }

        self.logger.log_metrics(metrics, step=batch_idx)

        # self.logger.experiment.log_metric('val_loss', loss.mean().data.cpu().numpy(), step=batch_idx, epoch=self.current_epoch)
        # self.logger.experiment.log_metric('seg_loss', seg_loss.mean().data.cpu().numpy(), step=batch_idx, epoch=self.current_epoch)
        # self.logger.experiment.log_metric('accuracy', d0.mean().data.cpu().numpy(), step=batch_idx, epoch=self.current_epoch)

        return {'loss':  loss, 'seg_loss': seg_loss, 'accuracy': d0 }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([x['loss'] for x in outputs]).mean()
        avg_seg_loss = torch.cat([x['seg_loss'] for x in outputs]).mean()
        avg_accuracy = torch.cat([x['accuracy'] for x in outputs]).mean()

        metrics = {
            'avg_val_loss': avg_loss.data.cpu().numpy(),
            'avg_seg_loss': avg_seg_loss.data.cpu().numpy(),
            'avg_accuracy': avg_accuracy.data.cpu().numpy()
        }

        self.logger.log_metrics(metrics, step=self.current_epoch)

        # self.logger.experiment.log_metric('avg_val_loss', avg_loss.data.cpu().numpy(), epoch=self.current_epoch)
        # self.logger.experiment.log_metric('avg_seg_loss', avg_seg_loss.data.cpu().numpy(), epoch=self.current_epoch)
        # self.logger.experiment.log_metric('avg_accuracy', avg_accuracy.data.cpu().numpy(), epoch=self.current_epoch)
        
        return {'val_acc': avg_accuracy, 'val_loss': avg_loss, 'val_seg_loss': avg_seg_loss}



model = SeResNext50Lightning(
    batch_size = 20,
    val_batch_size = 4,
)

# model = SeResNext50Lightning.load_from_checkpoint(checkpoint_path="epoch_13.ckpt")

# arguments made to CometLogger are passed on to the comet_ml.Experiment class
comet_logger = CometLogger(
    # api_key="rjFRslN5SxsTdEQOqr1RySaYl",
    save_dir="comet",
    project_name="general", 
    workspace="lezwon"
)

tb_logger = TensorBoardLogger("lightning_logs", name="SeResNext50Lightning")

checkpoint_callback = ModelCheckpoint(
    filepath='checkpoints/{epoch}-{val_loss:.2f}-{val_acc:.2f}',
     monitor='val_acc',
     save_top_k=-1
     )

trainer = Trainer(gpus=2, distributed_backend='dp', logger=[comet_logger, tb_logger], amp_level='O1', precision=16, profiler=True, log_gpu_memory=True, use_amp=True, auto_lr_find=True, resume_from_checkpoint='epoch_17.ckpt', checkpoint_callback=checkpoint_callback)
trainer.fit(model)
trainer.test()