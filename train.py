# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import os
import json
from typing import NamedTuple

from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from tqdm import tqdm

import torch
import torch.nn as nn

import checkpoint
from utils import get_logger

log = get_logger(__name__)


class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431  # random seed
    batch_size: int = 32
    lr: int = 5e-5  # learning rate
    n_epochs: int = 10  # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100  # interval for saving model
    total_steps: int = 100000  # total number of steps to train

    @classmethod
    def from_json(cls, file):  # load config from json file
        return cls(**json.load(open(file, "r")))


class Trainer(object):
    """Training Helper Class"""

    def __init__(self, cfg, model, data_iter, optimizer, save_dir, device):
        self.cfg = cfg  # config for training : see class Config
        self.model = model
        self.data_iter = data_iter  # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device  # device name

    def train(self, get_loss, model_file=None, pretrain_file=None, data_parallel=True):
        """ Train Loop """
        self.model.train()  # train mode
        self.load(model_file, pretrain_file)
        model = self.model.to(self.device)
        if data_parallel:  # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0  # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum = 0.  # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(self.data_iter)
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]  # 数据送到 device

                self.optimizer.zero_grad()
                loss = get_loss(model, batch, global_step).mean()  # mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                iter_bar.set_description(f'Epoch: {e}--loss: {loss.item():.3f}--Step: {global_step}--Iter: {i}')

                if global_step % self.cfg.save_steps == 0:  # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    log.info('Epoch %d/%d : Average Loss %5.3f' % (e + 1, self.cfg.n_epochs, loss_sum / (i + 1)))
                    log.info('The Total Steps have been reached.')
                    self.save(global_step)  # save and finish when global_steps reach total_steps
                    return

            log.info('Epoch %d/%d : Average Loss %5.3f' % (e + 1, self.cfg.n_epochs, loss_sum / (i + 1)))
        self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval()  # evaluation mode
        self.load(model_file, None)
        model = self.model.to(self.device)
        if data_parallel:  # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = []  # prediction results
        logits = []
        labels = []
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), "Elapsed:", TimeElapsedColumn()) as progress:
            for batch in progress.track(self.data_iter, description='Iter', total=len(self.data_iter)):
                batch = [t.to(self.device) for t in batch]
                with torch.no_grad():  # evaluation without gradient calculation
                    accuracy, result, logit, label_id = evaluate(model, batch)  # accuracy to print
                results.append(result)
                logits.append(logit)
                labels.append(label_id)

        return results, logits, labels

    def load(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            log.info(f'Loading the model from {model_file}')
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file:  # use pretrained transformer
            log.info('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'):  # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'):  # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                     for key, value in torch.load(pretrain_file).items()
                     if key.startswith('transformer')}
                )  # load only transformer parts

    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(),  # save model object before nn.DataParallel
                   os.path.join(self.save_dir, 'model_steps_' + str(i) + '.pt'))
