import datetime
import random
import sys
import time
from typing import Dict, List

import numpy as np
from visdom import Visdom

import torch
from torch import nn
from torch.autograd import Variable


def tensor2image(tensor: torch.Tensor) -> np.ndarray:

    image: np.ndarray = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)

    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))

    return image.astype(np.uint8)


class Logger():

    def __init__(self, n_epochs: int, batches_epoch: int):

        self.viz: Visdom = Visdom(server='127.0.0.1', port=8097)
        self.n_epochs: int = n_epochs
        self.batches_epoch: int = batches_epoch
        self.epoch: int = 1
        self.batch: int = 1
        self.prev_time: time.time = time.time()
        self.mean_period: int = 0
        self.losses: Dict = {}
        self.loss_windows: Dict = {}
        self.image_windows: Dict = {}

    def log(self, losses: Dict = None, images: Dict = None) -> None:
        
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(f'Epoch {self.epoch:03d}/{self.n_epochs:03d} \
            [{self.batch:04d}/{self.batches_epoch:04d} -- ')

        for i, loss_name in enumerate(losses.keys()):

            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data.item()
            else:
                self.losses[loss_name] += losses[loss_name].data.item()
            
            sys.stdout.write(f'{loss_name}: {(self.losses[loss_name] / self.batches):.4f}')
            sys.stdout.write(' -- ' if (i + 1) == len(losses.keys()) else ' | ')

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        
        eta = datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)
        sys.stdout.write(f'ETA: {eta}')

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data),
                                                                opts=dict(title=image_name))
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
                               opts=dict(title=image_name))

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(
                        X=np.array([self.epoch]),
                        Y=np.array([loss / self.batch]),
                        opts=dict(xlabel='epochs', ylabel=loss_name, title=loss_name)
                    )
                else:
                    self.viz.line(
                        X=np.array([self.epoch]),
                        Y=np.array([loss / self.batch]),
                        win=self.loss_windows[loss_name], update='append'
                    )
                
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')

        else:
            self.batch += 1

        
class ReplayBuffer():

    def __init__(self, max_size: int = 50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size: int = max_size
        self.data = []

    def push_and_pop(self, data: torch.Tensor) -> Variable:
        
        to_return: List = []

        for element in data.data:
            
            element = torch.unsqueeze(element, 0)

            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)

        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs: int, offset: int, decay_start_epoch: int) -> None:

        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs: int = n_epochs
        self.offset: int = offset
        self.decay_start_epoch: int = decay_start_epoch

    def step(self, epoch: int) -> float:
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) \
            / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m: nn.Module) -> None:
    
    classname: str = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
