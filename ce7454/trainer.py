import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from model import CLIP_loss


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class BaseTrainer:
    def __init__(self,
                 net: nn.Module,
                 train_loader: DataLoader,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 lr_factor=1e-6,
                 epochs: int = 100,
                 model_name: str = 'model') -> None:
        self.net = net

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.net.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.train_loader = train_loader

        # self.optimizer = torch.optim.SGD(
        #     net.parameters(),
        #     learning_rate,
        #     momentum=momentum,
        #     weight_decay=weight_decay,
        #     nesterov=True,
        # )
        print("Only optimize prompt")
        self.optimizer = torch.optim.Adam(
            self.net.prompt_learner.parameters(), 
            learning_rate, 
            # weight_decay=weight_decay,
        )

        LR_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                lr_factor / learning_rate,
            ),
        )
        
        self.scheduler = GradualWarmupScheduler(
            self.optimizer, multiplier=1, total_epoch=100, after_scheduler=LR_scheduler
        )

        logfolder=f'./log/{model_name}'
        os.makedirs(logfolder, exist_ok=True)
        self.summary_writer = SummaryWriter(logfolder)
        self.curr_epoch = 0

    def cal_clip_loss(self, imgs, logits, relations, k=3):
        prob = torch.sigmoid(logits)
        pred = torch.topk(prob, k)[1]               # [N, K]
        
        texts = []
        for batch_pred in pred:
            text = ''
            for idx in batch_pred:
                text += relations[idx] + ' '
            texts.append(text)

        return self.clip_loss(imgs, texts)


    def train_epoch(self, pos_weight):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1, len(train_dataiter) + 1)):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['soft_label'].cuda()
            # forward
            logits = self.net(data)
            loss = F.binary_cross_entropy_with_logits(logits,
                                                      target,
                                                      reduction='sum',
                                                      pos_weight=pos_weight)

            # clip_loss = self.cal_clip_loss(data, logits, relations)
            
            total_loss = loss
            # backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.summary_writer.add_scalar('loss', loss.detach().item(), global_step=train_step+self.curr_epoch*len(train_dataiter))
            self.summary_writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step=train_step+self.curr_epoch*len(train_dataiter))
            # self.summary_writer.add_scalar('clip_loss', clip_loss.detach().item(), global_step=train_step+self.curr_epoch*len(train_dataiter))
            
            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['train_loss'] = loss_avg

        self.curr_epoch += 1

        return metrics
