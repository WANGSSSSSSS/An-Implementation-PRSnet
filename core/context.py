import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from core.model import Net
from data.dataset import ShapeDataset
from core.loss import Regularization, DistanceLoss
from torch.utils.tensorboard import SummaryWriter


class Context :
    def __init__(self, SN, PN, lr, epochs, batch_size, alpha, train=False,
                 save_dir = "",
                 log_dir = "",
                 device = None):
        self.lr = lr
        self.epoch = 1
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.train = train
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.log_step = 0
        # basic item
        self.Net = Net(PN,PN,"bn", device)
        self.train_dataset = ShapeDataset()
        self.train_dataloader = DataLoader(self.train_dataset,self.batch_size)
        self.test_dataset = ShapeDataset()
        self.test_dataloader = DataLoader(self.test_dataset,self.batch_size)
        # train item
        if self.train :
            self.reg_loss = Regularization(device)
            self.dis_loss = DistanceLoss(SN, device)
            self.optimizer = Adam(self.Net.parameters(), lr=self.lr)
            self.scheduler = StepLR(self.optimizer, step_size=8000)
        else :
            pass

        self.writer = SummaryWriter(self.log_dir)
    def train_process(self):
        for epoch in range(self.epoch, self.epochs):
            self.epoch += 1
            loss, reg_loss, dis_loss = self.train_epoch()
            self.writer.add_scalar("train-epoch/Loss", loss, self.epoch)
            self.writer.add_scalar("train-epoch/Loss-reg", reg_loss, self.epoch)
            self.writer.add_scalar("train-epoch/Loss-dis", dis_loss, self.epoch)

            self.test_process()


    # TODO add tensorboard
    def train_epoch(self):
        sum_loss = 0
        sum_reg = 0
        sum_dis = 0
        for (i, batch) in enumerate(self.train_dataloader):
            self.log_step += 1
            self.optimizer.zero_grad()
            v = batch["voxel"].to(self.device)
            cp = batch["close_points"].to(self.device)
            points = batch["sample"].to(self.device)
            plane, quat = self.Net(v)
            reg_loss = self.reg_loss(plane, quat)
            dis_loss = self.dis_loss(points,cp, v,plane, quat)
            loss = (self.alpha * reg_loss + dis_loss)/self.batch_size  #reduce for mini-batch
            sum_loss += loss.item()
            sum_reg += reg_loss.item()
            sum_dis += dis_loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.lr = self.scheduler.get_lr()
            if self.log_step % 50 == 0 :
                self.writer.add_scalar("train/loss", loss.item(), self.log_step)
                self.writer.add_scalar("train/reg_loss", reg_loss.item(), self.log_step)
                self.writer.add_scalar("train/dis_loss", dis_loss.item(), self.log_step)

        sum_loss /= len(self.train_dataloader)
        sum_reg  /= len(self.train_dataloader)
        sum_dis  /= len(self.train_dataloader)
        return sum_loss, sum_reg, sum_dis

    def test_process(self):
        pass

    def __call__(self):
        if self.train :
            self.train_process()
        else :
            self.test_process()

    def save_context(self):
        dir_path = self.save_dir
        if not os.path.exists(dir_path) :
             os.mkdir(dir_path)
        check_point = {}
        check_point["model"] = self.Net.state_dict()
        check_point["optim"] = self.optimizer.state_dict()
        check_point["sched"] = self.scheduler.state_dict()
        check_point["epoch"] = self.epoch
        check_point["lr"] = self.lr
        torch.save(self.Net.state_dict(), dir_path + "/{}.pth".format(self.epoch))

    def load_context(self, path):
        check_point = torch.load(path)
        self.Net.load_state_dict(check_point["model"])
        self.optimizer.load_state_dict(check_point["optim"])
        self.scheduler.load_state_dict(check_point["sched"])
        self.lr = check_point["lr"]
        self.epoch = check_point["epoch"]
