# !/usr/bin/env python
# encoding: utf-8

import argparse
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.DPDTRN_x4 import Usmsrn
from loss.L1_loss import hdrt_loss
import math
import numpy as np
from dataset.dataset_x4_x8 import DatasetFromFolder
from torch.utils.tensorboard import SummaryWriter

import os
import cv2
import sys
from tqdm import tqdm
import copy

import datetime

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DPDTRN")
parser.add_argument("--scale", type=int, default=4, help="training batch size")
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
parser.add_argument("--earlyLrStep", type=int, default=500, help="training batch size")
parser.add_argument("--lrReduce", type=int, default=0.5, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default="",
                    type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained",
                    default="",
                    type=str)

parser.add_argument("--train_file", default=['./data/Text330/train/total_crop',
                                             './data/Text330/train/total_crop_x4',
                                             './data/Text330/train/total_crop_x4',
                                             './data/Text330/train/total_crop_x2'
                                             ],
                    type=list)

Model_Name = 'DPDTRN_x4'


def main():
    time = datetime.datetime.now()
    global opt, model
    opt = parser.parse_args()
    print(opt)

    scale = opt.scale
    log_dir = './results/log/save/' + Model_Name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_writer = SummaryWriter(log_dir=log_dir)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 1000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.enabled = True
    cudnn.benchmark = True

    print("===> Loading datasets")

    train_set = DatasetFromFolder(opt.train_file[0], opt.train_file[1], opt.train_file[2], opt.train_file[3])
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)

    print("===> Building model")
    model = Usmsrn()

    criterion = nn.L1Loss(reduction='mean')
    criterion_lap = hdrt_loss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion_lap = criterion_lap.cuda()
    else:
        model = model.cpu()

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            opt.start_epoch = 0
            state_dict = torch.load(opt.resume)
            model.load_state_dict(state_dict)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}' ".format(opt.pretrained))
            state_dict = torch.load(opt.pretrained)
            model.load_state_dict(state_dict)
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch, tb_writer, criterion_lap)
        if epoch % 1 == 0:
            save_best_checkpoint(model, epoch, time, scale)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    if epoch < opt.earlyLrStep:
        lr = opt.lr
    else:
        step = opt.step
        lr = opt.lr * (opt.lrReduce ** (epoch // step - 1))
        if lr <= 1e-6:
            lr = 1e-6
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch, tb_writer, criterion_lap):
    avg_loss = 0
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, batch in tqdm(enumerate(training_data_loader, 1), desc='train dataset',
                                 total=len(training_data_loader)):

        input_img, x2_label, HR_label, TextGt = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(
            batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)

        if opt.cuda:
            input_img = input_img.cuda()
            HR_label = HR_label.cuda()

        x4, textures = model(input_img)

        textures = torch.tanh(textures)
        loss = criterion(x4, HR_label) + 0.1 * criterion_lap(x4, HR_label, textures)

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): loss: {:.10f}".format(epoch,
                                                                iteration,
                                                                len(
                                                                    training_data_loader),
                                                                loss.item()))

        tb_writer.add_scalar(Model_Name + '_loss', loss, epoch)
        avg_loss += loss.item()
    print('average loss ....................', avg_loss / len(training_data_loader))


def save_best_checkpoint(model, epoch, time, scale):
    time = str(time).split('.')[0].replace(' ', '').replace(':', '').replace('-', '')
    model_folder = "./results/trained_models/save/" + Model_Name + '/' + str(time) + '_x' + str(scale) + '/'
    model_out_path = model_folder + "best_model_" + str(epoch) + ".pth"
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state["model"].state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
