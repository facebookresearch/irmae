#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import argparse
from tqdm import tqdm
import model
import utils


parser = argparse.ArgumentParser(description="Training Autoencoders")
parser.add_argument('-n', type=int, help='latent dimension', default=128)
parser.add_argument('-l', type=int, help='implicit layer', default=0)
parser.add_argument('--gpu', action='store_true', help='Use GPU?')
parser.add_argument('--train-size', type=int, default=50000)
parser.add_argument('--valid-size', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, help='#epochs', default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('--model-name', type=str)
parser.add_argument('--vae', action='store_true', help='VAE')
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--checkpoint', type=str, default="./checkpoint/")
parser.add_argument('--data-path', type=str, default="./data/")


def main(args):

    # use gpu ##########################################
    device = torch.device("cuda" if args.gpu else "cpu")
    torch.manual_seed(0)

    # dataset ##########################################
    if args.dataset == "mnist":
        args.data_path = args.data_path + "mnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        train_set = datasets.MNIST(args.data_path, train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
        valid_set = datasets.MNIST(args.data_path, train=False,
                                   download=True,
                                   transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
    elif args.dataset == "shape":
        train_set = utils.ShapeDataset(
            data_size=args.train_size)
        valid_set = utils.ShapeDataset(
            data_size=args.valid_size)
    elif args.dataset == "celeba":
        train_set = utils.ImageFolder(
            args.data_path + '/train/',
            transform=transforms.Compose([transforms.CenterCrop(148),
                                          transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        valid_set = utils.ImageFolder(
            args.data_path + '/val/',
            transform=transforms.Compose([transforms.CenterCrop(148),
                                          transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=32,
        batch_size=args.batch_size
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        num_workers=32,
        batch_size=args.batch_size
    )

    # init networks ##########################################

    net = model.AE(args)
    net.to(device)

    # optimizer ##########################################
    optimizer = optim.Adam(net.parameters(), args.lr)

    # train ################################################
    save_path = args.checkpoint + "/" + args.dataset + "/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for e in range(args.epochs):

        recon_loss = 0

        for yi, _, in tqdm(train_loader):
            net.train()

            optimizer.zero_grad()

            yi = yi.to(device)
            loss = net(yi)
            recon_loss += loss.item()

            loss.backward()
            optimizer.step()

        recon_loss /= len(train_loader)
        print("epoch " + str(e) + '\ttraining loss = ' + str(recon_loss))

        # save model ##########################################
        torch.save(net.state_dict(), save_path + args.model_name)

        valid_loss = 0

        for yi, _ in tqdm(valid_loader):
            net.eval()

            yi = yi.to(device)
            eval_loss = net(yi)
            valid_loss += eval_loss.item()

        valid_loss /= len(valid_loader)

        print("epoch " + str(e) + '\tvalid loss = ' + str(valid_loss))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
