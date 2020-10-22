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

    if args.dataset == "mnist":
        if args.vae:
            enc = model.MNIST_Encoder(args.n * 2)
            dec = model.MNIST_Decoder(args.n, vae=True)
        else:
            enc = model.MNIST_Encoder(args.n)
            dec = model.MNIST_Decoder(args.n)
    elif args.dataset == "celeba":
        if args.vae:
            enc = model.CelebA_Encoder(args.n * 2)
            dec = model.CelebA_Decoder(args.n, vae=True)
        else:
            enc = model.CelebA_Encoder(args.n)
            dec = model.CelebA_Decoder(args.n)
    elif args.dataset == "shape":
        if args.vae:
            enc = model.Shape_Encoder(args.n * 2)
            dec = model.Shape_Decoder(args.n, vae=True)
        else:
            enc = model.Shape_Encoder(args.n)
            dec = model.Shape_Decoder(args.n)

    dec.to(device)
    enc.to(device)

    if not args.vae and args.l > 0:
        mlp = model.MLP(args.n, args.l)
        mlp.to(device)

    # optimizer ##########################################
    if args.l > 0:
        optimizer = optim.Adam([
            {'params': dec.parameters(), 'lr': args.lr},
            {'params': enc.parameters(), 'lr': args.lr},
            {'params': mlp.parameters(), 'lr': args.lr},
        ])
    else:
        optimizer = optim.Adam([
            {'params': dec.parameters(), 'lr': args.lr},
            {'params': enc.parameters(), 'lr': args.lr},
        ])

    # train ################################################
    save_path = args.checkpoint + "/" + args.dataset

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for e in range(args.epochs):

        recon_loss = 0

        for yi, _, in tqdm(train_loader):
            enc.train()
            dec.train()
            if args.l > 0:
                mlp.train()

            optimizer.zero_grad()

            yi = yi.to(device)
            z_hat = enc(yi)

            if args.vae:
                mu = z_hat[:, :args.n]
                logvar = z_hat[:, args.n:]
                z_bar = model.reparametrization(mu, logvar)
            else:
                if args.l > 0:
                    z_bar = mlp(z_hat)
                else:
                    z_bar = z_hat

            y_hat = dec(z_bar)
            if args.vae:
                loss = F.binary_cross_entropy(y_hat, yi)
            else:
                loss = F.mse_loss(y_hat, yi)
            recon_loss += loss.item()

            if args.vae:
                loss -= args.beta * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp())

            loss.backward()
            optimizer.step()

        recon_loss /= len(train_loader)
        z_norm = np.average(np.sqrt(np.sum(
            z_hat.detach().cpu().numpy()**2, axis=1)))

        print("epoch " + str(e) + '\ttraining loss = ' + str(recon_loss)
              + '\tz norm = ' + str(z_norm))

        # save model ##########################################
        torch.save(enc.state_dict(),
                   save_path + "/enc_" + args.model_name)
        torch.save(dec.state_dict(),
                   save_path + "/dec_" + args.model_name)

        if args.l > 0:
            torch.save(mlp.state_dict(),
                       save_path + "/mlp_" + args.model_name)

        valid_loss = 0

        for yi, _ in tqdm(valid_loader):
            enc.eval()
            dec.eval()

            if args.l > 0:
                mlp.eval()

            yi = yi.to(device)
            z_eval = enc(yi)

            if args.vae:
                mu = z_eval[:, :args.n]
                logvar = z_eval[:, args.n:]
                z_bar_eval = model.reparametrization(mu, logvar)
            else:
                if args.l > 0:
                    z_bar_eval = mlp(z_eval)
                else:
                    z_bar_eval = z_eval
            y_eval = dec(z_bar_eval)

            eval_loss = F.mse_loss(y_eval, yi)
            valid_loss += eval_loss.item()

        valid_loss /= len(valid_loader)

        print("epoch " + str(e) + '\tvalid loss = ' + str(valid_loss))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
