#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm
import model


parser = argparse.ArgumentParser(description="MNIST interpolation")
parser.add_argument('-n', type=int, help='latent dimension', default=128)
parser.add_argument('--gpu', action='store_true', help='Use GPU?')
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--train-size', type=int, default=60000)
parser.add_argument('--epochs', type=int, help='#epochs', default=500)
parser.add_argument('--model-name', type=str, default="default")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--supervised', action='store_true', help='supervised')
parser.add_argument('--vae', action='store_true', help='VAE')
parser.add_argument('--checkpoint', type=str, default="./checkpoint/")
parser.add_argument('--data-path', type=str, default="./data/")


def main(args):

    torch.manual_seed(0)

    # use gpu ##########################################
    device = torch.device("cuda" if args.gpu else "cpu")

    # dataset ##########################################
    args.data_path = args.data_path + "mnist/"
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    train_set = datasets.MNIST(args.data_path, train=True,
                                transform=transforms.Compose(
                                [transforms.Resize(32),
                                    transforms.ToTensor()]))
    valid_set = datasets.MNIST(args.data_path, train=False,
                                transform=transforms.Compose(
                                [transforms.Resize(32),
                                    transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=32,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(
            range(args.train_size))
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        num_workers=32,
        batch_size=args.batch_size,
    )

    # load model ##########################################

    if args.vae:
        enc = model.MNIST_Encoder(args.n * 2)
    else:
        enc = model.MNIST_Encoder(args.n)

    enc.to(device)

    if not args.supervised:
        enc.load_state_dict(torch.load(
            args.checkpoint + "/mnist/enc_" + args.model_name,
            map_location=torch.device('cpu')))

    head = model.Head(args.n, 10)
    head.to(device)

    # optimizer ##########################################
    if args.supervised:
        optimizer = optim.Adam([
            {'params': enc.parameters(), 'lr': args.lr},
            {'params': head.parameters(), 'lr': args.lr}
        ])
    else:
        optimizer = optim.Adam([
            {'params': head.parameters(), 'lr': args.lr}
        ])

    # train ################################################

    max_val = 0
    for e in range(args.epochs):
        for xi, yi in tqdm(train_loader):
            if args.supervised:
                enc.train()

            head.train()

            xi = xi.to(device)
            yi = yi.to(device)

            optimizer.zero_grad()

            if args.vae:
                z_hat = enc(xi)
                zi = z_hat[:, :args.n]
            else:
                zi = enc(xi)

            y_hat = head(zi)

            logits = F.log_softmax(y_hat, dim=1)
            loss = F.nll_loss(logits, yi)

            pred = y_hat.argmax(dim=1, keepdim=False)
            loss.backward()
            optimizer.step()

        # validation ###
        if e % 10 == 9:
            acc = 0
            for xi, yi in tqdm(valid_loader):

                if args.supervised:
                    enc.eval()

                head.eval()

                xi = xi.to(device)
                yi = yi.to(device)

                if args.vae:
                    z_hat = enc(xi)
                    zi = z_hat[:, :args.n]
                else:
                    zi = enc(xi)

                y_hat = head(zi)

                pred = y_hat.argmax(dim=1, keepdim=False)
                acc += pred.eq(yi).sum().item()
            acc /= len(valid_loader.dataset)
            max_val = max(max_val, acc)
            print("epoch " + str(e) + '\taccuracy = ' + str(acc))
            print("max accuracy = " + str(max_val))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
