#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import model
import utils


parser = argparse.ArgumentParser(description="interpolation")
parser.add_argument('-n', type=int, help='latent dimension', default=128)
parser.add_argument('-l', type=int, help='layers', default=0)
parser.add_argument('--vae', action='store_true', help='VAE')
parser.add_argument('--model-name', type=str, default="default")
parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('--checkpoint', type=str, default="./checkpoint/")
parser.add_argument('--data-path', type=str, default="./data/")
parser.add_argument('--save-path', type=str, default="./results/")


def main(args):

    # dataset ##########################################
    if args.dataset == "mnist":
        args.data_path = args.data_path + "mnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        test_set = datasets.MNIST(args.data_path, train=False,
                                  transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))

        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=100
        )
    elif args.dataset == "shape":
        test_set = utils.ShapeDataset()
        test_loader = torch.utils.data.DataLoader(
            test_set,
            shuffle=True,
            num_workers=32,
            batch_size=100
        )
    elif args.dataset == "celeba":
        test_set = utils.ImageFolder(
            args.data_path + '/test/',
            transform=transforms.Compose([transforms.CenterCrop(148),
                                          transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=100
        )

    # load model ##########################################

    plt.figure(figsize=(6, 4))

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

    dec.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset
                        + "/dec_" + args.model_name,
                        map_location=torch.device('cpu')))
    enc.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset
                        + "/enc_" + args.model_name,
                        map_location=torch.device('cpu')))
    dec.eval()
    enc.eval()

    if args.l > 0:
        mlp = model.MLP(args.n, args.l)
        mlp.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset
                            + "/mlp_" + args.model_name,
                            map_location=torch.device('cpu')))
        mlp.eval()

    z = []
    for yi, _ in test_loader:
        if args.vae:
            z_hat = enc(yi)
            mu = z_hat[:, :args.n]
            logvar = z_hat[:, args.n:]
            for j in range(100):
                z_hat = model.reparametrization(mu, logvar)
                z.append(z_hat)
        else:
            if args.l > 0:
                z_hat = mlp(enc(yi))
            else:
                z_hat = enc(yi)
            z.append(z_hat)

    z = torch.cat(z, dim=0).data.numpy()

    c = np.cov(z, rowvar=False)
    u, d, v = np.linalg.svd(c)

    d = d / d[0]

    plt.plot(range(args.n), d)

    plt.autoscale(enable=True, axis='y', tight=True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim(0, 0.4)
    plt.xlim(0, args.n)
    plt.xlabel("Singular Value Rank")
    plt.ylabel("Singular Values")
    plt.title("Singular Values of Covariance Matrix")

    if args.dataset == "shape":
        plt.axvline(x=7, color='k', linestyle='dashed', linewidth=1)

    path = args.save_path + "/" + args.dataset + "/singular/"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(args.save_path + "/" + args.dataset + "/singular/" + 
                args.model_name + ".png", bbox_inches='tight')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
