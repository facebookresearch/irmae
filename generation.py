#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import numpy as np
import argparse
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import mixture
import matplotlib.pyplot as plt
import model
import utils


parser = argparse.ArgumentParser(description="Generative and Downstream Tasks")
parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('--task', type=str, default="reconstruction")
parser.add_argument('-n', type=int, help='latent dimension', default=128)
parser.add_argument('-l', type=int, help='layers', default=0)
parser.add_argument('-d', type=int, help='PCA dimension', default=4)
parser.add_argument('-X', type=int, default=10)
parser.add_argument('-Y', type=int, default=10)
parser.add_argument('-N', type=int, default=1000)
parser.add_argument('--test-size', type=int, default=1000)
parser.add_argument('--model-name', type=str, default="default")
parser.add_argument('--vae', action='store_true', help='VAE')
parser.add_argument('--checkpoint', type=str, default="./checkpoint/")
parser.add_argument('--data-path', type=str, default="./data/")
parser.add_argument('--save-path', type=str, default="./results/")


def main(args):

    # seed ##############
    np.random.seed(2020)
    torch.manual_seed(1)

    # dataset ##########################################
    if args.dataset == "mnist":
        args.data_path = args.data_path + "mnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        test_set = datasets.MNIST(args.data_path, train=False,
                                  download=True,
                                  transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size
        )
    elif args.dataset == "shape":
        test_set = utils.ShapeDataset(data_size=args.test_size)
        test_set.set_seed(2020)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            shuffle=True,
            num_workers=32,
            batch_size=args.test_size
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
            batch_size=args.test_size
        )

    # load model ##########################################

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

    dec.load_state_dict(torch.load(
        args.checkpoint + "/" + args.dataset + "/dec_" + args.model_name,
        map_location=torch.device('cpu')))
    enc.load_state_dict(torch.load(
        args.checkpoint + "/" + args.dataset + "/enc_" + args.model_name,
        map_location=torch.device('cpu')))
    dec.eval()
    enc.eval()

    if args.l > 0:
        mlp = model.MLP(args.n, args.l)
        mlp.load_state_dict(torch.load(
                args.checkpoint + "/" + args.dataset +
                "/mlp_" + args.model_name,
                map_location=torch.device('cpu')))
        mlp.eval()

    #####################################################

    fig, axs = plt.subplots(args.X, args.Y, figsize=[args.Y, args.X])

    if args.task == "reconstruction":
        yi, _ = next(iter(test_loader))
        if args.vae:
            z_hat = enc(yi)
            mu = z_hat[:, :args.n]
            logvar = z_hat[:, args.n:]
            zi = model.reparametrization(mu, logvar)
        else:
            if args.l > 0:
                zi = mlp(enc(yi))
            else:
                zi = enc(yi)

        y_hat = dec(zi[:args.X * args.Y]).data.numpy()

    elif args.task == "interpolation":
        yi, _ = next(iter(test_loader))

        if args.vae:
            z_hat = enc(yi)
            mu = z_hat[:, :args.n]
            logvar = z_hat[:, args.n:]
            zi = model.reparametrization(mu, logvar)
        else:
            if args.l > 0:
                zi = mlp(enc(yi))
            else:
                zi = enc(yi)

        zs = []
        for i in range(args.X):
            z0 = zi[i*2]
            z1 = zi[i*2+1]

            for j in range(args.Y):
                zs.append((z0 - z1) * j / args.Y + z1)
        zs = torch.stack(zs, axis=0)
        y_hat = dec(zs).data.numpy()

    elif args.task == "mvg":
        z = []
        for yi, _ in test_loader:
            if args.vae:
                z_hat = enc(yi)
                mu = z_hat[:, :args.n]
                logvar = z_hat[:, args.n:]
                zi = model.reparametrization(mu, logvar)
            else:
                if args.l > 0:
                    zi = mlp(enc(yi))
                else:
                    zi = enc(yi)
            z.append(zi.detach().numpy())
        z = np.concatenate(z, axis=0)
        mu = np.average(z, axis=0)
        sigma = np.cov(z, rowvar=False)

        # generate corresponding sample z
        zs = np.random.multivariate_normal(mu, sigma, args.X * args.Y)
        zs = torch.Tensor(zs)

        y_hat = dec(zs).data.numpy()
    elif args.task == "gmm":
        z = []
        for yi, _ in test_loader:
            if args.vae:
                z_hat = enc(yi)
                mu = z_hat[:, :args.n]
                logvar = z_hat[:, args.n:]
                zi = model.reparametrization(mu, logvar)
            else:
                if args.l > 0:
                    zi = mlp(enc(yi))
                else:
                    zi = enc(yi)
            z.append(zi.detach().numpy())
        z = np.concatenate(z, axis=0)
        gmm = mixture.GaussianMixture(
            n_components=args.d, covariance_type='full')
        gmm.fit(z)

        zs, _ = gmm.sample(args.X * args.Y)
        zs = torch.Tensor(zs)
        y_hat = dec(zs).data.numpy()

    elif args.task == "pca":
        z = []
        for yi, _ in test_loader:
            if args.vae:
                z_hat = enc(yi)
                mu = z_hat[:, :args.n]
                logvar = z_hat[:, args.n:]
                zi = model.reparametrization(mu, logvar)
            else:
                if args.l > 0:
                    zi = mlp(enc(yi))
                else:
                    zi = enc(yi)
            z.append(zi.detach().numpy())
        z = np.concatenate(z, axis=0)

        pca = PCA(n_components=args.d)
        pca.fit(z)
        x = pca.transform(z)
        mu = np.average(x, axis=0)
        sigma = np.cov(x, rowvar=False)

        sigma_0 = np.sqrt(sigma[0][0])
        sigma_1 = np.sqrt(sigma[1][1])
        center = mu.copy()
        center[0] -= sigma_0 * 2
        center[1] -= sigma_1 * 2

        zs = []
        for i in range(args.X):
            tmp = []
            x = center.copy()
            x[0] += sigma_0 * i / args.X * 4
            for j in range(args.Y):
                x[1] += sigma_1 / args.Y * 4
                zi = pca.inverse_transform(x)
                tmp.append(zi)
            tmp = np.stack(tmp, axis=0)
            zs.append(tmp)
        zs = np.concatenate(zs, axis=0)
        zs = torch.Tensor(zs)

        y_hat = dec(zs).data.numpy()

    # now plot
    for i in range(args.X):
        for j in range(args.Y):

            if args.dataset == 'mnist':
                im = y_hat[i*args.Y+j][0, :, :]
            else:
                im = np.transpose(y_hat[i*args.Y+j], [1, 2, 0])
            if args.dataset == 'mnist':
                axs[i, j].imshow(1-im, interpolation='nearest', cmap='Greys')
            else:
                axs[i, j].imshow(im, interpolation='nearest')
            axs[i, j].axis('off')

    fig.tight_layout(pad=0.1)

    path = args.save_path + "/" + args.dataset + "/" + args.task
    if not os.path.exists(path):
        os.makedirs(path)
    path += "/" + args.model_name
    plt.savefig(path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
