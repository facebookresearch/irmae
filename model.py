#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import numpy as np


# VAE sample
def reparametrization(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# for downstream classification
class Head(nn.Module):
    def __init__(self, code_dim, out_channels):
        super(Head, self).__init__()
        self.code_dim = code_dim
        self.out_channels = out_channels
        self.hidden = nn.ModuleList()
        for k in range(2):
            self.hidden.append(nn.Linear(code_dim, code_dim, bias=True))
            self.hidden.append(nn.ReLU(True))
            self.hidden.append(nn.Dropout2d(0.5))
        self.hidden.append(nn.Linear(code_dim, out_channels, bias=False))

    def forward(self, z):
        for l in self.hidden:
            z = l(z)
        return z


class MLP(nn.Module):
    def __init__(self, code_dim, layers):
        super(MLP, self).__init__()
        self.code_dim = code_dim
        self.layers = layers
        self.hidden = nn.ModuleList()
        for k in range(layers):
            linear_layer = nn.Linear(code_dim, code_dim, bias=False)
            self.hidden.append(linear_layer)

    def forward(self, z):
        for l in self.hidden:
            z = l(z)
        return z


class Shape_Decoder(nn.Module):
    def __init__(self, code_dim, vae=False):
        super(Shape_Decoder, self).__init__()
        self.code_dim = code_dim
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(code_dim, 256, 2, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid() if vae else nn.Tanh()
        )

    def forward(self, z):
        out = self.dcnn(z.view(z.size(0), self.code_dim, 1, 1))
        return out


class Shape_Encoder(nn.Module):
    def __init__(self, code_dim):
        super(Shape_Encoder, self).__init__()
        self.code_dim = code_dim
        self.dcnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    # 32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),      # 16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),      # 8x8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),      # 4x4
            nn.ReLU(True),
            nn.Conv2d(256, code_dim, 2, 1, 0),    # 1x1
        )

    def forward(self, z):
        return self.dcnn(z).view(z.size(0), self.code_dim)


class MNIST_Encoder(nn.Module):
    def __init__(self, code_dim):
        super(MNIST_Encoder, self).__init__()
        self.code_dim = code_dim
        self.dcnn = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(256*2*2, self.code_dim)

    def forward(self, z):
        z = self.dcnn(z)
        z = z.view(z.size(0), 256*2*2)
        z = self.fc(z)
        return z


class MNIST_Decoder(nn.Module):
    def __init__(self, code_dim, vae=False):
        super(MNIST_Decoder, self).__init__()
        self.code_dim = code_dim
        self.fc = nn.Linear(self.code_dim, 8*8*128)
        self.dcnn = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 1, 1, 0),
            nn.Sigmoid() if vae else nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 128, 8, 8)
        z = self.dcnn(z)
        return z


class CelebA_Encoder(nn.Module):
    def __init__(self, code_dim):
        super(CelebA_Encoder, self).__init__()
        self.code_dim = code_dim
        self.dcnn = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(1024*4*4, self.code_dim)

    def forward(self, z):
        z = self.dcnn(z)
        z = z.view(z.size(0), 1024*4*4)
        z = self.fc(z)
        return z


class CelebA_Decoder(nn.Module):
    def __init__(self, code_dim, vae=False):
        super(CelebA_Decoder, self).__init__()
        self.code_dim = code_dim
        self.fc = nn.Linear(self.code_dim, 8*8*1024)
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 1, 1, 0),
            nn.Sigmoid() if vae else nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 1024, 8, 8)
        z = self.dcnn(z)
        return z
