import os
import sys
import time
import math
import random
import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import models
import numpy as np 
import pdb
import torch.nn.functional as f
from torch.nn.utils import spectral_norm
# from main import args

resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight)
        init.xavier_uniform(m.bias)
    elif isinstance(m, nn.Linear):
        init.normal(m.weight, std=0.001)
        init.constant(m.bias, 0)

class net0(nn.Module):
    def __init__(self, num_class, backbone):
        super(net0, self).__init__()
        self.num_class = num_class

        # backbone
        if backbone=='res18':
            resnet = resnet18
            self.feature_dim = 512
        elif backbone=='res50':
            resnet = resnet50
            self.feature_dim = 2048

        self.layer0 = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool
                    )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = resnet.avgpool

        self.classifier1 = nn.Linear(self.feature_dim, self.num_class)
        self.classifier2 = nn.Linear(self.feature_dim, self.num_class)
        self.classifier3 = nn.Linear(self.feature_dim, self.num_class)
        self.classifiers = []
        self.classifiers.append(self.classifier1)
        self.classifiers.append(self.classifier2)
        self.classifiers.append(self.classifier3)


    def forward(self, x, domainid):
        # pdb.set_trace()
        z = self.layer0(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.pool(z)

        z = z.squeeze()
        # pdb.set_trace()
        y = self.classifiers[domainid](z)

        # return yc, common_kl, self.z
        return y, z

class zzclsfier(nn.Module):
    def __init__(self, input_f, num_class):
        super(zzclsfier, self).__init__()
        self.num_class = num_class
        self.input_f = input_f
        self.zlayer = nn.Linear(self.input_f, self.input_f)
        self.zlayermu = nn.Linear(self.input_f, self.input_f)
        self.zlayersg = nn.Linear(self.input_f, self.input_f)

    def forward(self, inputz, mc_times, domainid):
        # pdb.set_trace()
        inputz = self.zlayer(inputz)
        inputz = f.relu(inputz)
        pzmu = self.zlayermu(inputz)
        pzsigma = 0.1 + 0.9 * torch.nn.functional.softplus(self.zlayersg(inputz))
        if self.training:
            z_mu_samp = pzmu.unsqueeze(1).repeat(1, mc_times, 1)
            z_sigma_samp = pzsigma.unsqueeze(1).repeat(1, mc_times, 1)
            eps_q = z_mu_samp.new(z_mu_samp.size()).normal_()
            qz = z_mu_samp + 1 * z_sigma_samp * eps_q
            inputz = inputz.unsqueeze(1).repeat(1, mc_times, 1)
            qz = torch.cat([inputz, qz], -1)
            qz = qz.view(inputz.size()[0]*mc_times, self.input_f*2)
            # y = self.classifiers[domainid](qz)
            # y = y.view(inputz.size()[0])
        else:
            qz = pzmu
            qz = torch.cat([inputz, qz], -1)
        # y = self.classifiers[domainid](qz)
        # y = y.view(inputz.size()[0], mc_times**int(self.training), self.num_class)

        return 0, pzmu, pzsigma

class zzzclsfier(nn.Module):
    def __init__(self, input_f, num_class, drop=0):
        super(zzzclsfier, self).__init__()
        self.num_class = num_class
        self.input_f = input_f
        self.drop = drop
        self.zlayer = nn.Linear(self.input_f, self.input_f)
        self.zlayer2 = nn.Linear(self.input_f, self.input_f)
        self.zlayer3 = nn.Linear(self.input_f, self.input_f)
        self.zlayermu = nn.Linear(self.input_f, self.input_f)
        self.zlayersg = nn.Linear(self.input_f, self.input_f)
        if drop:
            # pdb.set_trace()
            self.dropout1 = nn.Dropout(0.2)
            self.dropout2 = nn.Dropout(0.1)
            self.dropout3 = nn.Dropout(0.1)

    def forward(self, inputz, mc_times, domainid):
        # pdb.set_trace()
        inputz = self.zlayer(inputz)
        inputz = f.relu(inputz)
        if self.drop:
            inputz = self.dropout1(inputz)
        inputz = self.zlayer2(inputz)
        inputz = f.relu(inputz)
        if self.drop:
            inputz = self.dropout2(inputz)
        inputz = self.zlayer3(inputz)
        inputz = f.relu(inputz)
        if self.drop:
            inputz = self.dropout3(inputz)
        pzmu = self.zlayermu(inputz)
        pzsigma = 0.1 + 0.9 * torch.nn.functional.softplus(self.zlayersg(inputz))

        return 0, pzmu, pzsigma

class net1(nn.Module):
    def __init__(self, num_class, backbone):
        super(net1, self).__init__()
        self.num_class = num_class

        # backbone
        if backbone=='res18':
            resnet = resnet18
            self.feature_dim = 512
        elif backbone=='res50':
            resnet = resnet50
            self.feature_dim = 2048

        self.layer0 = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool
                    )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = resnet.avgpool

        self.classifier = nn.Linear(self.feature_dim, self.num_class)


    def forward(self, x, domainid):
        # pdb.set_trace()
        z = self.layer0(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.pool(z)

        z = z.squeeze()
        # pdb.set_trace()
        y = self.classifier(z)

        # return yc, common_kl, self.z
        return y, z


def swish(x):
    return x * torch.sigmoid(x)

class ebm(nn.Module):
    def __init__(self, feature_dim, spec_norm=False, energy_type='sigmoid', prenorm='tanh'):
        super(ebm, self).__init__()
        self.feature_dim = feature_dim
        self.energy_type = energy_type

        self.layer0 = nn.Linear(self.feature_dim, self.feature_dim)
        self.drop0 = nn.Dropout(0.2)
        self.layer1 = nn.Linear(self.feature_dim, 128)
        self.drop1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(128, 1)
        self.act = swish
        self.prenorm = prenorm
        self.IN = nn.InstanceNorm1d(self.feature_dim)
        # self.act = nn.LeakyReLU(0.2)
        # self.act = nn.ReLU()

        if spec_norm:
            self.layer0 = spectral_norm(self.layer0)
            self.layer1 = spectral_norm(self.layer1)
            self.layer2 = spectral_norm(self.layer2)

    def forward(self, x):
        # pdb.set_trace()
        if self.prenorm=='tanh':
            x = f.tanh(x)
        elif self.prenorm=='sigmoid':
            x = f.sigmoid(x)
        elif self.prenorm=='in':
            x = x.view(x.size()[0], x.size()[1], 1)
            x = self.IN(x)
            x = x.view(x.size()[0], x.size()[1])
        # pdb.set_trace()
        x = self.layer0(x)
        x = self.act(x)
        x = self.drop0(x)
        x = self.layer1(x)
        x = self.act(x)
        x = self.drop1(x)
        energy = self.layer2(x)

        if self.energy_type == 'square':
            energy = torch.pow(energy, 2)

        elif self.energy_type == 'sigmoid':
            energy = f.sigmoid(energy)

        return energy


class ebms(nn.Module):
    def __init__(self, backbone='res18', num_domain=3, spec_norm=False, energy_type='sigmoid', prenorm='tanh'):
        super(ebms, self).__init__()
        if backbone=='res18':
            resnet = resnet18
            self.feature_dim = 512
        elif backbone=='res50':
            resnet = resnet50
            self.feature_dim = 2048

        self.models = []
        # for i in range(num_domain):
        self.ebm1 = ebm(self.feature_dim, spec_norm, energy_type, prenorm)
        self.ebm2 = ebm(self.feature_dim, spec_norm, energy_type, prenorm)
        self.ebm3 = ebm(self.feature_dim, spec_norm, energy_type, prenorm)
        self.models.append(self.ebm1)
        self.models.append(self.ebm2)
        self.models.append(self.ebm3)

    def forward(self, x, domainid):
        energy = self.models[domainid](x)

        return energy


class ebmz_cat(nn.Module):
    def __init__(self, feature_dim, spec_norm=False, energy_type='sigmoid', prenorm='tanh', init_lr=100):
        super(ebmz_cat, self).__init__()
        self.feature_dim = feature_dim
        self.energy_type = energy_type

        self.layer0 = nn.Linear(self.feature_dim*2, self.feature_dim)
        self.drop0 = nn.Dropout(0.2)
        self.layer1 = nn.Linear(self.feature_dim, 128)
        self.drop1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(128, 1)
        self.act = swish
        self.prenorm = prenorm
        self.IN = nn.InstanceNorm1d(self.feature_dim)
        self.lr = nn.Parameter(torch.ones(1)*init_lr)
        # self.act = nn.LeakyReLU(0.2)
        # self.act = nn.ReLU()

        if spec_norm:
            self.layer0 = spectral_norm(self.layer0)
            self.layer1 = spectral_norm(self.layer1)
            self.layer2 = spectral_norm(self.layer2)

    def forward(self, x, z):
        # pdb.set_trace()
        x = torch.cat([x,z], -1)
        x = self.layer0(x)
        x = self.act(x)
        x = self.drop0(x)
        x = self.layer1(x)
        x = self.act(x)
        x = self.drop1(x)
        energy = self.layer2(x)

        if self.energy_type == 'square':
            energy = torch.pow(energy, 2)

        elif self.energy_type == 'sigmoid':
            energy = f.sigmoid(energy)

        return energy


class ebmz_prod(nn.Module):
    def __init__(self, feature_dim, spec_norm=False, energy_type='sigmoid', prenorm='tanh', init_lr=100):
        super(ebmz_prod, self).__init__()
        self.feature_dim = feature_dim
        self.energy_type = energy_type

        self.layerx0 = nn.Linear(self.feature_dim, self.feature_dim)
        self.dropx0 = nn.Dropout(0.2)
        self.layerx1 = nn.Linear(self.feature_dim, 128)

        self.layerz0 = nn.Linear(self.feature_dim, self.feature_dim)
        self.dropz0 = nn.Dropout(0.2)
        self.layerz1 = nn.Linear(self.feature_dim, 128)
        self.lr = nn.Parameter(torch.ones(1)*init_lr)

        # self.dropx1 = nn.Dropout(0.2)
        # self.layerx2 = nn.Linear(128, 1)
        self.act = swish
        self.prenorm = prenorm
        # self.act = nn.LeakyReLU(0.2)
        # self.act = nn.ReLU()

        if spec_norm:
            self.layerx0 = spectral_norm(self.layerx0)
            self.layerx1 = spectral_norm(self.layerx1)
            self.layerz0 = spectral_norm(self.layerz0)
            self.layerz1 = spectral_norm(self.layerz1)

    def forward(self, x, z):
        # pdb.set_trace()
        x = self.layerx0(x)
        x = self.act(x)
        x = self.dropx0(x)
        x = self.layerx1(x)

        z = self.layerz0(z)
        z = self.act(z)
        z = self.dropz0(z)
        z = self.layerz1(z)

        energy = (- x * z).mean(-1)

        if self.energy_type == 'square':
            energy = torch.pow(energy, 2)

        elif self.energy_type == 'sigmoid':
            energy = f.sigmoid(energy)

        return energy

class ebmz_sum(nn.Module):
    def __init__(self, feature_dim, spec_norm=False, energy_type='sigmoid', prenorm='tanh', init_lr=100):
        super(ebmz_sum, self).__init__()
        self.feature_dim = feature_dim
        self.energy_type = energy_type

        self.layer0 = nn.Linear(self.feature_dim, self.feature_dim)
        self.drop0 = nn.Dropout(0.2)
        self.layer1 = nn.Linear(self.feature_dim, 128)
        self.drop1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(128, 1)
        self.act = swish
        self.lr = nn.Parameter(torch.ones(1)*init_lr)

        if spec_norm:
            self.layer0 = spectral_norm(self.layer0)
            self.layer1 = spectral_norm(self.layer1)
            self.layer2 = spectral_norm(self.layer2)

    def forward(self, x, z):
        # pdb.set_trace()
        x1 = self.layer0(x)
        x1 = self.act(x1)
        x1 = self.drop0(x1)
        x1 = self.layer1(x1)
        x1 = self.act(x1)
        x1 = self.drop1(x1)
        energy = self.layer2(x1)
        # pdb.set_trace()
        energy2 = -(x*z).mean()
        # pdb.set_trace()

        if self.energy_type == 'square':
            energy = (torch.pow(energy, 2) + torch.pow(energy2)) / 2

        elif self.energy_type == 'sigmoid':
            energy = (f.sigmoid(energy) + f.sigmoid(energy2)) / 2

        return energy

class ebmzs(nn.Module):
    def __init__(self, backbone='res18', num_domain=3, spec_norm=False, energy_type='sigmoid', fusionmethod='concat', init_lr=100):
        super(ebmzs, self).__init__()
        if backbone=='res18':
            resnet = resnet18
            self.feature_dim = 512
        elif backbone=='res50':
            resnet = resnet50
            self.feature_dim = 2048
        elif backbone=='res26':
            self.feature_dim = 64

        self.models = []
        if fusionmethod == 'concat':
        # for i in range(num_domain):
            self.ebm1 = ebmz_cat(self.feature_dim, spec_norm, energy_type, init_lr=init_lr)
            self.ebm2 = ebmz_cat(self.feature_dim, spec_norm, energy_type, init_lr=init_lr)
            self.ebm3 = ebmz_cat(self.feature_dim, spec_norm, energy_type, init_lr=init_lr)
        elif fusionmethod=='dotp':
            self.ebm1 = ebmz_prod(self.feature_dim, spec_norm, energy_type, init_lr=init_lr)
            self.ebm2 = ebmz_prod(self.feature_dim, spec_norm, energy_type, init_lr=init_lr)
            self.ebm3 = ebmz_prod(self.feature_dim, spec_norm, energy_type, init_lr=init_lr)
        elif fusionmethod=='sum':
            self.ebm1 = ebmz_sum(self.feature_dim, spec_norm, energy_type, init_lr=init_lr)
            self.ebm2 = ebmz_sum(self.feature_dim, spec_norm, energy_type, init_lr=init_lr)
            self.ebm3 = ebmz_sum(self.feature_dim, spec_norm, energy_type, init_lr=init_lr)
        self.models.append(self.ebm1)
        self.models.append(self.ebm2)
        self.models.append(self.ebm3)

    def forward(self, x, z, domainid):
        energy = self.models[domainid](x, z)

        return energy


class domain_ebms(nn.Module):
    def __init__(self, backbone='res18', num_domain=3, spec_norm=False, energy_type='sigmoid', prenorm='tanh'):
        super(domain_ebms, self).__init__()
        if backbone=='res18':
            resnet = resnet18
            self.feature_dim = 512
        elif backbone=='res50':
            resnet = resnet50
            self.feature_dim = 2048

        self.models = []
        self.energy_type = energy_type
        # for i in range(num_domain):
        self.ebm1 = domain_ebm(self.feature_dim, spec_norm, energy_type)
        self.ebm2 = domain_ebm(self.feature_dim, spec_norm, energy_type)
        self.ebm3 = domain_ebm(self.feature_dim, spec_norm, energy_type)
        self.models.append(self.ebm1)
        self.models.append(self.ebm2)
        self.models.append(self.ebm3)

    def forward(self, x, domainid):
        if self.energy_type=='softmax':
            # pdb.set_trace()
            energys = torch.zeros(x.size()[0], len(self.models))
            for i in range(len(self.models)):
                energys[:,i] = self.models[i](x)[:,0]
            # pdb.set_trace()
            energys = f.softmax(energys, 1)
            energy = -energys[:, domainid]
            # pdb.set_trace()
        else:
            energy = -self.models[domainid](x)

        return energy

class classifier_generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(classifier_generator, self).__init__()
        self.shared_net = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU()
                        )
        self.shared_mu = nn.Linear(hidden_size, output_size)
        self.shared_sigma = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        z = self.shared_net(x)
        return self.shared_mu(z), self.shared_sigma(z)
