import math
import time
import numpy as np
from itertools import cycle

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import distributions as D


def train_one_step(i_exps, current_AL_loop, epoch, model, alpha, optimizer, labelled, unlabelled, use_cuda):
    y_dim = model.y_dim
    t0 = time.time()
    model.train()
    total_U, total_loss, accuracy = (0, 0, 0)
    batch_id = 0

    if len(labelled) > len(unlabelled):
        m = len(labelled)
        zipped_data = zip(labelled, cycle(unlabelled))
    else:
        m = len(unlabelled)
        zipped_data = zip(cycle(labelled), unlabelled)

    # for (x, y, l_idx), (u, _, u_idx) in zip(cycle(labelled), unlabelled):
    for (x, y, l_idx), (u, _, u_idx) in zipped_data:
        batch_id += 1
        # If y is not in onehot format
        if len(y.shape) == 1:
            y = F.one_hot(y, num_classes=y_dim)
        optimizer.zero_grad()
        # Wrap in variables
        x, y, u = Variable(x.float()), Variable(y.float()), Variable(u.float())

        if use_cuda:
            x, y = x.cuda(), y.cuda()
            u = u.cuda()

        L = -model(x, y)
        U = -model(u)

        # Add auxiliary classification loss q(y|x)
        logProb_y = model.classify(x)

        # cross entropy
        classication_loss = -torch.sum(y * logProb_y, dim=1).mean()

        J_alpha = L + U + alpha * classication_loss

        J_alpha.backward()
        optimizer.step()

        total_U += -U.item()

        total_loss += J_alpha.item()
        accuracy += torch.mean((torch.max(logProb_y, 1)[1].data == torch.max(y, 1)[1].data).float()).item()
        if batch_id % 10 == 0:
            print(".", end='')
    print()

    t1 = time.time()
    dtime = t1 - t0
    # display.clear_output(wait=False)
    # m = len(unlabelled)
    # if epoch % 10 == 0:
    print("Exp {} - Active Learning Loop number: {}, Epoch: {}, \t Time per Epoch: {:.2f} s".format(
            i_exps, current_AL_loop, epoch, dtime))
    print("[Train]\t\t ELBO_px: {:.2f}, J_a: {:.2f}, accuracy: {:.2f}".format(total_U / m, total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


def test(i_exps, current_AL_loop, epoch, model, alpha, validation, use_cuda):

    y_dim = model.y_dim
    model.eval()

    total_U, total_loss, accuracy = (0, 0, 0)

    for x, y, _ in validation:
        # If y is not in onehot format
        if len(y.shape) == 1:
            y = F.one_hot(y, num_classes=y_dim)
        x, y = Variable(x.float()), Variable(y.float())

        if use_cuda:
            x, y = x.cuda(), y.cuda()

        L = -model(x, y)
        U = -model(x)

        logProb_y = model.classify(x)
        classication_loss = -torch.sum(y * logProb_y, dim=1).mean()

        J_alpha = L + U + alpha * classication_loss

        total_U += -U.item()

        total_loss += J_alpha.item()

        _, pred_idx = torch.max(logProb_y, 1)
        _, lab_idx = torch.max(y, 1)
        accuracy += torch.mean((pred_idx == lab_idx).float()).item()

    m = len(validation)
    print("[Validation]\t ELBO_px: {:.2f}, J_a: {:.2f}, accuracy: {:.2f}".format(total_U / m, total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


def draw_halfmoon_samples(model, use_cuda, n=10):
    y_dim = model.y_dim
    z_dim = model.z_dim
    model.eval()
    z = Variable(torch.randn(n, z_dim))

    y_ = np.array([0] * (n // 2) + [1] * (n - n// 2))
    y = torch.from_numpy(y_)
    y = Variable(F.one_hot(y, num_classes=y_dim).float())

    if use_cuda:
        z, y = z.cuda(), y.cuda()

    x = model.sample(y, z)
    samples = x.data.view(-1, 2).cpu().numpy()

    return samples, y_






class M2(nn.Module):
    """ Semi-supervised Generative Model M2 from Kingma's paper

    """

    def __init__(self, x_dim=784, y_dim=10, z_dim=32,
                 h_en_dim=[256, 128], h_de_dim=[128, 256],
                 h_cls_dim=[256, 128], use_mnist=False):
        super(M2, self).__init__()

        if use_mnist:
            print("Created M2 MNIST model.")
        else:
            print("Created M2 HalfMoon model.")

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.use_mnist = use_mnist

        en_neurons = [x_dim + y_dim, *h_en_dim]
        de_neurons = [y_dim + z_dim, *h_de_dim]
        cls_neurons = [x_dim, *h_cls_dim]

        en_linear_layers = [nn.Linear(en_neurons[i - 1], en_neurons[i]) for i in range(1, len(en_neurons))]
        de_linear_layers = [nn.Linear(de_neurons[i - 1], de_neurons[i]) for i in range(1, len(de_neurons))]
        cls_linear_layers = [nn.Linear(cls_neurons[i - 1], cls_neurons[i]) for i in range(1, len(cls_neurons))]

        # For encoder - q(z|x,y)
        self.en_hidden = nn.ModuleList(en_linear_layers)
        self.mu = nn.Linear(h_en_dim[-1], z_dim)
        self.log_var = nn.Linear(h_en_dim[-1], z_dim)

        # For decoder - p(x|y,z)
        self.de_hidden = nn.ModuleList(de_linear_layers)
        if self.use_mnist:
            self.reconstruction = nn.Linear(h_de_dim[-1], x_dim)
        else:
            self.reconstruction_mu = nn.Linear(h_de_dim[-1], x_dim)
            self.reconstruction_log_var = nn.Linear(h_de_dim[-1], x_dim)

        # For classifier - q(y|x)
        self.cls_hidden = nn.ModuleList(cls_linear_layers)
        self.logits = nn.Linear(h_cls_dim[-1], y_dim)

    def encode(self, x, y):
        h = torch.cat([x, y], dim=1)
        for layer in self.en_hidden:
            h = F.relu(layer(h))

        mu = self.mu(h)
        log_var = F.softplus(self.log_var(h))

        return self.reparameterize(mu, log_var), mu, log_var

    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        if mu.is_cuda:
            eps = eps.cuda()

        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def decode(self, y, z):
        h = torch.cat([y, z], dim=1)
        for layer in self.de_hidden:
            h = F.relu(layer(h))

        if self.use_mnist:
            return torch.sigmoid(self.reconstruction(h))
        else:
            mu = self.reconstruction_mu(h)
            # log_var = torch.full_like(mu, -0.5)
            # if mu.is_cuda:
            #     log_var = log_var.cuda()
            log_var = self.reconstruction_log_var(h)
            # print(log_var) # ----------------------------------------------------------------
            # print(log_var.size())
            return mu, log_var

    def classify(self, x, logits=False):
        for layer in self.cls_hidden:
            x = F.relu(layer(x))
        lg = self.logits(x)
        if logits:
            return lg
        logprobs = F.log_softmax(lg, dim=-1)
        return logprobs

    def sample(self, y, z):
        y = y.float()

        if self.use_mnist:
            x = self.decode(y, z)
            return x
        else:
            x_mu, x_log_var = self.decode(y, z)
            return self.reparameterize(x_mu, x_log_var)

    # Calculating ELBO (Both labelled or unlabelled)
    def forward(self, x, y=None, gamma=1, in_batch=True):
        labelled = False if y is None else True

        xs, ys = (x, y)
        # Duplicating samples and generate labels if not labelled
        if not labelled:
            batch_size = xs.size(0)
            ys = torch.from_numpy(np.arange(self.y_dim))
            ys = ys.view(-1, 1).repeat(1, batch_size).view(-1)
            ys = F.one_hot(ys, self.y_dim)
            ys = Variable(ys.float())
            ys = ys.cuda() if xs.is_cuda else ys
            xs = xs.repeat(self.y_dim, 1)

        # Reconstruction
        zs, z_mu, z_log_var = self.encode(xs, ys)
        if self.use_mnist:
            x_theta = self.decode(ys, zs)

            # p(x|y,z)
            loglikelihood = torch.sum(xs * torch.log(x_theta + 1e-8) + (1 - xs)
                                         * torch.log(1 - x_theta + 1e-8), dim=-1)
        else:
            x_mu, x_log_var = self.decode(ys, zs)

            # p(x|y,z)
            px_yz = D.normal.Normal(x_mu, torch.exp(0.5 * x_log_var))
            px_yz = D.independent.Independent(px_yz, 1)
            loglikelihood = px_yz.log_prob(xs)

        # p(y)
        logprior_y = -math.log(self.y_dim)

        # KL(q(z|x,y)||p(z))
        p_z = D.normal.Normal(torch.zeros_like(zs), torch.ones_like(zs))
        p_z = D.independent.Independent(p_z, 1)
        q_z = D.normal.Normal(z_mu, torch.exp(0.5 * z_log_var))
        q_z = D.independent.Independent(q_z, 1)
        kl = D.kl.kl_divergence(q_z, p_z)

        # ELBO : -L(x,y)
        neg_L = loglikelihood + logprior_y - kl

        if labelled:
            return torch.mean(neg_L)

        logits_y = self.classify(x, logits=True)
        prob_y = F.softmax(logits_y, dim=-1)
        logprob_y = F.log_softmax(logits_y, dim=-1)

        neg_L = neg_L.view_as(prob_y.t()).t()

        # H(q(y|x)) and sum over all labels
        H = -torch.sum(torch.mul(prob_y, logprob_y), dim=-1)
        neg_L = torch.sum(torch.mul(prob_y, neg_L), dim=-1)

        # ELBO : -U(x)
        neg_U = gamma * neg_L + H

        if in_batch:
            return torch.mean(neg_U)
        else:
            return neg_U


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels

    def forward(self, input):
        size = int((input.size(1) // self.n_channels)**0.5)
        return input.view(input.size(0), self.n_channels, size, size)


class M2_CNN(nn.Module):
    """ Semi-supervised Generative Model M2 from Kingma's paper with CNN

    """

    def __init__(self, x_channels=1, x_h=28, x_w=28, y_dim=10, z_dim=32):
        super(M2_CNN, self).__init__()

        print("Created M2 MNIST CNN model.")
        self.x_channels = x_channels
        self.x_h = x_h
        self.x_w = x_w
        self.y_dim = y_dim
        self.z_dim = z_dim

        # For encoder - q(z|x,y)
        en_channels = x_channels + y_dim
        # self.en_layers = nn.Sequential(
        #     nn.Conv2d(en_channels, 32, (3, 3), stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (4, 4), stride=1, padding=1),
        #     nn.ReLU(),
        #     Flatten()
        # )

        self.en_layers = nn.Sequential(
            nn.Conv2d(en_channels, 8, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (5, 5), stride=2, padding=0),
            nn.ReLU(),
            Flatten()
        )

        ## output size depends on input image size
        demo_input = torch.ones([1, en_channels, x_h, x_w])
        h_dim = self.en_layers(demo_input).shape[1]
        # print('h_dim: ', h_dim)

        self.mu = nn.Linear(h_dim, z_dim)
        self.log_var = nn.Linear(h_dim, z_dim)

        # For decoder - p(x|y,z)

        self.de_fc = nn.Linear(y_dim + z_dim, h_dim)
        # self.de_layers = nn.Sequential(
        #     UnFlatten(64),
        #     nn.ConvTranspose2d(64, 32, (4, 4), stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 1, (3, 3), stride=1, padding=1),
        #     nn.Sigmoid()
        # )

        self.de_layers = nn.Sequential(
            UnFlatten(32),
            nn.ConvTranspose2d(32, 16, (6, 6), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, (6, 6), stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, (5, 5), stride=1, padding=1),
            nn.Sigmoid()
        )

        ## output size of decoder
        # demo_input = torch.ones([1, h_dim])
        # convTrans_shape = self.de_layers(demo_input).shape
        # print('convTrans_shape: ', convTrans_shape)

        # For classifier - q(y|x)
        self.cls_conv1 = nn.Conv2d(1, 32, 3, 1)
        self.cls_conv2 = nn.Conv2d(32, 64, 3, 1)
        self.cls_fc1 = nn.Linear(9216, 128)
        self.cls_fc2 = nn.Linear(128, y_dim)

    def encode(self, x, y):
        # Add conditions as channel
        y = y.view(-1, self.y_dim, 1, 1).repeat(1, 1, self.x_h, self.x_w)

        h = torch.cat([x, y], dim=1)
        h = self.en_layers(h)
        mu = self.mu(h)
        log_var = F.softplus(self.log_var(h))

        return self.reparameterize(mu, log_var), mu, log_var

    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        if mu.is_cuda:
            eps = eps.cuda()

        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def decode(self, y, z):
        h = torch.cat([y, z], dim=1)
        h = self.de_fc(h)

        return self.de_layers(h)

    def classify(self, x, logits=False):
        x = self.cls_conv1(x)
        x = F.relu(x)
        x = self.cls_conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.cls_fc1(x)
        x = F.relu(x)
        x = self.cls_fc2(x)

        if logits:
            return x

        output = F.log_softmax(x, dim=1)
        return output

    def sample(self, y, z):
        y = y.float()

        x = self.decode(y, z)
        return x

    # Calculating ELBO (Both labelled or unlabelled)
    def forward(self, x, y=None, gamma=1, in_batch=True):
        labelled = False if y is None else True

        xs, ys = (x, y)
        # Duplicating samples and generate labels if not labelled
        if not labelled:
            batch_size = xs.size(0)
            ys = torch.from_numpy(np.arange(self.y_dim))
            ys = ys.view(-1, 1).repeat(1, batch_size).view(-1)
            ys = F.one_hot(ys, self.y_dim)
            ys = Variable(ys.float())
            ys = ys.cuda() if xs.is_cuda else ys
            xs = xs.repeat(self.y_dim, 1, 1, 1)

        # Reconstruction
        zs, z_mu, z_log_var = self.encode(xs, ys)
        x_theta = self.decode(ys, zs)

        # p(x|y,z)
        loglikelihood = torch.sum(
                            torch.flatten(xs * torch.log(x_theta + 1e-8)
                                          + (1 - xs) * torch.log(1 - x_theta + 1e-8),
                                          start_dim=1),
                            dim=-1)

        # p(y)
        logprior_y = -math.log(self.y_dim)

        # KL(q(z|x,y)||p(z))
        p_z = D.normal.Normal(torch.zeros_like(zs), torch.ones_like(zs))
        p_z = D.independent.Independent(p_z, 1)
        q_z = D.normal.Normal(z_mu, torch.exp(0.5 * z_log_var))
        q_z = D.independent.Independent(q_z, 1)
        kl = D.kl.kl_divergence(q_z, p_z)

        # ELBO : -L(x,y)
        neg_L = loglikelihood + logprior_y - kl

        if labelled:
            return torch.mean(neg_L)

        logits_y = self.classify(x, logits=True)
        prob_y = F.softmax(logits_y, dim=1)
        logprob_y = F.log_softmax(logits_y, dim=1)

        neg_L = neg_L.view_as(prob_y.T).T

        # H(q(y|x)) and sum over all labels
        H = -torch.sum(prob_y * logprob_y, dim=-1)
        neg_L = torch.sum(prob_y * neg_L, dim=-1)

        # ELBO : -U(x)
        neg_U = gamma * neg_L + H

        if in_batch:
            return torch.mean(neg_U)
        else:
            return neg_U