import torch
import numpy as np
from itertools import cycle
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils.dataset import create_halfmoon_dataset, ActiveMoon, create_mnist_mnist, ActiveMNIST

import matplotlib.pyplot as plt



class SimpleNN(nn.Module):
    """ Simple NN classifier

    """

    def __init__(self, y_dim):
        super(SimpleNN, self).__init__()

        self.y_dim = y_dim

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0)
        self.dropout2 = nn.Dropout2d(0)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, y_dim)

    def forward(self, x, logits=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        if logits:
            return x

        output = F.log_softmax(x, dim=1)
        return output


class SimpleHalMoonNN(nn.Module):
    """ Simple NN classifier for HalfMoon
    """

    def __init__(self, x_dim=784, y_dim=10, h_dim=[256, 128]):
        super(SimpleHalMoonNN, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        h_neurons = [x_dim, *h_dim]

        h_linear_layers = [nn.Linear(h_neurons[i - 1], h_neurons[i]) for i in range(1, len(h_neurons))]

        self.cls_hidden = nn.ModuleList(h_linear_layers)
        self.logits = nn.Linear(h_dim[-1], y_dim)

    def forward(self, x, logits=False):
        for layer in self.cls_hidden:
            x = F.relu(layer(x))
        lg = self.logits(x)
        if logits:
            return lg
        logprobs = F.log_softmax(lg, dim=-1)
        return logprobs


def _train(epoch, model, optimizer, labelled, use_cuda):
    y_dim = model.y_dim
    # t0 = time.time()
    model.train()
    total_loss, accuracy = (0, 0)
    batch_id = 0
    for (x, y, l_idx) in labelled:
        batch_id += 1

        # # If y is in onehot format
        # if len(y.shape) != 1:
        #     y = torch.argmax(y, dim=1)

        # If y is not in onehot format
        if len(y.shape) == 1:
            y = F.one_hot(y, num_classes=y_dim)

        optimizer.zero_grad()
        # Wrap in variables
        x, y = Variable(x.float()), Variable(y)

        if use_cuda:
            x, y = x.cuda(), y.cuda()

        logProb_y = model(x)

        # J = F.nll_loss(logProb_y, y)

        ll = torch.sum(y * -logProb_y, dim=-1)
        J = torch.mean(ll)

        J.backward()
        optimizer.step()

        total_loss += J.item()
        # accuracy += torch.mean((torch.argmax(logProb_y, dim=1).data == y.data).float()).item()
        accuracy += torch.mean((torch.max(logProb_y, 1)[1].data == torch.max(y, 1)[1].data).float()).item()

        if batch_id % 10 == 0:
            print(".", end='')

    print()
    # t1 = time.time()
    # dtime = t1 - t0
    # display.clear_output(wait=False)
    m = len(labelled)
    # if epoch % 10 == 0:
        # print("Exp {} - Active Learning Loop number: {}, Epoch: {}, \t Time per Epoch: {:.2f} s".format(
        #         i_exps, current_AL_loop, epoch, dtime))
    print("[Train]\t\t J: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


def _test(model, validation, use_cuda):

    y_dim = model.y_dim
    model.eval()

    total_loss, accuracy = (0, 0)

    for x, y, _ in validation:
        # # If y is in onehot format
        # if len(y.shape) != 1:
        #     y = torch.argmax(y, dim=1)

        # If y is not in onehot format
        if len(y.shape) == 1:
            y = F.one_hot(y, num_classes=y_dim)

        x, y = Variable(x.float()), Variable(y)

        if use_cuda:
            x, y = x.cuda(), y.cuda()

        logProb_y = model(x)

        # J = F.nll_loss(logProb_y, y)

        ll = torch.sum(y * -logProb_y, dim=-1)
        J = torch.mean(ll)

        total_loss += J.item()

        pred_idx = torch.argmax(logProb_y, dim=1)
        lab_idx = torch.argmax(y, dim=1)
        accuracy += torch.mean((pred_idx == lab_idx).float()).item()

    m = len(validation)
    print("[Validation]\t J: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


def train_with_H(epoch, model, alpha, optimizer, labelled, unlabelled, use_cuda):
    y_dim = model.y_dim
    # t0 = time.time()
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

        logProb_y = model(x)

        ll = torch.sum(y * logProb_y, dim=-1)
        L = torch.mean(ll)

        logits_v = model(u, logits=True)
        prob_v = F.softmax(logits_v, dim=1)
        logProb_v = F.log_softmax(logits_v, dim=1)
        neg_H = torch.sum(prob_v * logProb_v, dim=-1)
        U = torch.mean(neg_H)

        J = -(L + alpha * U)

        J.backward()
        optimizer.step()

        total_loss += J.item()
        accuracy += torch.mean((torch.max(logProb_y, 1)[1].data == torch.max(y, 1)[1].data).float()).item()

        if batch_id % 10 == 0:
            print(".", end='')

    print()
    # t1 = time.time()
    # dtime = t1 - t0
    # display.clear_output(wait=False)
    # m = len(labelled)
    # if epoch % 10 == 0:
        # print("Exp {} - Active Learning Loop number: {}, Epoch: {}, \t Time per Epoch: {:.2f} s".format(
        #         i_exps, current_AL_loop, epoch, dtime))
    print("[Train]\t\t J: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


def test_with_H(model, alpha, validation, use_cuda):

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

        logits_y = model(x, logits=True)

        prob_y = F.softmax(logits_y, dim=1)
        logProb_y = F.log_softmax(logits_y, dim=1)

        ll = torch.sum(y * logProb_y, dim=-1)
        L = torch.mean(ll)

        neg_H = torch.sum(prob_y * logProb_y, dim=-1)
        U = torch.mean(neg_H)

        J = -(L + alpha * U)

        total_loss += J.item()

        _, pred_idx = torch.max(prob_y, 1)
        _, lab_idx = torch.max(y, 1)
        accuracy += torch.mean((pred_idx == lab_idx).float()).item()

    m = len(validation)
    print("[Validation]\t J: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


def train_each_alpha_loop(i_exps, max_epochs, model, alpha, optimizer,
                       labelled, unlabelled, validation, use_cuda):
    """Single Loop for training one model at a given Active Learning dataset"""
    train_loss = []
    test_loss = []

    train_accuracy = []
    test_accuracy = []

    # Hyper-parameter for convergence check
    early_stopping_patience = 20
    early_stopping_counter = 0
    best_test_loss = None


    # Train the new model till finish
    for epoch in range(1, max_epochs + 1):

        # Baseline with unlabelled data -------------
        # train_l, train_acc = train_with_H(epoch, model, alpha, optimizer, labelled, unlabelled, use_cuda)
        # test_l, test_acc = test_with_H(model, alpha, validation, use_cuda)

        # # Normal MNIST baseline model ---------------
        train_l, train_acc = _train(epoch, model, optimizer, labelled, use_cuda)
        test_l, test_acc = _test(model, validation, use_cuda)

        train_loss.append(train_l)
        test_loss.append(test_l)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

        if epoch % 100 == 0:
            plt.title('Exp {} Train Loss'.format(i_exps))
            plt.plot(train_loss)
            plt.show()

            plt.title('Exp {} Test Loss'.format(i_exps))
            plt.plot(test_loss)
            plt.show()

            plt.title('Exp {} Train Accuracy'.format(i_exps))
            plt.plot(train_accuracy)
            plt.show()

            plt.title('Exp {} Test Accuracy'.format(i_exps))
            plt.plot(test_accuracy)
            plt.show()

        # Convergence Check
        if best_test_loss is None:
            best_test_loss = test_l
        elif test_l > best_test_loss:
            early_stopping_counter += 1
        else:
            best_test_loss = test_l
            early_stopping_counter = 0

        if early_stopping_counter >= early_stopping_patience:
            print('* Exp {} Epoch {} EarlyStopping *'.format(i_exps, epoch))
            break

    return test_accuracy[-1]


# *******************************************************


use_cuda = torch.cuda.is_available()
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
np.random.seed(seed)

# *******************************************************
#                       Half-Moon
# *******************************************************


# Model Hyper-Params ----
x_dim = 2
y_dim = 2       # Number of classes
h_dim = [8, 16, 8, 4]

# alpha_n = 21
# alphas = np.linspace(0, 2, alpha_n)

alphas = [0,0,0,0]

n_samples = 1000
labels_per_class = 10
valid_ratio = .4
training_data_size = n_samples * (1 - valid_ratio)

max_epochs=500

datasets = create_halfmoon_dataset(n_samples=n_samples, noise=0.1, random_state=0, to_plot=True)
AL_data = ActiveMoon(datasets, use_cuda, valid_ratio=valid_ratio, batch_size=64, labels_per_class=labels_per_class, seed=78)

validation = AL_data.get_valid()
labelled, unlabelled = AL_data.get_next_train()
# X_train, X_test, y_train, y_test, x_lab, y_lab, u_unlab, v_unlab = AL_data.get_xyuv_for_printing(labelled,unlabelled)


all_exp_last_test = []
i_exp = 0
for alpha in alphas:
    i_exp += 1
    torch.manual_seed(seed)
    model = SimpleHalMoonNN(x_dim, y_dim, h_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
    if use_cuda:
        model = model.cuda()
    test_acc = train_each_alpha_loop(i_exp, max_epochs, model, alpha, optimizer, labelled, unlabelled, validation, use_cuda)
    all_exp_last_test.append(test_acc)


plt.title('Lambda vs Test Acc')
plt.plot(alphas, all_exp_last_test)
plt.show()


# *******************************************************
#                       MNIST
# *******************************************************
#
#
# # Model Hyper-Params ----
# num_class = 10
# x_dim = 784
# y_dim = num_class       # Number of classes
# h_dim = [400, 400]
#
# # alpha_n = 21
# # alphas = np.linspace(0, 2, alpha_n)
#
# alphas = [0]
#
# labels_per_class = 100
#
# max_epochs=100
#
# mnist_train, mnist_valid = create_mnist_mnist(location="../data", n_labels=num_class, use_cnn=True)
# AL_data = ActiveMNIST(mnist_train, mnist_valid, use_cuda,
#                       batch_size=128, labels_per_class=labels_per_class, n_labels=num_class, seed=78)
#
# validation = AL_data.get_valid()
# labelled, unlabelled = AL_data.get_next_train()
#
# all_exp_last_test = []
# i_exp = 0
# for alpha in alphas:
#     i_exp += 1
#     model = SimpleNN(num_class)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     if use_cuda:
#         model = model.cuda()
#     test_acc = train_each_alpha_loop(i_exp, max_epochs, model, alpha, optimizer, labelled, unlabelled, validation, use_cuda)
#     all_exp_last_test.append(test_acc)
#
# print(all_exp_last_test)
# print(alphas)
#
# plt.title('Lambda vs Test Acc')
# plt.plot(alphas, all_exp_last_test)
# plt.show()