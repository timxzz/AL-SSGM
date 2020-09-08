import torch
import numpy as np
from itertools import cycle
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils.dataset import create_halfmoon_dataset, ActiveMoon, create_mnist_mnist, ActiveMNIST
from utils.plots import plot_halfmoon_acquisitions
from models.acquisitions import acquisition_entropy

import matplotlib.pyplot as plt


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


class SimpleMNISTCNN(nn.Module):
    """ Simple CNN classifier for MNIST
    """

    def __init__(self, x_dim=784, y_dim=10):
        super(SimpleMNISTCNN, self).__init__()

        print("Created Simple MNIST CNN model.")
        self.x_dim = x_dim
        self.y_dim = y_dim

        # For classifier - q(y|x)
        self.cls_conv1 = nn.Conv2d(1, 32, 3, 1)
        self.cls_conv2 = nn.Conv2d(32, 64, 3, 1)
        self.cls_fc1 = nn.Linear(9216, 128)
        self.cls_fc2 = nn.Linear(128, y_dim)

    def forward(self, x, logits=False):
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

    # print()
    # t1 = time.time()
    # dtime = t1 - t0
    # display.clear_output(wait=False)
    m = len(labelled)
    # if epoch % 10 == 0:
        # print("Exp {} - Active Learning Loop number: {}, Epoch: {}, \t Time per Epoch: {:.2f} s".format(
        #         i_exps, current_AL_loop, epoch, dtime))
    if epoch % 20 == 1:
        print("[Train]\t\t J: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


def _test(epoch, model, validation, use_cuda):

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
    if epoch % 20 == 1:
        print("[Validation]\t J: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


def train_each_loop(i_exps, n_AL_loop, max_epochs, model, optimizer,
                       labelled, unlabelled, validation, use_cuda, acq_size):
    """Single Loop for training one model at a given Active Learning dataset"""
    train_loss = []
    test_loss = []

    train_accuracy = []
    test_accuracy = []

    # Hyper-parameter for convergence check
    early_stopping_patience = 7
    early_stopping_counter = 0
    best_test_loss = None


    # Train the new model till finish
    for epoch in range(1, max_epochs + 1):

        # # Normal MNIST baseline model ---------------
        train_l, train_acc = _train(epoch, model, optimizer, labelled, use_cuda)
        test_l, test_acc = _test(epoch, model, validation, use_cuda)

        train_loss.append(train_l)
        test_loss.append(test_l)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

        # if epoch % 100 == 0:
        #     plt.title('Exp {} AL Loop {} Train Loss'.format(i_exps, n_AL_loop))
        #     plt.plot(train_loss)
        #     plt.show()
        #
        #     plt.title('Exp {} AL Loop {} Test Loss'.format(i_exps, n_AL_loop))
        #     plt.plot(test_loss)
        #     plt.show()
        #
        #     plt.title('Exp {} AL Loop {} Train Accuracy'.format(i_exps, n_AL_loop))
        #     plt.plot(train_accuracy)
        #     plt.show()
        #
        #     plt.title('Exp {} AL Loop {} Test Accuracy'.format(i_exps, n_AL_loop))
        #     plt.plot(test_accuracy)
        #     plt.show()

        # Convergence Check
        if best_test_loss is None:
            best_test_loss = test_l
        elif test_l > best_test_loss:
            early_stopping_counter += 1
        else:
            best_test_loss = test_l
            early_stopping_counter = 0

        if early_stopping_counter >= early_stopping_patience:
            print('* Exp {} AL Loop {} Epoch {} EarlyStopping *'.format(i_exps, n_AL_loop, epoch))
            break

    # Data acquisition
    acquired_X, acquired_y, acquired_idx = acquisition_entropy(model, unlabelled, use_cuda, n_new=acq_size)

    return acquired_X, acquired_y, acquired_idx, test_accuracy[-1]


def one_AL_experiment(i_exps, max_acq_n, max_epochs, labels_per_class, datasets, seed):
    """ One Active Learning Experiment with entropy
    """

    AL_data = ActiveMoon(datasets, use_cuda, valid_ratio=valid_ratio, batch_size=64,
                         labels_per_class=labels_per_class, seed=78)  # Fix seed to fix initial labelling
    validation = AL_data.get_valid()

    # acq_size = labels_per_class * 2
    acq_size = 1

    finished = False
    acquired_idx = None
    current_size = labels_per_class * 2
    n_AL_loop = 0
    test_accuracy_list = []
    training_size_list = []
    full_acq_Xs = []
    full_acq_ys = []

    # Start Active Learning Loop
    while not finished:
        print("##################### Exp {} with Entropy - AL Loop {} #######################".format(
              i_exps, n_AL_loop))

        labelled, unlabelled = AL_data.get_next_train(acquired_idx)
        X_train, X_test, y_train, y_test, x_lab, y_lab, u_unlab, v_unlab = AL_data.get_xyuv_for_printing(labelled,
                                                                                                         unlabelled)

        # alpha = 0.1 * len(unlabelled) / len(labelled)

        alpha = 0.1 * current_size
        # alpha = 0 # --------------------------------------------------------------------------------------------

        torch.manual_seed(seed)
        model = SimpleHalMoonNN(x_dim, y_dim, h_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
        if use_cuda:
            model = model.cuda()

        acquired_X, acquired_y, acquired_idx, test_accuracy = train_each_loop(
            i_exps, n_AL_loop, max_epochs, model, optimizer,
            labelled, unlabelled, validation,
            use_cuda, acq_size)
        # Pick the acquisition data index
        acquired_idx = acquired_idx
        full_acq_Xs.extend(acquired_X)
        full_acq_ys.extend(acquired_y)

        test_accuracy_list.append(test_accuracy)
        training_size_list.append(current_size)


        # ---------------------------- Plot boundary --------------------------
        h = .01  # step size in the mesh
        x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - .5, max(X_train[:, 0].max(),
                                                                              X_test[:, 0].max()) + .5
        y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - .5, max(X_train[:, 1].max(),
                                                                              X_test[:, 1].max()) + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # if gammas_plot is not None:
        #     gamma_n = len(gammas_plot)
        #     plot_lw = 6
        #     # figure = plt.figure(figsize=(gamma_n * plot_lw, 2 * plot_lw))
        #     figure = plt.figure(figsize=(2 * plot_lw, gamma_n * plot_lw))
        #
        #     # ==================== Plot q(y|x) ==================
        #     model.eval()
        #     boundary = np.c_[xx.ravel(), yy.ravel()]
        #     boundary = torch.from_numpy(boundary)
        #     boundary = Variable(boundary.float())
        #     if use_cuda:
        #         boundary = boundary.cuda()
        #     boundary_prob_y = model.classify(boundary)
        #     boundary_y = boundary_prob_y.data.cpu().numpy()[:, 1]
        #
        #     # Put the result into a color plot
        #     boundary_y = boundary_y.reshape(xx.shape)
        #
        #     for i in range(gamma_n):
        #         # ax = plt.subplot(2, gamma_n, i+1)
        #         ax = plt.subplot(gamma_n, 2, 2 * i + 1)
        #
        #         title = "Gamma={:.3f}".format(gammas_plot[i])
        #         plot_points(ax, x_lab, y_lab, u_unlab, v_unlab, xx, yy,
        #                     X_test, y_test, acquired_Xs[i], acquired_ys[i],
        #                     title=title, boundary_y=boundary_y)
        #
        #     # ==================== Plot Acquisition Bound with different Gamma ==================
        #     for i in range(gamma_n):
        #         # ax = plt.subplot(2, gamma_n, gamma_n+i+1)
        #         ax = plt.subplot(gamma_n, 2, 2 * i + 2)
        #
        #         boundary_acq_bound = model(boundary, gamma=gammas_plot[i], in_batch=False)
        #         boundary_acq_bound = boundary_acq_bound.data.cpu().numpy()
        #         # Put the result into a color plot
        #         boundary_acq_bound = boundary_acq_bound.reshape(xx.shape)
        #
        #         # title="Acquisition boundary with gamma={}".format(gammas_plot[i])
        #         plot_points(ax, x_lab, y_lab, u_unlab, v_unlab, xx, yy,
        #                     X_test, y_test, acquired_Xs[i], acquired_ys[i], title=None,
        #                     boundary_y=boundary_acq_bound, acq_bound_cm=True)
        #
        #     plt.tight_layout()
        #     plt.show()
        #     # ===============================================================

        if acquired_idx is not None:
            current_size += len(acquired_idx)
        if acquired_idx is None or n_AL_loop == max_acq_n:
            finished = True

            # ---------------------------- Plot Final Acquisition --------------------------
            full_acq_Xs = np.asarray(full_acq_Xs).reshape((-1, 2))
            full_acq_ys = np.asarray(full_acq_ys)
            plot_halfmoon_acquisitions(x_lab, y_lab, u_unlab, v_unlab, xx, yy,
                                       X_test, y_test, full_acq_Xs, full_acq_ys)
        n_AL_loop += 1

        # ---------------------------- Plot Accuracy --------------------------
        if n_AL_loop % 10 == 0:
            plt.title('Test Accuracy')
            plt.plot(training_size_list, test_accuracy_list)
            plt.ylabel('Accuracy')
            plt.xlabel('Number of Labelled data')
            plt.show()

    return test_accuracy_list, training_size_list



# *******************************************************


use_cuda = torch.cuda.is_available()

torch.backends.cudnn.deterministic = True


# *******************************************************
#                       Half-Moon
# *******************************************************


# Model Hyper-Params ----
x_dim = 2
y_dim = 2       # Number of classes
h_dim = [8, 16, 8, 4]

n_samples = 1000
labels_per_class = 1
valid_ratio = .4
training_data_size = n_samples * (1 - valid_ratio)

max_acq_n = 30
max_epochs = 3000

all_seed_exp_tests = []
seeds = [0,1,2,3,4,5,6,7,8,9]
i_exp = 0

for seed in seeds:
    i_exp += 1

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    datasets = create_halfmoon_dataset(n_samples=n_samples, noise=0.2, random_state=0, to_plot=True)
    AL_data = ActiveMoon(datasets, use_cuda, valid_ratio=valid_ratio, batch_size=64, labels_per_class=labels_per_class, seed=78)

    validation = AL_data.get_valid()
    labelled, unlabelled = AL_data.get_next_train()
    # X_train, X_test, y_train, y_test, x_lab, y_lab, u_unlab, v_unlab = AL_data.get_xyuv_for_printing(labelled,unlabelled)

    one_exp_tests, one_exp_train_sizes = one_AL_experiment(i_exp, max_acq_n, max_epochs, labels_per_class, datasets, seed)

    all_seed_exp_tests.append(one_exp_tests)


print(all_seed_exp_tests)

# all_exp_last_test.append(test_acc)
#
#
# plt.title('Lambda vs Test Acc')
# plt.plot(all_exp_last_test)
# plt.show()


# *******************************************************
#                       MNIST
# *******************************************************
#
#
# # Model Hyper-Params ----
# num_class = 2
#
# x_dim = 784
# y_dim = num_class       # Number of classes
#
# n_samples = 1000
# labels_per_class = 1
# valid_ratio = .4
# training_data_size = n_samples * (1 - valid_ratio)
#
# max_acq_n = 30
# max_epochs = 3000
#
# all_seed_exp_tests = []
# seeds = [0,1,2,3,4,5,6,7,8,9]
# i_exp = 0
#
# mnist_train, mnist_valid = create_mnist_mnist(location="../data", n_labels=num_class, use_cnn=True)
#
# for seed in seeds:
#     i_exp += 1
#
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     AL_data = ActiveMNIST(mnist_train, mnist_valid, use_cuda,
#                           batch_size=128, labels_per_class=labels_per_class, n_labels=num_class, seed=78)
#
#     validation = AL_data.get_valid()
#     labelled, unlabelled = AL_data.get_next_train()
#     # X_train, X_test, y_train, y_test, x_lab, y_lab, u_unlab, v_unlab = AL_data.get_xyuv_for_printing(labelled,unlabelled)
#
#     one_exp_tests, one_exp_train_sizes = one_AL_experiment(i_exp, max_acq_n, max_epochs, labels_per_class, datasets, seed)
#
#     all_seed_exp_tests.append(one_exp_tests)
#
#
# print(all_seed_exp_tests)
#

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