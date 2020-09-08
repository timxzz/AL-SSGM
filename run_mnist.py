import numpy as np
import torch

import matplotlib.pyplot as plt

from models.M2 import M2, M2_CNN, train_one_step, test
from utils.dataset import create_mnist_mnist, ActiveMNIST
from models.acquisitions import acquisition_gamma, acquisition_entropy, acquisition_random
from utils.plots import draw_mnist_samples




def train_each_AL_loop(i_exps, current_AL_loop, max_epochs, model, alpha, optimizer,
                       labelled, unlabelled, validation, acq_size,
                       use_cuda, gamma_list):
    """Single Loop for training one model at a given Active Learning dataset"""
    train_loss = []
    test_loss = []

    train_accuracy = []
    test_accuracy = []

    acquired_Xs = []
    acquired_ys = []
    acquired_idxs = []

    # Hyper-parameter for convergence check
    early_stopping_patience = 7
    early_stopping_counter = 0
    best_test_loss = None


    # Train the new model till finish
    for epoch in range(1, max_epochs + 1):
        train_l, train_acc = train_one_step(i_exps, current_AL_loop, epoch, model, alpha, optimizer,
                                            labelled, unlabelled, use_cuda)
        test_l, test_acc = test(i_exps, current_AL_loop, epoch, model, alpha, validation, use_cuda)

        train_loss.append(train_l)
        test_loss.append(test_l)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

        if epoch % 100 == 0:
            plt.title('Exp {} AL Loop {} Train Loss'.format(i_exps, current_AL_loop))
            plt.plot(train_loss)
            plt.show()

            plt.title('Exp {} AL Loop {} Test Loss'.format(i_exps, current_AL_loop))
            plt.plot(test_loss)
            plt.show()

            plt.title('Exp {} AL Loop {} Train Accuracy'.format(i_exps, current_AL_loop))
            plt.plot(train_accuracy)
            plt.show()

            plt.title('Exp {} AL Loop {} Test Accuracy'.format(i_exps, current_AL_loop))
            plt.plot(test_accuracy)
            plt.show()
            # sample_x, sample_y = draw_samples()
        #draw_mnist_samples(model, n=10, d=7, n_label=10, use_cuda=use_cuda)

        # Convergence Check
        if best_test_loss is None:
            best_test_loss = test_l
        elif test_l > best_test_loss:
            early_stopping_counter += 1
        else:
            best_test_loss = test_l
            early_stopping_counter = 0

        if early_stopping_counter >= early_stopping_patience:
            print('* Exp {} AL Loop {} Epoch {} EarlyStopping *'.format(i_exps, current_AL_loop, epoch))
            break

    # Data acquisition
    # acquired_X, acquired_y, acquired_idx = acquisition_random(model, unlabelled, n_new=acq_size)
    for gamma in gamma_list:
        if gamma is None:
            acquired_X, acquired_y, acquired_idx = acquisition_random(model, unlabelled, n_new=acq_size)
            # acquired_X, acquired_y, acquired_idx = acquisition_entropy(model, unlabelled, use_cuda, n_new=acq_size,
            #                                                            use_M2=True)
        else:
            acquired_X, acquired_y, acquired_idx = acquisition_gamma(model, unlabelled, use_cuda,
                                                                    gamma=gamma, n_new=acq_size)
        acquired_Xs.append(acquired_X)
        acquired_ys.append(acquired_y)
        acquired_idxs.append(acquired_idx)

    return acquired_Xs, acquired_ys, acquired_idxs, test_accuracy[-1]


def one_AL_experiment(i_exps, max_acq_n, max_epochs, labels_per_class, gamma, mnist_train, mnist_valid,
                      num_class=10, batch_size=64, use_cuda=False, use_cnn=False, seed=None):
    """ One Active Learning Experiment with a given Gamma

    :param gamma:           The gamma for acquisition
    :param gammas_plot:     THe list of gamma for plotting in each step
    """

    AL_data = ActiveMNIST(mnist_train, mnist_valid, use_cuda,
                          batch_size=batch_size, labels_per_class=labels_per_class, n_labels=num_class, seed=78)
    validation = AL_data.get_valid()

    # acq_size = labels_per_class * 2
    acq_size = 1

    finished = False
    acquired_idx = None
    current_size = labels_per_class * num_class
    n_AL_loop = 0
    test_accuracy_list = []
    training_size_list = []

    # Start Active Learning Loop
    while not finished:
        print("##################### Exp {} with Gamma={} - AL Loop {} #######################".format(
              i_exps, gamma, n_AL_loop))

        labelled, unlabelled = AL_data.get_next_train(acquired_idx)

        # alpha = 0.1 * len(unlabelled) / len(labelled)
        alpha = 0.1 * current_size
        # alpha = 0 # --------------------------------------------------------------------------------------------

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if use_cnn:
            model = M2_CNN(x_channels=1, x_h=28, x_w=28, y_dim=y_dim, z_dim=z_dim)
        else:
            model = M2(x_dim, y_dim, z_dim, h_en_dim, h_de_dim, h_cls_dim, use_mnist=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
        if use_cuda:
            model = model.cuda()

        acquired_Xs, acquired_ys, acquired_idxs, test_accuracy = train_each_AL_loop(
            i_exps, n_AL_loop, max_epochs, model, alpha, optimizer,
            labelled, unlabelled, validation, acq_size,
            use_cuda, [gamma])
        # Pick the acquisition data index
        acquired_idx = acquired_idxs[0]

        test_accuracy_list.append(test_accuracy)
        training_size_list.append(current_size)

        # ---------------------------- Plot Accuracy --------------------------
        if n_AL_loop % 10 == 0:
            plt.title('Test Accuracy')
            plt.plot(training_size_list, test_accuracy_list)
            plt.ylabel('Accuracy')
            plt.xlabel('Number of Labelled data')
            plt.show()

        if acquired_idx is not None:
            current_size += len(acquired_idx)
        if acquired_idx is None or n_AL_loop == max_acq_n:
            finished = True

        n_AL_loop += 1
        # ---------------------------- Plot Samples --------------------------
        draw_mnist_samples(model, n=10, d=7, n_label=num_class, use_cuda=use_cuda)

    return test_accuracy_list, training_size_list


# *******************************************************


use_cuda = torch.cuda.is_available()

torch.backends.cudnn.deterministic = True

num_class = 2
# Model Hyper-Params ----
x_dim = 784
y_dim = num_class
z_dim = 32
h_en_dim = [256, 128]
h_de_dim = [128, 256]
h_cls_dim = [256]

batch_size = 128
labels_per_class = 1
use_cnn=True


# AL Params ------------
max_acq_n = 1  # Max num of data points to be acquired
max_epochs = 100


seeds = [0,1,2,3,4]
# seeds = [5,6,7,8,9]

# Experiments Loop ----------

all_exp_tests = []
all_exp_train_sizes = []
all_exp_last_test = []

mnist_train, mnist_valid = create_mnist_mnist(location="../data", n_labels=num_class, use_cnn=use_cnn)

i_exps = 0

for seed in seeds:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # gamma_n = 7
    # gammas = np.linspace(0, 1, gamma_n)
    gammas = [None]
    for gamma in gammas:
        i_exps += 1
        one_exp_tests, one_exp_train_sizes = one_AL_experiment(i_exps, max_acq_n, max_epochs, labels_per_class,
                                                               gamma, mnist_train, mnist_valid,
                                                               num_class=num_class, batch_size=batch_size,
                                                               use_cuda=use_cuda, use_cnn=use_cnn, seed=seed)

        all_exp_tests.append(one_exp_tests)
        all_exp_train_sizes.append(one_exp_train_sizes)
        all_exp_last_test.append(one_exp_tests[-1])

print(all_exp_last_test)
print(all_exp_tests)


# batch_size=128
# epochs=10
# use_cuda = torch.cuda.is_available()
# seed=1
# log_interval=10
#
# torch.manual_seed(seed)
#
# device = torch.device("cuda" if use_cuda else "cpu")
#
# x_dim=784
# y_dim=10
# z_dim=32
# h_en_dim=[256, 128]
# h_de_dim=[128, 256]
# h_cls_dim=[256]
#
# model = M2(x_dim, y_dim, z_dim, h_en_dim, h_de_dim, h_cls_dim, use_mnist=True)
#
#
# # Only use 10 labelled examples per class
# # The rest of the data is unlabelled.
# mnist_train, mnist_valid = create_mnist_mnist(location="../data", n_labels=10)
# AL_data = ActiveMNIST(mnist_train, mnist_valid, use_cuda, batch_size, labels_per_class=10, n_labels=10)
#
# labelled, unlabelled = AL_data.get_next_train()
# validation = AL_data.get_valid()
#
# alpha = 0.1 * len(unlabelled) / len(labelled)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
# if use_cuda:
#     model = model.cuda()


# def train(epoch):
#     model.train()
#     total_loss, accuracy = (0, 0)
#     for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
#         optimizer.zero_grad()
#         # Wrap in variables
#         x, y, u = Variable(x), Variable(y), Variable(u)
#
#         if use_cuda:
#             # They need to be on the same device and be synchronized.
#             x, y = x.cuda(device=0), y.cuda(device=0)
#             u = u.cuda(device=0)
#
#         L = -model(x, y)
#         U = -model(u)
#
#         # Add auxiliary classification loss q(y|x)
#         prob_y = model.classify(x)
#
#         # cross entropy
#         classication_loss = -torch.sum(y * torch.log(prob_y + 1e-8), dim=1).mean()
#
#         J_alpha = L + U + alpha * classication_loss
#
#         J_alpha.backward()
#         optimizer.step()
#
#         total_loss += J_alpha.item()
#         accuracy += torch.mean((torch.max(prob_y, 1)[1].data == torch.max(y, 1)[1].data).float()).item()
#
#     # display.clear_output(wait=False)
#     m = len(unlabelled)
#     print("Epoch: {}".format(epoch))
#     print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
#     return total_loss / m, accuracy / m
#
#
# def test(epoch):
#     model.eval()
#
#     total_loss, accuracy = (0, 0)
#
#     for x, y in validation:
#         x, y = Variable(x), Variable(y)
#
#         if use_cuda:
#             x, y = x.cuda(device=0), y.cuda(device=0)
#
#         L = -model(x, y)
#         U = -model(x)
#
#         prob_y = model.classify(x)
#         classication_loss = -torch.sum(y * torch.log(prob_y + 1e-8), dim=1).mean()
#
#         J_alpha = L + U + alpha * classication_loss
#
#         total_loss += J_alpha.item()
#
#         _, pred_idx = torch.max(prob_y, 1)
#         _, lab_idx = torch.max(y, 1)
#         accuracy += torch.mean((pred_idx == lab_idx).float()).item()
#
#     m = len(validation)
#     print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
#     return total_loss / m, accuracy / m


#
# train_loss=[]
# test_loss=[]
#
# train_accuracy=[]
# test_accuracy=[]
#
# for epoch in range(1, epochs + 1):
#     train_l, train_acc = train(epoch)
#     test_l, test_acc = test(epoch)
#
#     train_loss.append(train_l)
#     test_loss.append(test_l)
#     train_accuracy.append(train_acc)
#     test_accuracy.append(test_acc)
#
#     plt.title('Train Loss')
#     plt.plot(train_loss)
#     plt.show()
#
#     plt.title('Test Loss')
#     plt.plot(test_loss)
#     plt.show()
#
#     plt.title('Train Accuracy')
#     plt.plot(train_accuracy)
#     plt.show()
#
#     plt.title('Test Accuray')
#     plt.plot(test_accuracy)
#     plt.show()

    # draw_samples()

# train_loss = []
# test_loss = []
#
# train_accuracy = []
# test_accuracy = []
#
# acquired_Xs = []
# acquired_ys = []
# acquired_idxs = []
#
# # Hyper-parameter for convergence check
# early_stopping_patience = 7
# early_stopping_counter = 0
# best_test_loss = None
#
#
# # Train the new model till finish
# for epoch in range(1, epochs + 1):
#
#     train_l, train_acc = train_one_step(1, 1, epoch, model, alpha, optimizer,
#                                         labelled, unlabelled, use_cuda)
#     test_l, test_acc = test(1, 1, epoch, model, alpha, validation, use_cuda)
#
#     train_loss.append(train_l)
#     test_loss.append(test_l)
#     train_accuracy.append(train_acc)
#     test_accuracy.append(test_acc)
#
#     plt.title('Exp {} AL Loop {} Train Loss'.format(1, 1))
#     plt.plot(train_loss)
#     plt.show()
#
#     plt.title('Exp {} AL Loop {} Test Loss'.format(1, 1))
#     plt.plot(test_loss)
#     plt.show()
#
#     plt.title('Exp {} AL Loop {} Train Accuracy'.format(1, 1))
#     plt.plot(train_accuracy)
#     plt.show()
#
#     plt.title('Exp {} AL Loop {} Test Accuracy'.format(1, 1))
#     plt.plot(test_accuracy)
#     plt.show()

