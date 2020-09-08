import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from models.M2 import M2, train_one_step, test
from models.acquisitions import acquisition_gamma, acquisition_entropyXdensity, acquisition_ProbEntropyMixDensity, acquisition_EntropyAndDensity, acquisition_entropy, acquisition_random

from utils.dataset import create_halfmoon_dataset, ActiveMoon
from utils.plots import plot_points, plot_halfmoon_acquisitions


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

        # if epoch % 100 == 0:
        #     plt.title('Exp {} AL Loop {} Train Loss'.format(i_exps, current_AL_loop))
        #     plt.plot(train_loss)
        #     plt.show()
        #
        #     plt.title('Exp {} AL Loop {} Test Loss'.format(i_exps, current_AL_loop))
        #     plt.plot(test_loss)
        #     plt.show()
        #
        #     plt.title('Exp {} AL Loop {} Train Accuracy'.format(i_exps, current_AL_loop))
        #     plt.plot(train_accuracy)
        #     plt.show()
        #
        #     plt.title('Exp {} AL Loop {} Test Accuracy'.format(i_exps, current_AL_loop))
        #     plt.plot(test_accuracy)
        #     plt.show()
            # sample_x, sample_y = draw_samples()

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
            # acquired_X, acquired_y, acquired_idx = acquisition_entropyXdensity(model, unlabelled, use_cuda, n_new=acq_size)
            acquired_X, acquired_y, acquired_idx = acquisition_entropy(model, unlabelled, use_cuda, n_new=acq_size, use_M2=True)
        else:
            # acquired_X, acquired_y, acquired_idx = acquisition_gamma(model, unlabelled, use_cuda,
            #                                                         gamma=gamma, n_new=acq_size)
            # acquired_X, acquired_y, acquired_idx = acquisition_ProbEntropyMixDensity(model, unlabelled, use_cuda,
            #                                                                          densityProb=gamma, n_new=acq_size)
            acquired_X, acquired_y, acquired_idx = acquisition_EntropyAndDensity(model, unlabelled, use_cuda,
                                                                                 propo=gamma, entropy_inter_density=False,
                                                                                 n_new=acq_size)

        acquired_Xs.append(acquired_X)
        acquired_ys.append(acquired_y)
        acquired_idxs.append(acquired_idx)

    return acquired_Xs, acquired_ys, acquired_idxs, test_accuracy[-1]


def one_AL_experiment(i_exps, max_acq_n, max_epochs, labels_per_class, gamma, datasets, gammas_plot=None, seed=None):
    """ One Active Learning Experiment with a given Gamma

    :param gamma:           The gamma for acquisition
    :param gammas_plot:     THe list of gamma for plotting in each step
    """

    AL_data = ActiveMoon(datasets, use_cuda, valid_ratio=valid_ratio, batch_size=64,
                         labels_per_class=labels_per_class, seed=78)
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
        print("##################### Exp {} with Gamma={} - AL Loop {} #######################".format(
              i_exps, gamma, n_AL_loop))

        labelled, unlabelled = AL_data.get_next_train(acquired_idx)
        X_train, X_test, y_train, y_test, x_lab, y_lab, u_unlab, v_unlab = AL_data.get_xyuv_for_printing(labelled,
                                                                                                         unlabelled)

        # alpha = 0.1 * len(unlabelled) / len(labelled)

        alpha = 0.1 * current_size
        # alpha = 0 # --------------------------------------------------------------------------------------------

        torch.manual_seed(seed)
        model = M2(x_dim, y_dim, z_dim, h_en_dim, h_de_dim, h_cls_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
        if use_cuda:
            model = model.cuda()

        if gammas_plot is None:
            acquired_Xs, acquired_ys, acquired_idxs, test_accuracy = train_each_AL_loop(
                i_exps, n_AL_loop, max_epochs, model, alpha, optimizer,
                labelled, unlabelled, validation, acq_size,
                use_cuda, [gamma])
            # Pick the acquisition data index
            acquired_idx = acquired_idxs[0]
            full_acq_Xs.extend(acquired_Xs[0])
            full_acq_ys.extend(acquired_ys[0])

        else:
            acquired_Xs, acquired_ys, acquired_idxs, test_accuracy = train_each_AL_loop(
                i_exps, n_AL_loop, max_epochs, model, alpha, optimizer,
                labelled, unlabelled, validation, acq_size,
                use_cuda, gammas_plot)
            # Pick the acquisition data index and plot the others
            idx, = np.where(gammas_plot == gamma)
            acquired_idx = acquired_idxs[idx[0]]
            full_acq_Xs.extend(acquired_Xs[idx[0]])
            full_acq_ys.extend(acquired_ys[idx[0]])


        test_accuracy_list.append(test_accuracy)
        training_size_list.append(current_size)

        # ---------------------------- Plot Accuracy --------------------------
        if n_AL_loop % 10 == 0:
            plt.title('Test Accuracy')
            plt.plot(training_size_list, test_accuracy_list)
            plt.ylabel('Accuracy')
            plt.xlabel('Number of Labelled data')
            plt.show()

        # ---------------------------- Plot boundary --------------------------
        h = .01  # step size in the mesh
        x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - .5, max(X_train[:, 0].max(),
                                                                              X_test[:, 0].max()) + .5
        y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - .5, max(X_train[:, 1].max(),
                                                                              X_test[:, 1].max()) + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        if gammas_plot is not None:
            gamma_n = len(gammas_plot)
            plot_lw = 6
            # figure = plt.figure(figsize=(gamma_n * plot_lw, 2 * plot_lw))
            figure = plt.figure(figsize=(2 * plot_lw, gamma_n * plot_lw))

            # ==================== Plot q(y|x) ==================
            model.eval()
            boundary = np.c_[xx.ravel(), yy.ravel()]
            boundary = torch.from_numpy(boundary)
            boundary = Variable(boundary.float())
            if use_cuda:
                boundary = boundary.cuda()
            boundary_prob_y = model.classify(boundary)
            boundary_y = boundary_prob_y.data.cpu().numpy()[:, 1]

            # Put the result into a color plot
            boundary_y = boundary_y.reshape(xx.shape)

            for i in range(gamma_n):
                # ax = plt.subplot(2, gamma_n, i+1)
                ax = plt.subplot(gamma_n, 2, 2 * i + 1)

                title = "Gamma={:.3f}".format(gammas_plot[i])
                plot_points(ax, x_lab, y_lab, u_unlab, v_unlab, xx, yy,
                            X_test, y_test, acquired_Xs[i], acquired_ys[i],
                            title=title, boundary_y=boundary_y)

            # ==================== Plot Acquisition Bound with different Gamma ==================
            for i in range(gamma_n):
                # ax = plt.subplot(2, gamma_n, gamma_n+i+1)
                ax = plt.subplot(gamma_n, 2, 2 * i + 2)

                boundary_acq_bound = model(boundary, gamma=gammas_plot[i], in_batch=False)
                boundary_acq_bound = boundary_acq_bound.data.cpu().numpy()
                # Put the result into a color plot
                boundary_acq_bound = boundary_acq_bound.reshape(xx.shape)

                # title="Acquisition boundary with gamma={}".format(gammas_plot[i])
                plot_points(ax, x_lab, y_lab, u_unlab, v_unlab, xx, yy,
                            X_test, y_test, acquired_Xs[i], acquired_ys[i], title=None,
                            boundary_y=boundary_acq_bound, acq_bound_cm=True)

            plt.tight_layout()
            plt.show()
            # ===============================================================

        if acquired_idx is not None:
            current_size += len(acquired_idx)
        if acquired_idx is None or n_AL_loop == max_acq_n:
            finished = True

            # ---------------------------- Plot Final Acquisition --------------------------
            full_acq_Xs = np.asarray(full_acq_Xs).reshape((-1, 2))
            full_acq_ys = np.asarray(full_acq_ys)
            plot_halfmoon_acquisitions(x_lab, y_lab, u_unlab, v_unlab, xx, yy,
                                       X_test, y_test, full_acq_Xs, full_acq_ys, gamma)
        n_AL_loop += 1

    return test_accuracy_list, training_size_list


# *******************************************************


use_cuda = torch.cuda.is_available()
torch.backends.cudnn.deterministic = True

# Model Hyper-Params ----
x_dim = 2
y_dim = 2       # Number of classes
z_dim = 2
h_en_dim = [8, 16, 8]
h_de_dim = [8, 16, 8]
h_cls_dim = [8, 16, 8, 4]

n_samples = 1000
labels_per_class = 1
valid_ratio = .4
training_data_size = n_samples * (1 - valid_ratio)


# AL Params ------------
max_acq_n = 30 # Max num of data points to be acquired
max_epochs = 500



# Experiments Loop ----------
i_exps = 0
all_exp_tests = []
all_exp_train_sizes = []
all_exp_last_test = []

# seeds = [0,1,2,3,4]
seeds = [5,6,7,8,9]

datasets = create_halfmoon_dataset(n_samples=n_samples, noise=0.2, random_state=0, to_plot=True)

for seed in seeds:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # gamma_n = 11
    # gammas = np.linspace(0, 1, gamma_n)
    gammas = [None]
    for gamma in gammas:
        i_exps += 1
        one_exp_tests, one_exp_train_sizes = one_AL_experiment(i_exps, max_acq_n, max_epochs, labels_per_class, gamma, datasets, seed=seed)

        all_exp_tests.append(one_exp_tests)
        all_exp_train_sizes.append(one_exp_train_sizes)
        all_exp_last_test.append(one_exp_tests[-1])

print(all_exp_last_test)
print(all_exp_tests)