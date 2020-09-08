import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.autograd import Variable
from utils.dataset import  onehot

def plot_points(ax, x_lab, y_lab, u_unlab, v_unlab, x_grid, y_grid,
                X_test, y_test, acquired_X, acquired_y,
                title=None, boundary_y=None, acq_bound_cm=False):
    """ Plots 2D data points from HalfMoon and the contour for acquisition score
    """

    cm = plt.cm.RdBu
    cm_elbo = plt.cm.Greys
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    cm_new = ListedColormap(['#FFFF00', '#00FFFF'])

    if boundary_y is not None and not acq_bound_cm:
        ax.contourf(x_grid, y_grid, boundary_y, cmap=cm, alpha=.8)
    elif boundary_y is not None and acq_bound_cm:
        ax.contourf(x_grid, y_grid, boundary_y, cmap=cm_elbo, alpha=.8)

    # Plot the training points
    ax.scatter(x_lab[:, 0], x_lab[:, 1], c=y_lab, cmap=cm_bright, linewidths=3,
               edgecolors='k', s=100)
    ax.scatter(u_unlab[:, 0], u_unlab[:, 1], c=v_unlab, cmap=cm_bright,
               edgecolors='k', alpha=0.1, s=50)
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.3,
               edgecolors='k', marker="*", s=50)
    # Plot the acquired points
    ax.scatter(acquired_X[:, 0], acquired_X[:, 1], c=acquired_y, cmap=cm_new, linewidths=3,
               edgecolors='k', s=100)

    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(y_grid.min(), y_grid.max())
    ax.set_xticks(())
    ax.set_yticks(())
    # if ds_cnt == 0:
    if title is not None:
        ax.set_title(title)


def plot_halfmoon_acquisitions(x_lab, y_lab, u_unlab, v_unlab, x_grid, y_grid,
                               X_test, y_test, acquired_X, acquired_y, gamma=None):

    fig, ax = plt.subplots()
    if gamma is not None:
        ax.set_title("Acqusitions with Gamma={:.6f}".format(gamma))

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    cm_new = ListedColormap(['#FFFF00', '#00FFFF'])

    # Plot the training points
    ax.scatter(x_lab[:, 0], x_lab[:, 1], c=y_lab, cmap=cm_bright, linewidths=3,
               edgecolors='k', s=100)
    ax.scatter(u_unlab[:, 0], u_unlab[:, 1], c=v_unlab, cmap=cm_bright,
               edgecolors='k', alpha=0.1, s=50)
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.3,
               edgecolors='k', marker="*", s=50)
    # Plot the acquired points
    ax.scatter(acquired_X[:, 0], acquired_X[:, 1], c=acquired_y, cmap=cm_new, linewidths=3,
               edgecolors='k', s=100)

    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(y_grid.min(), y_grid.max())
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()



def draw_mnist_samples(model, n=10, d=7, n_label=10, use_cuda=False):
    model.eval()
    z = Variable(torch.randn(16, 32))

    # Generate a batch of 7s
    y = Variable(onehot(n_label)(d).repeat(16, 1))

    if use_cuda:
        z, y = z.cuda(device=0), y.cuda(device=0)

    x_mu = model.sample(y, z)
    samples = x_mu.data.view(-1, 28, 28).cpu().numpy()

    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(np.reshape(samples[i], (28, 28)),
                   interpolation="None",
                   cmap='gray')
        plt.axis('off')
    plt.show()