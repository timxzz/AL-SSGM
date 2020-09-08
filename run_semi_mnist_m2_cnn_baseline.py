import torch
from torch.autograd import Variable

import numpy as np

from itertools import cycle
import matplotlib.pyplot as plt

from models.M2 import M2, M2_CNN
from utils.dataset import create_mnist_mnist, ActiveMNIST, onehot



batch_size=128
epochs=10
use_cuda = torch.cuda.is_available()
seed=1
log_interval=10

torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")

y_dim=10
z_dim=32

model = M2_CNN(x_channels=1, x_h=28, x_w=28, y_dim=y_dim, z_dim=z_dim)

mnist_train, mnist_valid = create_mnist_mnist(location="../data", n_labels=10, use_cnn=True)
AL_data = ActiveMNIST(mnist_train, mnist_valid, use_cuda,
                          batch_size=batch_size, labels_per_class=10, n_labels=10, seed=78)

validation = AL_data.get_valid()
labelled, unlabelled = AL_data.get_next_train()

# alpha = 0.1 * len(unlabelled) / len(labelled)
alpha = 0.1 * 10 * 2

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
if use_cuda: model = model.cuda()


def train(epoch):
    model.train()
    total_loss, accuracy = (0, 0)
    batch_id = 0
    for (x, y, _), (u, _, _) in zip(cycle(labelled), unlabelled):
        batch_id += 1
        optimizer.zero_grad()
        # Wrap in variables
        x, y, u = Variable(x), Variable(y), Variable(u)

        if use_cuda:
            # They need to be on the same device and be synchronized.
            x, y = x.cuda(device=0), y.cuda(device=0)
            u = u.cuda(device=0)

        L = -model(x, y)
        U = -model(u)

        # Add auxiliary classification loss q(y|x)
        logprob_y = model.classify(x)

        # cross entropy
        classication_loss = -torch.sum(y * logprob_y, dim=1).mean()

        J_alpha = L + U + alpha * classication_loss

        J_alpha.backward()
        optimizer.step()

        total_loss += J_alpha.item()
        accuracy += torch.mean((torch.max(logprob_y, 1)[1].data == torch.max(y, 1)[1].data).float()).item()

        if batch_id % 10 == 0:
            print(".", end='')
    print()

    # display.clear_output(wait=False)
    m = len(unlabelled)
    print("Epoch: {}".format(epoch))
    print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


def test(epoch):
    model.eval()

    total_loss, accuracy = (0, 0)

    for x, y, _ in validation:
        x, y = Variable(x), Variable(y)

        if use_cuda:
            x, y = x.cuda(device=0), y.cuda(device=0)

        L = -model(x, y)
        U = -model(x)

        logprob_y = model.classify(x)
        classication_loss = -torch.sum(y * logprob_y, dim=1).mean()

        J_alpha = L + U + alpha * classication_loss

        total_loss += J_alpha.item()

        _, pred_idx = torch.max(logprob_y, 1)
        _, lab_idx = torch.max(y, 1)
        accuracy += torch.mean((pred_idx == lab_idx).float()).item()

    m = len(validation)
    print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
    return total_loss / m, accuracy / m


def draw_samples(n=10, d=7):
    model.eval()
    z = Variable(torch.randn(16, 32))

    # Generate a batch of 7s
    y = Variable(onehot(10)(d).repeat(16, 1))

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


train_loss=[]
test_loss=[]

train_accuracy=[]
test_accuracy=[]

for epoch in range(1, epochs + 1):
    train_l, train_acc = train(epoch)
    test_l, test_acc = test(epoch)

    train_loss.append(train_l)
    test_loss.append(test_l)
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)

    plt.title('Train Loss')
    plt.plot(train_loss)
    plt.show()

    plt.title('Test Loss')
    plt.plot(test_loss)
    plt.show()

    plt.title('Train Accuracy')
    plt.plot(train_accuracy)
    plt.show()

    plt.title('Test Accuray')
    plt.plot(test_accuracy)
    plt.show()

    draw_samples()